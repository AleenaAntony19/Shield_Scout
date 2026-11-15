from __future__ import annotations

import asyncio
import html
import os
import re
import time
import zlib
from collections.abc import KeysView
from dataclasses import dataclass, replace
from urllib.parse import urldefrag, urljoin, urlparse

from bs4 import BeautifulSoup

from .models import ImageRef


MIN_IMAGE_WIDTH = 64  # the matcher ignores anything smaller
# shortcut: a long article's later images go unchecked and are counted, raise it with more compute
MAX_IMAGES_PER_PAGE = int(os.getenv("SHIELD_CRAWL_MAX_IMAGES_PER_PAGE", "30"))
RETRYABLE = {"http_403", "http_429"}
RETRY_WAIT = 5.0
IMAGE_RPS = 3.0  # images were unpaced: real sites answer a burst with 403, and the match is then silently lost
MAX_SITEMAPS = 10  # sitemap files read per site: an index can list hundreds
MAX_SITEMAP_PAGES = 5000
IMAGE_LINK = re.compile(r"\.(jpg|jpeg|png|webp|gif|bmp|pdf)(\?|$)", re.I)
# files that are never a page with images on it: fetching them only spends the page budget
NOT_A_PAGE = re.compile(r"\.(css|js|json|xml|rss|atom|zip|gz|tar|7z|rar|exe|dmg|apk|mp3|mp4|avi|mov|webm|woff2?|ttf|ico|svg|docx?|xlsx?|pptx?|csv|txt)(\?|$)", re.I)


@dataclass
class Job:
    url: str
    kind: str = "page"
    depth: int = 0
    domain: str = ""
    page_url: str = ""
    seed_category: str = ""
    risk: float = 0.0
    tier: int = 2
    seq: int = 0
    retries: int = 0
    site: str = ""  # the site agent that owns this job


def classify(soup: BeautifulSoup, seed_category: str = "") -> str:
    if seed_category:
        return {
            "paste": "default",
            "image_host": "default",
            "mirror": "gallery",
        }.get(seed_category, seed_category)
    if soup.select_one("div.thread, a[rel=next]"):
        return "forum"
    if soup.select_one("div.card"):
        return "directory"
    if len(soup.select("img")) >= 20:
        return "gallery"
    return "default"


def _abs(base: str, href: str) -> str:
    return urljoin(base, href)


def _largest(srcset: str) -> str:
    """The biggest candidate of a srcset: the first is often a thumbnail too small to match."""
    best, size = "", -1.0
    for part in srcset.split(","):
        bits = part.split()
        if not bits:
            continue
        try:
            n = float(bits[1][:-1]) if len(bits) > 1 else 0.0
        except ValueError:
            n = 0.0
        if n > size:
            best, size = bits[0], n
    return best


def _image_urls(soup: BeautifulSoup, base: str) -> KeysView[str]:
    out: dict[str, None] = {}  # keeps page order: the main image comes early, and the per-page cap must not cut it
    for tag in soup.select("img, picture source, video[poster]"):
        if str(tag.get("width", "")).isdigit() and int(tag["width"]) < MIN_IMAGE_WIDTH:
            continue  # icons and flags: never a match, and each one costs a rate-limited fetch
        # lazy loaders keep the real address in a data attribute until the page scrolls
        attrs = ("poster",) if tag.name == "video" else ("src", "data-src", "data-lazy-src", "data-original", "srcset", "data-srcset")
        if tag.get("srcset") or tag.get("data-srcset"):
            attrs = ("srcset", "data-srcset")  # src is a smaller copy of the same picture: one fetch, the largest, is enough
        for attr in attrs:
            value = tag.get(attr)
            if not value:
                continue
            out[_abs(base, _largest(value) if attr.endswith("srcset") else value)] = None
    for tag in soup.select('meta[property="og:image"], meta[name="twitter:image"], link[rel~=image_src]'):
        value = tag.get("content") or tag.get("href")  # the share image: often the only full-size copy on the page
        if value:
            out[_abs(base, value)] = None
    for tag in soup.select('[style*="url("]'):
        for value in re.findall(r"url\(\s*['\"]?([^'\")]+)", tag["style"]):
            out[_abs(base, value)] = None
    for a in soup.select("a[href]"):
        href = _abs(base, a["href"])
        if IMAGE_LINK.search(href) and not re.search(r"/wiki/[^/]+:", href):  # a wiki file page in any language, not an image
            out[href] = None
    # vector icons cannot be hashed, and only http(s) can be fetched
    return {u: None for u in out if u.startswith(("http://", "https://")) and not re.search(r"\.svg(\?|$)", u, re.I)}.keys()


# menus, headers and footers: the same links on every page of a site (MediaWiki marks its own with these ids)
CHROME = ["nav", "header", "footer", "aside", "[role=navigation]", "[role=banner]", "[role=contentinfo]",
          "#mw-panel", "#mw-navigation", "#mw-head", "#footer", "#sidebar"]
LINK_TAGS = "a[href], area[href], iframe[src], frame[src], link[rel~=next][href]"


def _page_links(soup: BeautifulSoup, base: str) -> list[str]:
    """Every address on the page that can itself be a page. #a and #b of one page are one page.
    Links in the page's own content come before menu, header and footer links: under a page budget, the
    content is where a leaked copy sits, and every page repeats the same menu."""
    content: dict[str, None] = {}
    chrome: dict[str, None] = {}
    in_chrome = {id(t) for t in soup.select(", ".join(f"{c} {t}" for c in CHROME for t in LINK_TAGS.split(", ")))}
    for tag in soup.select(LINK_TAGS):
        url = urldefrag(_abs(base, tag.get("href") or tag.get("src"))).url
        if url.startswith(("http://", "https://")) and not IMAGE_LINK.search(url) and not NOT_A_PAGE.search(url):
            (chrome if id(tag) in in_chrome else content)[url] = None
    return list(content) + [u for u in chrome if u not in content]


def _within(host: str, domain: str) -> bool:
    return bool(domain) and (host == domain or host.endswith("." + domain))


def _sitemap_text(body: bytes) -> str:
    if body[:2] == b"\x1f\x8b":  # sitemap.xml.gz; the cap stops a decompression bomb
        try:
            body = zlib.decompressobj(31).decompress(body, 20 * 1024 * 1024)
        except zlib.error:
            return ""
    return body.decode("utf-8", "ignore")


class Scheduler:
    def __init__(self, rps_per_domain: float = 1.0, global_concurrency: int = 8, budget_pages: int = 500, max_depth: int = 10):
        self.rps = rps_per_domain
        self.image_gap = 1.0 / max(IMAGE_RPS, rps_per_domain)
        self.max_depth = max_depth
        self.sem = asyncio.Semaphore(global_concurrency)
        self.budget_pages = budget_pages
        self.sites: dict[str, asyncio.PriorityQueue] = {}
        self.site_stats: dict[str, dict] = {}
        self._agents: dict[str, asyncio.Task] = {}
        self._images_paced: set[str] = set()
        self._ctx: tuple | None = None
        self.pending = 0
        self.seq = 0
        self.blocked: set[str] = set()
        # pages_found minus pages is what the crawl knew about and did not read
        self.counters = {"pages": 0, "images": 0, "blocked": 0, "robots_denied": 0, "pages_found": 0}
        self._last: dict[str, float] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._done = asyncio.Event()
        self._seen: set[tuple[str, str, str]] = set()
        self._fetching = 0  # page fetches in flight: they count toward the budget before they land
        self._sitemapped: set[str] = set()
        self._home: dict[str, tuple[str, int, str]] = {}  # domain -> (site agent, its seed depth, category)

    def push(self, tier: int, risk: float, job: Job):
        job.site = job.site or job.domain or urlparse(job.url).netloc
        if not job.retries:  # a retry is the same job again, not a new find
            key = (job.kind, urldefrag(job.url).url, job.page_url if job.kind == "image" else "")
            if key in self._seen:
                return  # pages link back to each other: without this a loop eats the whole page budget
            self._seen.add(key)
            if job.kind == "page":
                self.counters["pages_found"] += 1
                if job.domain:
                    self._home.setdefault(job.domain, (job.site, job.depth, job.seed_category))
        if job.kind == "page" and self.counters["pages"] >= self.budget_pages:
            return  # images are never refused here: the last page inside the budget still gets its images checked
        job.tier = tier
        job.risk = risk
        job.seq = self.seq
        self.seq += 1
        self.pending += 1
        self.sites.setdefault(job.site, asyncio.PriorityQueue()).put_nowait((tier, -risk, job.seq, job))
        self._spawn(job.site)

    def _spawn(self, site: str) -> None:
        """One agent per site, so a slow or rate-limited site never holds up the others."""
        if self._ctx and (site not in self._agents or self._agents[site].done()):
            self._agents[site] = asyncio.create_task(self._site_agent(site))

    async def _pace(self, domain: str, gap: float | None = None):
        lock = self._locks.setdefault(domain, asyncio.Lock())
        async with lock:
            now = time.monotonic()
            last = self._last.get(domain, 0.0)
            wait = (1.0 / self.rps if gap is None else gap) - (now - last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last[domain] = time.monotonic()

    async def _sitemap_pages(self, job: Job, domain: str, fetcher, on_event) -> list[str]:
        """Pages the site lists about itself: a sitemap reaches corners that no link read so far leads to."""
        root = "{0.scheme}://{0.netloc}".format(urlparse(job.url))
        pages: dict[str, None] = {}
        try:
            await self._pace(domain)
            robots = await fetcher.get(root + "/robots.txt")
            todo = [u.decode("ascii", "ignore") for u in re.findall(rb"(?im)^\s*sitemap:\s*(\S+)", robots.body or b"")] or [root + "/sitemap.xml"]
            for _ in range(MAX_SITEMAPS):
                if not todo or len(pages) >= MAX_SITEMAP_PAGES:
                    break
                await self._pace(domain)
                res = await fetcher.get(todo.pop(0))  # the fetcher applies robots.txt and the private-network guard
                if res.blocked_reason or res.status != 200:
                    continue
                text = _sitemap_text(res.body)
                locs = [html.unescape(u) for u in re.findall(r"<loc>\s*([^<\s]+)\s*</loc>", text)]
                if "<sitemapindex" in text:
                    todo += locs
                else:
                    pages.update(dict.fromkeys(locs))
        except asyncio.CancelledError:
            raise
        except Exception:
            pass  # a broken sitemap must not cost the links already found on the page
        if pages:
            await on_event({"kind": "sitemap", "url": root + "/", "site": job.site, "found": len(pages)})
        return list(pages)[:MAX_SITEMAP_PAGES]

    async def _process(self, job: Job, fetcher, on_image, on_event):
        domain = job.domain or (urlparse(job.url).netloc.split(":")[0])
        stats = self.site_stats.setdefault(job.site, {"pages": 0, "images": 0})
        if job.site in self.blocked:
            return

        if job.kind == "image":
            self.counters["images"] += 1
            stats["images"] += 1
            if job.url not in self._images_paced:  # a repeat is answered from memory: no request, so no wait
                self._images_paced.add(job.url)
                await self._pace(urlparse(job.url).netloc, self.image_gap)
            async with self.sem:
                await on_image(ImageRef(job.page_url or job.url, job.url, domain))
            return

        if self.counters["pages"] + self._fetching >= self.budget_pages:
            return  # found, but past the page budget: it stays in pages_found so the report shows the gap
        self._fetching += 1
        try:
            await self._pace(domain)
            async with self.sem:
                res = await fetcher.get(job.url)
        finally:
            self._fetching -= 1

        if res.blocked_reason in RETRYABLE and job.retries < 1:  # often a rate limit, not a ban: back off once before giving up on the site
            await asyncio.sleep(RETRY_WAIT)
            self.push(job.tier, job.risk, replace(job, retries=job.retries + 1))
            return
        if res.blocked_reason:
            if res.blocked_reason == "robots":
                self.counters["robots_denied"] += 1
                await on_event({"kind": "robots_denied", "url": job.url})
            else:
                self.counters["blocked"] += 1
                self.blocked.add(job.site)  # the site, not the host: one refused section must not stop the rest of a host
                await on_event({"kind": "blocked", "url": job.url, "reason": res.blocked_reason})
            return

        if res.body[:5] == b"%PDF-":  # a page URL that is really a PDF: its embedded images go to the matcher
            self.push(job.tier, job.risk, replace(job, kind="image", page_url=job.url))
            self.counters["pages_found"] -= 1  # it was a document, not a page left unread
            return
        if not 200 <= res.status < 300:  # a dead link or a failed connection is not a page that was read
            self.counters["pages_failed"] = self.counters.get("pages_failed", 0) + 1
            await on_event({"kind": "crawl_error", "url": job.url, "error": f"HTTP {res.status}" if res.status else "no response"})
            return
        self.counters["pages"] += 1
        stats["pages"] += 1
        await on_event({"kind": "crawl_progress", **self.counters, "url": job.url, "site": job.site})

        soup = BeautifulSoup(res.body, "html.parser")
        kind = classify(soup, job.seed_category)
        refs = await _gallery_refs(job.url, fetcher, soup) if kind == "gallery" else _image_urls(soup, job.url)
        if kind == "default" and len(refs) > MAX_IMAGES_PER_PAGE:
            self.counters["images_skipped"] = self.counters.get("images_skipped", 0) + len(refs) - MAX_IMAGES_PER_PAGE
            refs = list(refs)[:MAX_IMAGES_PER_PAGE]
        for ref in refs if kind == "default" else sorted(refs):
            self.push(job.tier, job.risk, Job(ref, "image", job.depth, domain, job.url, job.seed_category, site=job.site))

        if job.depth >= self.max_depth:
            return
        links = _page_links(soup, job.url)
        netloc = urlparse(job.url).netloc
        if netloc not in self._sitemapped:
            self._sitemapped.add(netloc)
            links += await self._sitemap_pages(job, domain, fetcher, on_event)
        for url in links:
            host = urlparse(url).hostname or ""
            if urlparse(url).netloc == netloc or _within(host, domain):
                self.push(job.tier, job.risk, Job(url, "page", job.depth + 1, domain, job.url, job.seed_category, site=job.site))
                continue
            # a link into another watched site: tell that site's agent, which reads it under its own follow limit
            home = next((d for d in self._home if _within(host, d)), None)
            if home and self._home[home][0] != job.site:
                site, seed_depth, category = self._home[home]
                known = len(self._seen)
                self.push(job.tier, job.risk, Job(url, "page", seed_depth + 1, home, job.url, category, site=site))
                if len(self._seen) > known:
                    await on_event({"kind": "handoff", "from": job.site, "to": site, "url": url})

    async def _site_agent(self, site: str):
        fetcher, on_image, on_event = self._ctx
        await on_event({"kind": "site_agent", "site": site, "status": "spawned"})
        q = self.sites[site]
        while not q.empty():
            job = q.get_nowait()[-1]
            try:
                await self._process(job, fetcher, on_image, on_event)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await on_event({"kind": "crawl_error", "url": job.url, "error": str(exc)})
            finally:  # one place: every way a job ends must count, or the run never finishes
                self.pending -= 1
                if self.pending <= 0:
                    self._done.set()
        await on_event({"kind": "site_agent", "site": site, "status": "done", **self.site_stats.get(site, {})})

    async def run(self, fetcher, on_image, on_event, timeout: float = 120) -> dict:
        if self.pending == 0:
            return dict(self.counters)
        self._ctx = (fetcher, on_image, on_event)
        for site in list(self.sites):
            self._spawn(site)
        try:
            await asyncio.wait_for(self._done.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            self.counters["timed_out"] = True  # the report must say the crawl stopped early
        finally:
            agents = list(self._agents.values())
            for a in agents:
                a.cancel()
            await asyncio.gather(*agents, return_exceptions=True)
        return dict(self.counters)


async def _gallery_refs(url: str, fetcher, soup: BeautifulSoup) -> set[str]:
    try:
        from playwright.async_api import async_playwright

        async with async_playwright() as p:
            # SHIELD_PLAYWRIGHT_CHANNEL=chrome uses an installed Chrome when the bundled build is missing
            browser = await p.chromium.launch(headless=True, channel=os.getenv("SHIELD_PLAYWRIGHT_CHANNEL") or None)
            page = await browser.new_page()
            await page.route("**/*", fetcher.guard_route)
            await page.goto(url, wait_until="domcontentloaded", timeout=15000)
            last = 0
            for _ in range(15):
                srcs = await page.eval_on_selector_all(
                    "img", "els => els.map(e => e.currentSrc || e.dataset.src || e.src)"
                )
                if len(srcs) == last:
                    break
                last = len(srcs)
                await page.mouse.wheel(0, 2000)
                await page.wait_for_timeout(800)
            srcs = await page.eval_on_selector_all(
                "img", "els => els.map(e => e.currentSrc || e.dataset.src || e.src)"
            )
            await browser.close()
            return {_abs(url, s) for s in srcs if s}
    except Exception:
        return _image_urls(soup, url)

