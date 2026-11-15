# Changelog

One version number covers the engine, the API, the web app and the VS Code extension. The extension's own notes are in [vscode/CHANGELOG.md](vscode/CHANGELOG.md).

## 1.0.0

First stable release.

**Stable:** detection, redaction with fail-closed verification, the SHA-256 ledger and dossier export, the `shield` CLI, the MCP server and the VS Code extension. Breaking changes to the `/api/v1` contract, the CLI flags or the MCP tool names will need a new major version.

**Scope:** the web app and API are a single-team tool. One instance keeps jobs in memory, and one shared token protects it. See [SECURITY.md](SECURITY.md) before you put it on a network.

### Security
- A server with no token now answers `/api` and `/mcp` only for localhost `Host` names or those in `SHIELD_ALLOWED_HOSTS`. This blocks DNS rebinding from a web page.
- `SHIELD_API_TOKEN=public` or `demo` no longer turns authentication off. Only `SHIELD_PUBLIC_DEMO=1` opens the site.
- Notes between agents are masked before they are stored, logged or shown.

### Agents
- Agents send each other notes with the new `message_agent` tool. An agent written to after it answered gets one rebuttal turn.
- Crawl agents hand a link into another listed site to that site's agent.

### Crawler
- Reads every link, frame and "next" link inside a listed site and its subdomains, each page once. It used to stop at 20 links per page and fetch pages again.
- Reads the sitemap from `robots.txt`, which reaches pages that no link leads to.
- Finds images in lazy-load attributes, the largest `srcset` entry, `<picture>`, share images, video posters and background styles.
- Reports what it did not read: pages found but not fetched, dead links, blocked pages.
- New setting `SHIELD_CRAWL_MAX_IMAGES_PER_PAGE`; `SHIELD_USER_AGENT` names the crawler.

### Fixes
- The web app no longer treats a refused or missing access token as success.
- `/api/health` works on Windows and reports memory correctly on macOS.
- The `shield` command pointed at a module that did not exist.
- The extension's swarm command was missing from the Command Palette; Windows rescans left stale findings; SARIF export reported success after a failed scan.
- Default AI models come from the engine only, and the web app reads them from `/api/config`.

### Removed
- Unused dependencies: `presidio-analyzer`, `spacy`, `en-core-web-sm`, `anthropic`.
