"""MCP tools + agent swarm. The LLM is replaced by a scripted provider at the HTTP layer (httpx.MockTransport),
so the ReAct loop, native tool-call parsing, and every MCP tool run for real."""
import ast
import asyncio
import json
import secrets
import string
import subprocess

import httpx
import pytest

from shield import llm, verify
from shield.mcp import server
from shield.mcp.tools import SESSIONS, list_findings, open_document, open_workspace, register_session
from shield.models import Entity
from shield.swarm import run_agentic

GROQ = llm.configure(llm.LLMConfig(retries=0), provider="groq", api_key="test-key")


def _reply(content="", calls=None):
    msg = {"content": content}
    if calls:
        msg["tool_calls"] = [{"id": f"c{i}", "type": "function", "function": {"name": n, "arguments": json.dumps(a)}} for i, (n, a) in enumerate(calls)]
    return httpx.Response(200, json={"choices": [{"message": msg}], "usage": {"total_tokens": 100}})


def scripted(decide=None, log=None):
    """Plays each agent: first turn calls its tool(s), next turn returns the JSON verdict."""

    def handler(req: httpx.Request):
        body = json.loads(req.content)
        system, msgs = body["messages"][0]["content"], body["messages"][1:]
        task = json.loads(msgs[0]["content"])
        tools = {t["function"]["name"] for t in body.get("tools", [])} - {"message_agent"}  # this scripted model works alone
        done = msgs[-1]["role"] == "tool"
        findings = task.get("findings") or [{"id": i} for i in task.get("finding_ids", [])]
        ids = [f["id"] if isinstance(f, dict) else f for f in findings]
        if log is not None:
            log.append(system.split(".")[0][:40])
        if "You plan a privacy run" in system:
            if done:
                return _reply(json.dumps({"plan": "spawned", "spawned": ["SovereignPIIAgent", "VisualLikenessAgent"]}))
            if task["kind"] == "workspace":
                return _reply(calls=[("spawn_specialist_agent", {"role": "SecretsAuditor", "goal": "audit", "allowed_tools": ["list_code_findings"]})])
            return _reply(calls=[("spawn_specialist_agent", {"role": "SovereignPIIAgent", "goal": "prove ids", "allowed_tools": ["validate_checksum"]}),
                                 ("spawn_specialist_agent", {"role": "VisualLikenessAgent", "goal": "hash images", "allowed_tools": ["compute_perceptual_pdq"]})])
        if done or not tools:
            if "Synthesize the debate" in system:
                rows = [{"id": i, "type": r["type"]} for i, r in task["debate"].items()]
                want = {d["finding_id"]: d["action"] for d in decide(rows)}
                return _reply(json.dumps({"keep": [{"finding_id": i, "reason": "placeholder"} for i, a in want.items() if a == "keep"],
                                          "actions": [{"finding_id": i, "action": a} for i, a in want.items() if a != "keep"]}))
            if "optimize code" in system:
                return _reply(json.dumps({"proposals": [{"id": f["id"], "action": "move_to_env", "reason": "secret"} for f in findings]}))
            keep = set()
            if "You are PragmaticDev" in system and decide:
                keep = {d["finding_id"] for d in decide(findings) if d["action"] == "keep"}
            return _reply(json.dumps({"opinions": [{"id": i, "stance": "keep" if i in keep else "redact", "reason": "tool evidence"} for i in ids],
                                      "clean": True, "summary": "ok"}))
        first = sorted(tools)[0]
        args = {}
        if first == "validate_checksum":
            sid = next((i for i in ids if i), None)
            args = {"finding_id": sid or "none"}
        return _reply(calls=[(first, args)])

    return httpx.MockTransport(handler)


def test_mcp_validate_checksum_truth(tmp_path):
    s = register_session(tmp_path / "x.txt", {"text": ""}, {"ok": Entity("IN_AADHAAR", "2345 6789 0124", 0, 14, 0.9, "text"),
                                                            "bad": Entity("IN_AADHAAR", "2345 6789 0128", 0, 14, 0.9, "text")}, tmp_path)

    async def call(fid):
        r = await server.call_tool("validate_checksum", {"session_id": s.id, "finding_id": fid})
        return json.loads(r.content[0].text)

    assert asyncio.run(call("ok"))["valid"] is True
    assert asyncio.run(call("bad"))["valid"] is False and asyncio.run(call("bad"))["algorithm"] == "Verhoeff_D5"


def test_binding_verdict_drives_redaction_with_safety_floor(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIELD_MCP_ROOT", str(tmp_path))
    doc = tmp_path / "hr.txt"
    doc.write_text("Aadhaar 2345 6789 0124\nPAN ABCPK1234F\nmail ravi.kumar@acme.in\n")
    sid = open_document(str(doc))["session_id"]

    ids = {f["type"]: f["id"] for f in list_findings(sid)["findings"]}
    real = scripted(lambda fs: [{"finding_id": f["id"], "action": "token", "reason": "x"} for f in fs])

    def handler(req):  # PragmaticDev and the adjudicator both want every finding kept, even PAN, which is never debated
        body = json.loads(req.content)
        system, last = body["messages"][0]["content"], body["messages"][-1]["role"]
        if "You are PragmaticDev" in system and last == "tool":
            return _reply(json.dumps({"opinions": [{"id": i, "stance": "keep", "reason": "x"} for i in ids.values()]}))
        if "Synthesize the debate" in system:
            return _reply(json.dumps({"keep": [{"finding_id": i, "reason": "x"} for i in ids.values()], "actions": []}))
        return real.handle_request(req)

    events = []
    res = asyncio.run(run_agentic(sid, GROQ, events.append, httpx.MockTransport(handler)))
    out = open(res["output"]).read()
    assert "ravi.kumar@acme.in" in out, "the binding 'keep' verdict must be honoured for non-floor findings"
    assert "ABCPK1234F" not in out and "2345 6789 0124" not in out, "floor types are always redacted"
    assert res["clean"] and res["tokens"] > 0
    by = {a["name"]: a for a in res["agents"]}
    assert all(a["steps"] > 0 for a in res["agents"])
    assert {"SovereignPIIAgent", "VisualLikenessAgent"} == {c["name"] for c in by["MetaPlanner"]["children"]}
    assert by["StatutoryAuditor"]["tools"] == ["validate_checksum"] and by["FormatGuard"]["tools"] == ["measure_layout"]
    assert {"PrivacyHawk", "SwarmAdjudicator"} <= set(by), "a challenge starts the debate"
    assert "AdversarialRedTeamAgent" not in by and res["attacks"], "the attack runs as a tool, without a model"
    assert any(e.get("agent") == "AdversarialRedTeamAgent" and e.get("event") == "final" and e["clean"] for e in events)


def test_nothing_contested_is_binding_without_hawk_or_adjudicator(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIELD_MCP_ROOT", str(tmp_path))
    doc = tmp_path / "a.txt"
    doc.write_text("mail ravi.kumar@acme.in\n")
    sid = open_document(str(doc))["session_id"]
    res = asyncio.run(run_agentic(sid, GROQ, transport=scripted(lambda fs: [{"finding_id": f["id"], "action": "token", "reason": "x"} for f in fs])))
    assert not {"PrivacyHawk", "SwarmAdjudicator"} & {a["name"] for a in res["agents"]}
    assert res["fail_safe_defaults"] == 0 and res["clean"] and all(d["action"] != "keep" for d in res["decisions"])
    assert "ravi.kumar@acme.in" not in open(res["output"]).read()


def test_self_healing_replans_after_leak(fx, tmp_path, monkeypatch):
    monkeypatch.setenv("SHIELD_MCP_ROOT", str(fx["pdf"].parent))
    sid = open_document(str(fx["pdf"]))["session_id"]
    real, calls = verify.attack, []

    def flaky(path, originals):  # first attack "recovers" a value; later ones are real
        calls.append(path)
        res = real(path, originals)
        if len(calls) == 1:
            res[0].passed, res[0].evidence = False, ["1 original value(s) recoverable from text"]
        return res

    monkeypatch.setattr(verify, "attack", flaky)
    res = asyncio.run(run_agentic(sid, GROQ, transport=scripted(lambda fs: [{"finding_id": f["id"], "action": "remove", "reason": "x"} for f in fs])))
    assert res["cycles"] == 2 and res["clean"], res
    assert "FormatGuard" not in {a["name"] for a in res["agents"]}, "PDFs always fill boxes: nothing for it to choose"
    assert res["decisions"] and all(d["action"] == "remove" for d in res["decisions"])
    assert any(a["name"] == "MetaPlanner" and "execute_destructive_redaction" in a["tools"] for a in res["agents"][1:])


def test_image_session_spawns_visual_specialist(fx, tmp_path, monkeypatch):
    monkeypatch.setenv("SHIELD_MCP_ROOT", str(fx["png"].parent))
    sid = open_document(str(fx["png"]))["session_id"]
    res = asyncio.run(run_agentic(sid, GROQ, transport=scripted(lambda fs: [{"finding_id": f["id"], "action": "remove", "reason": "x"} for f in fs])))
    child = next(c for c in res["agents"][0]["children"] if c["name"] == "VisualLikenessAgent")
    assert child["tools"] == ["compute_perceptual_pdq"] and child["steps"] > 0
    assert res["clean"]


def test_refactorbot_workspace_fix_is_atomic_and_parses(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "app").mkdir(parents=True)
    key = "sk-proj-" + "".join(secrets.choice(string.ascii_letters + string.digits) for _ in range(48))
    (repo / "app/config.py").write_text(f'import json\nOPENAI_API_KEY = "{key}"\nprint(json.dumps({{}}))\n')
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    monkeypatch.setenv("SHIELD_MCP_ROOT", str(tmp_path))
    sid = open_workspace(str(repo))["session_id"]
    assert SESSIONS[sid].kind == "workspace"

    def decide(findings):
        return [{"finding_id": f["id"], "action": "move_to_env" if "move_to_env" in (f.get("fixes") or []) else "keep", "reason": "x"} for f in findings]

    res = asyncio.run(run_agentic(sid, GROQ, transport=scripted(decide)))
    src = (repo / "app/config.py").read_text()
    ast.parse(src)
    assert key not in src and 'os.environ["OPENAI_API_KEY"]' in src
    assert key in (repo / ".env").read_text() and list((repo / ".shield/backup").iterdir())
    assert res["clean"] and any(a["name"] == "RefactorBot" for a in res["agents"])


def test_llm_errors_fail_safe_never_skip_verification(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIELD_MCP_ROOT", str(tmp_path))
    doc = tmp_path / "a.txt"
    doc.write_text("mail ravi.kumar@acme.in\n")
    sid = open_document(str(doc))["session_id"]
    down = httpx.MockTransport(lambda r: httpx.Response(503))
    res = asyncio.run(run_agentic(sid, GROQ, transport=down))
    # no agent answered, so the plan is a fallback and must say so: nobody contested is not the same as nobody replied
    assert res["fail_safe_defaults"] == 1 and res["clean"] and res["attacks"]  # every finding redacted, attack still ran
    assert all(d["action"] != "keep" for d in res["decisions"])
    assert "ravi.kumar@acme.in" not in open(res["output"]).read()


def test_adjudicator_failure_redacts_all_even_what_was_challenged(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIELD_MCP_ROOT", str(tmp_path))
    doc = tmp_path / "a.txt"
    doc.write_text("mail ravi.kumar@acme.in\n")
    sid = open_document(str(doc))["session_id"]
    real = scripted(lambda fs: [{"finding_id": f["id"], "action": "keep", "reason": "x"} for f in fs])

    def handler(req):
        if "Synthesize the debate" in json.loads(req.content)["messages"][0]["content"]:
            return httpx.Response(503)
        return real.handle_request(req)

    res = asyncio.run(run_agentic(sid, GROQ, transport=httpx.MockTransport(handler)))
    assert res["fail_safe_defaults"] == 1 and res["clean"] and res["attacks"]
    assert "ravi.kumar@acme.in" not in open(res["output"]).read()


def test_malformed_opinion_still_redacts(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIELD_MCP_ROOT", str(tmp_path))
    doc = tmp_path / "a.txt"
    doc.write_text("mail ravi.kumar@acme.in\n")
    sid = open_document(str(doc))["session_id"]
    real = scripted()

    def handler(req):
        body = json.loads(req.content)
        if "You are PragmaticDev" in body["messages"][0]["content"] and body["messages"][-1]["role"] == "tool":
            return _reply(json.dumps({"opinions": [{"stance": "keep"}]}))
        return real.handle_request(req)

    res = asyncio.run(run_agentic(sid, GROQ, transport=httpx.MockTransport(handler)))
    assert res["clean"] and "ravi.kumar@acme.in" not in open(res["output"]).read()


def test_mcp_resources_and_tool_list(tmp_path, monkeypatch):
    monkeypatch.setenv("SHIELD_MCP_ROOT", str(tmp_path))
    doc = tmp_path / "a.txt"
    doc.write_text("Aadhaar 2345 6789 0124\n")
    sid = open_document(str(doc))["session_id"]

    async def go():
        names = {t.name for t in await server.list_tools()}
        layers = await server.read_resource(f"shield://doc/{sid}/layers")
        return names, layers

    names, layers = asyncio.run(go())
    assert {"validate_checksum", "extract_ocr_region", "compute_perceptual_pdq", "compute_pca_projection", "execute_destructive_redaction",
            "refactor_code_secret", "adversarial_penetration_test", "run_agent_swarm"} <= names
    assert "2345 6789" not in str(layers)
    with pytest.raises(Exception):
        open_document("/etc/hosts")  # outside SHIELD_MCP_ROOT


def test_mcp_stdio_transport_end_to_end(tmp_path):
    """Real JSON-RPC over stdio: spawn `python -m shield.mcp`, list tools, open a doc, validate a checksum."""
    import os
    import sys

    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    doc = tmp_path / "a.txt"
    doc.write_text("Aadhaar 2345 6789 0124\n")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    params = StdioServerParameters(command=sys.executable, args=["-m", "shield.mcp"],
                                   env={**os.environ, "PYTHONPATH": root, "SHIELD_MCP_ROOT": str(tmp_path)})

    async def go():
        async with stdio_client(params) as (r, w):
            async with ClientSession(r, w) as s:
                await s.initialize()
                names = {t.name for t in (await s.list_tools()).tools}
                opened = json.loads((await s.call_tool("open_document", {"path": str(doc)})).content[0].text)
                fid = opened["findings"][0]["id"]
                check = json.loads((await s.call_tool("validate_checksum", {"session_id": opened["session_id"], "finding_id": fid})).content[0].text)
                return names, opened, check

    names, opened, check = asyncio.run(go())
    assert "run_agent_swarm" in names and check["valid"] is True
    assert "2345 6789 0124" not in json.dumps(opened)


def test_keep_without_pragmatic_challenge_is_rejected(tmp_path, monkeypatch):
    """PragmaticDev challenges the e-mail, the adjudicator keeps the mobile instead: that keep is not grounded, so it is redacted."""
    monkeypatch.setenv("SHIELD_MCP_ROOT", str(tmp_path))
    doc = tmp_path / "m.txt"
    doc.write_text("Mobile +91 98765 43210\nmail ravi.kumar@acme.in\n")
    sid = open_document(str(doc))["session_id"]
    ids = {f["type"]: f["id"] for f in list_findings(sid)["findings"]}
    mobile, email = ids["MOBILE_IN"], ids["EMAIL_ADDRESS"]
    real = scripted(lambda fs: [{"finding_id": f["id"], "action": "keep" if f["id"] == email else "token", "reason": "x"} for f in fs])

    def handler(req):
        if "Synthesize the debate" in json.loads(req.content)["messages"][0]["content"]:
            return _reply(json.dumps({"keep": [{"finding_id": mobile, "reason": "trust me"}], "actions": []}))
        return real.handle_request(req)

    events = []
    res = asyncio.run(run_agentic(sid, GROQ, events.append, httpx.MockTransport(handler)))
    assert "SwarmAdjudicator" in {a["name"] for a in res["agents"]}, "the adjudicator must have run for this to prove anything"
    decision = {d["finding_id"]: d for d in res["decisions"]}
    assert decision[mobile]["action"] != "keep" and "keep rejected" in decision[mobile]["reason"]
    assert next(e for e in events if e.get("event") == "binding_plan")["rejected_keeps"] == 1
    out = open(res["output"]).read()
    assert "98765 43210" not in out and "ravi.kumar@acme.in" not in out


@pytest.mark.parametrize("when", ["mid_run", "after_answer"])
def test_a_note_from_one_agent_changes_another_agents_answer(tmp_path, monkeypatch, when):
    """StatutoryAuditor proves the Aadhaar and tells PragmaticDev, who wanted to keep it. Dev reads the note
    during its run, or in a rebuttal turn when it had already answered, and drops the keep."""
    import threading

    monkeypatch.setenv("SHIELD_MCP_ROOT", str(tmp_path))
    doc = tmp_path / "hr.txt"
    doc.write_text("Aadhaar 2345 6789 0124\nmail ravi.kumar@acme.in\n")
    sid = open_document(str(doc))["session_id"]
    ids = {f["type"]: f["id"] for f in list_findings(sid)["findings"]}
    aadhaar = ids["IN_AADHAAR"]
    real = scripted()
    note_sent, dev_answered, dev_requests = threading.Event(), threading.Event(), []

    def opinions(keep=()):
        return _reply(json.dumps({"opinions": [{"id": i, "stance": "keep" if i in keep else "redact", "reason": "x"} for i in ids.values()]}))

    def handler(req):
        body = json.loads(req.content)
        system, msgs = body["messages"][0]["content"], body["messages"][1:]
        text = json.dumps(msgs)
        if "You are StatutoryAuditor" in system:
            if len(msgs) == 1:
                return _reply(calls=[("validate_checksum", {"finding_id": aadhaar})])
            if "message_agent" not in text:
                if when == "after_answer":
                    assert dev_answered.wait(10)
                return _reply(calls=[("message_agent", {"to": "PragmaticDev", "text": f"{aadhaar} passes Verhoeff: 2345 6789 0124 is a real number"})])
            note_sent.set()
            return opinions()
        if "You are PragmaticDev" in system:
            dev_requests.append(text)
            if "passes Verhoeff" in text:
                return opinions()
            if len(msgs) == 1:
                return _reply(calls=[("list_findings", {})])
            if when == "mid_run":
                assert note_sent.wait(10)  # hold this turn until the note is on the board; the next step must deliver it
                return _reply(calls=[("list_findings", {})])
            dev_answered.set()
            return opinions(keep=[aadhaar])
        return real.handle_request(req)

    events = []
    res = asyncio.run(run_agentic(sid, GROQ, events.append, httpx.MockTransport(handler)))

    assert [(m["from"], m["to"]) for m in res["messages"]] == [("StatutoryAuditor", "PragmaticDev")]
    assert "2345 6789 0124" not in json.dumps(res["messages"]) and "2345 6789 0124" not in json.dumps(events), "a note is masked like every other agent text"
    assert any(e.get("event") == "message" and e["agent"] == "StatutoryAuditor" and e["to"] == "PragmaticDev" for e in events)
    told = [r for r in dev_requests if "passes Verhoeff" in r]
    assert told and ("PEER NOTES" in told[0]) == (when == "mid_run")
    devs = [a for a in res["agents"] if a["name"] == "PragmaticDev"]
    rebuttals = [e for e in events if e.get("event") == "round" and e.get("round") == "rebuttal"]
    if when == "after_answer":
        assert len(devs) == 2 and rebuttals[0]["agents"] == ["PragmaticDev"] and "your_opinions" in told[0] and "peer_notes" in told[0]
    else:
        assert len(devs) == 1 and not rebuttals, "the note was read in time: no second turn is spent"
    assert all(d["action"] != "keep" for d in res["decisions"]) and "SwarmAdjudicator" not in {a["name"] for a in res["agents"]}
    assert "2345 6789 0124" not in open(res["output"]).read() and res["clean"]


def test_notes_are_addressed_masked_bounded_and_read_once(tmp_path):
    from shield.agent_runtime import MAX_MESSAGES, MESSAGE_CHARS, AutonomousAgent

    s = register_session(tmp_path / "x.txt", {"text": ""}, {}, tmp_path)
    a, b = (AutonomousAgent(n, "", ["message_agent"], GROQ, s.id) for n in "AB")
    wrong = a._send({"to": "Nobody", "text": "hello"})
    assert "unknown agent" in wrong["error"] and "'B'" in wrong["error"]
    assert "error" in a._send({"to": "B", "text": "  "})
    assert a._send({"to": "B", "text": "card 4111 1111 1111 1111 seen " + "x" * 400}) == {"sent": True, "to": "B"}
    note = s.board[0]["text"]
    assert "4111 1111 1111 1111" not in note and len(note) <= MESSAGE_CHARS
    assert a._send({"to": "all", "text": "layer two unreadable"})["sent"]
    assert [n["from"] for n in b._inbox()] == ["A", "A"] and b._inbox() == [] and a._inbox() == [], "read once, and never your own"
    for _ in range(MAX_MESSAGES - 2):
        a._send({"to": "B", "text": "more"})
    assert "limit" in a._send({"to": "B", "text": "one too many"})["error"]
    mute = AutonomousAgent("C", "", [], GROQ, s.id)
    assert "not permitted" in asyncio.run(mute._call("message_agent", {"to": "B", "text": "x"}))["error"]


def test_claude_gets_a_peer_note_inside_the_tool_result_turn():
    """Anthropic wants user and assistant turns to alternate, so a note after tool results joins that user turn."""
    seen = []

    def handler(req):
        seen.append(json.loads(req.content))
        return httpx.Response(200, json={"content": [{"type": "text", "text": "{}"}], "usage": {}})

    cfg = llm.configure(llm.LLMConfig(retries=0), provider="anthropic", api_key="test-key")
    history = [{"role": "user", "content": "{}"},
               {"role": "assistant", "text": "", "tool_calls": [{"id": "t1", "name": "list_findings", "args": {}}]},
               {"role": "tool", "id": "t1", "name": "list_findings", "content": "{}"},
               {"role": "user", "content": "PEER NOTES: x"}]
    llm.chat_tools(cfg, "s", history, [], httpx.MockTransport(handler))
    msgs = seen[0]["messages"]
    assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
    assert [b["type"] for b in msgs[-1]["content"]] == ["tool_result", "text"]
