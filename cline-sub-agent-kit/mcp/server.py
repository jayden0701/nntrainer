#!/usr/bin/env python3
"""
Minimal stdio MCP server exposing a `cline_sub_agent` tool.

The tool hands ANY task to Cline (running in the user's VS Code window) via the
local Claude-Cline Bridge HTTP endpoint, lets Cline run it to completion
unattended (tool approvals auto-handled by the bridge), and returns Cline's final
message plus the list of changed files.

Intended use: let Claude Code use Cline as a general-purpose sub-agent —
exploration, implementation, refactoring, bug-fixing, review, running build/test,
etc. — instead of doing the work inline. For read-only use (explore/review),
the caller says so in the task and changedFiles comes back empty.

Pure stdlib — no external deps. MCP stdio transport = newline-delimited
JSON-RPC 2.0.
"""

import json
import os
import sys
import urllib.request
import urllib.error

TOKEN_FILE = os.path.expanduser("~/.claude-cline-bridge-token")
PORT_RANGE = range(39111, 39131)
HOST = "127.0.0.1"
# Must exceed the bridge's own per-run cap (20 min) so we receive its response.
RUN_TIMEOUT = 25 * 60

PROTOCOL_VERSION = "2025-06-18"

TOOLS = [
    {
        "name": "cline_sub_agent",
        "description": (
            "Use Cline (the agent running in the user's open VS Code workspace) "
            "as a general-purpose SUB-AGENT: hand it any task and get back its "
            "final message plus the list of files it changed. Not just for writing "
            "code — use it for exploration / explaining a codebase, implementation, "
            "refactoring, bug-fixing, reviewing a change, running builds or tests, "
            "etc. Delegate work here instead of doing it inline so you don't burn "
            "your own context. Cline runs the task end-to-end with tool approvals "
            "handled automatically and returns when done. For read-only work "
            "(exploration/review), say so explicitly in the task (e.g. 'only read "
            "and report — do not modify any files'); changedFiles should then come "
            "back empty. The task runs in whatever folder VS Code currently has "
            "open (you don't choose the directory). Cline must be in Act mode. "
            "One task per call (calls are serialized — Cline runs one at a time)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": (
                        "The task for Cline in natural language. Be explicit about "
                        "the goal and any constraints (which files to touch or "
                        "avoid, whether it's read-only, acceptance criteria). One "
                        "self-contained task per call."
                    ),
                }
            },
            "required": ["task"],
        },
    }
]


def read_token():
    try:
        return open(TOKEN_FILE).read().strip()
    except Exception:
        return None


def find_bridge_port():
    for port in PORT_RANGE:
        try:
            with urllib.request.urlopen(
                f"http://{HOST}:{port}/health", timeout=2
            ) as r:
                data = json.loads(r.read().decode())
                if data.get("ok") and data.get("clineId"):
                    return port, data
        except Exception:
            continue
    return None, None


def run_cline(task: str) -> str:
    token = read_token()
    if not token:
        return (
            "ERROR: bridge token not found at ~/.claude-cline-bridge-token. "
            "Is the Claude-Cline Bridge extension installed and has VS Code been "
            "reloaded at least once?"
        )
    port, health = find_bridge_port()
    if not port:
        return (
            "ERROR: Claude-Cline Bridge not reachable on 127.0.0.1:39111-39130. "
            "Open the VS Code window that has the bridge + Cline SR installed "
            "(and run 'Developer: Reload Window' if you just installed it)."
        )
    if not health.get("clineActivated"):
        return "ERROR: Cline SR is not activated in the bridge's VS Code host."

    body = json.dumps({"task": task}).encode()
    req = urllib.request.Request(
        f"http://{HOST}:{port}/run",
        data=body,
        headers={"Content-Type": "application/json", "X-Bridge-Token": token},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=RUN_TIMEOUT) as r:
            res = json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            res = json.loads(e.read().decode())
        except Exception:
            return f"ERROR: bridge HTTP {e.code}"
    except Exception as e:
        return f"ERROR: request to bridge failed: {e!r}"

    return format_result(res)


def format_result(res: dict) -> str:
    files = res.get("changedFiles") or []
    files_block = (
        "\n".join("  - " + f for f in files) if files else "  (none reported)"
    )
    final = (res.get("finalMessage") or "").strip()

    if res.get("ok") and res.get("completed"):
        return (
            "Cline completed the task.\n\n"
            f"Final message:\n{final or '(empty)'}\n\n"
            f"Changed files ({len(files)}):\n{files_block}\n\n"
            f"taskId: {res.get('taskId')}"
        )

    # not completed
    lines = ["Cline did NOT complete the task.",
             f"reason: {res.get('error', 'unknown')}"]
    pa = res.get("pendingAsk")
    if pa:
        lines.append(f"pending question ({pa.get('ask')}): {pa.get('text', '')}")
    if final:
        lines.append(f"\nLast message:\n{final}")
    if files:
        lines.append(f"\nFiles changed so far ({len(files)}):\n{files_block}")
    if res.get("taskId"):
        lines.append(f"\ntaskId: {res.get('taskId')}")
    return "\n".join(lines)


def make_response(req_id, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": req_id}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    return msg


def handle(req):
    method = req.get("method")
    req_id = req.get("id")
    if req_id is None:  # notification
        return None

    if method == "initialize":
        client_pv = (req.get("params") or {}).get("protocolVersion")
        return make_response(req_id, {
            "protocolVersion": client_pv or PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "cline-sub-agent", "version": "1.0.0"},
        })

    if method == "ping":
        return make_response(req_id, {})

    if method == "tools/list":
        return make_response(req_id, {"tools": TOOLS})

    if method == "tools/call":
        params = req.get("params") or {}
        name = params.get("name")
        args = params.get("arguments") or {}
        if name != "cline_sub_agent":
            return make_response(req_id, error={"code": -32602, "message": f"unknown tool: {name}"})
        task = args.get("task")
        if not task or not isinstance(task, str):
            return make_response(req_id, error={"code": -32602, "message": "missing required 'task' string"})
        text = run_cline(task)
        return make_response(req_id, {"content": [{"type": "text", "text": text}]})

    return make_response(req_id, error={"code": -32601, "message": f"method not found: {method}"})


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            continue
        try:
            resp = handle(req)
        except Exception as e:  # noqa: BLE001
            resp = make_response(req.get("id"), error={"code": -32603, "message": f"internal error: {e!r}"})
        if resp is not None:
            sys.stdout.write(json.dumps(resp) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
