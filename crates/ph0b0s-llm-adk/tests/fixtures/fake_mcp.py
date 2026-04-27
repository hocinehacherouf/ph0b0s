#!/usr/bin/env python3
"""Tiny stdio MCP server for hermetic ph0b0s-llm-adk tests.

Implements just enough of MCP 2024-11-05 to let McpToolset:
  1. initialize
  2. tools/list  -> returns one tool named `ping`
  3. tools/call  -> returns `{"pong": true}` for any args

No deps. Run: `python3 fake_mcp.py`. Communicates via JSON-RPC 2.0 over stdio.
"""
import json
import sys


def respond(id_, result=None, error=None):
    msg = {"jsonrpc": "2.0", "id": id_}
    if error is not None:
        msg["error"] = error
    else:
        msg["result"] = result
    sys.stdout.write(json.dumps(msg) + "\n")
    sys.stdout.flush()


def handle(req):
    method = req.get("method")
    rid = req.get("id")
    if method == "initialize":
        respond(rid, {
            "protocolVersion": "2024-11-05",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "0.1"},
        })
    elif method == "notifications/initialized":
        # No response for notifications.
        pass
    elif method == "tools/list":
        respond(rid, {
            "tools": [{
                "name": "ping",
                "description": "responds with pong",
                "inputSchema": {"type": "object", "properties": {}},
            }]
        })
    elif method == "tools/call":
        respond(rid, {
            "content": [{"type": "text", "text": json.dumps({"pong": True})}]
        })
    elif rid is not None:
        respond(rid, error={"code": -32601, "message": f"unknown method: {method}"})


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except Exception:
            continue
        handle(req)


if __name__ == "__main__":
    main()
