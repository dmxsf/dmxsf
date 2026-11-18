"""MCP tools: inspect discovery, calls and schema size.

Wiki pages
  Chinese: https://dmxsf.com/wiki/applications/mcp-and-tool-ecosystem/
  English: https://dmxsf.com/en/wiki/applications/mcp-and-tool-ecosystem/
"""

import json
from collections.abc import Mapping


def wire_size(message: Mapping) -> int:
    """UTF-8 byte count of one compact JSON object, excluding transport framing."""
    return len(json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))


def catalog_tools(size: int) -> list[dict]:
    """Synthetic read-only tools for the offline catalog measurement."""
    return [
        {
            "name": f"catalog_{i:02d}_lookup",
            "description": f"Look up an item in collection {i:02d}.",
            "inputSchema": {
                "type": "object",
                "properties": {"title": {"type": "string"}},
                "required": ["title"],
                "additionalProperties": False,
            },
        }
        for i in range(size)
    ]


def handle(request: Mapping, tools: list[dict], shelves: Mapping[str, str]) -> dict:
    """Handle a teaching subset of JSON-RPC messages without an SDK or transport."""
    response = {"jsonrpc": "2.0", "id": request["id"]}
    method = request.get("method")
    if method == "tools/list":
        return {**response, "result": {"tools": tools}}
    if method != "tools/call":
        return {**response, "error": {"code": -32601, "message": "method not found"}}
    params = request.get("params", {})
    invalid = {**response, "error": {"code": -32602, "message": "invalid tool arguments"}}
    if not isinstance(params, Mapping):
        return invalid
    arguments = params.get("arguments")
    known = any(tool["name"] == params.get("name") for tool in tools)
    if (
        not known
        or not isinstance(arguments, dict)
        or set(arguments) != {"title"}
        or not isinstance(arguments["title"], str)
    ):
        return invalid
    return {
        **response,
        "result": {
            "content": [{"type": "text", "text": shelves.get(arguments["title"], "not found")}]
        },
    }
