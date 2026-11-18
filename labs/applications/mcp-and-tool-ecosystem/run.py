"""Lab: MCP tools: inspect discovery, calls and schema size.

Wiki pages
  Chinese: https://dmxsf.com/wiki/applications/mcp-and-tool-ecosystem/
  English: https://dmxsf.com/en/wiki/applications/mcp-and-tool-ecosystem/
"""

from dmxsf import lab
from dmxsf.applications.mcp_messages import catalog_tools, handle, wire_size


def main() -> None:
    _args, report = lab.start(
        "MCP tools: inspect discovery, calls and schema size",
        wiki=["applications/mcp-and-tool-ecosystem"],
    )
    report.section("1. Match a response to the request and validate the arguments")
    request = {
        "jsonrpc": "2.0",
        "id": 7,
        "method": "tools/call",
        "params": {"name": "catalog_00_lookup", "arguments": {"title": "atlas"}},
    }
    tools = catalog_tools(8)
    good = handle(request, tools, {"atlas": "shelf A"})
    report.check(
        "tools/call preserves the request id and returns read-only content",
        good["id"] == 7 and good["result"]["content"] == [{"type": "text", "text": "shelf A"}],
    )
    request["params"]["arguments"]["title"] = 42
    report.check(
        "the server rejects a number where the schema asks for a string",
        handle(request, tools, {})["error"]["code"] == -32602,
    )
    report.section("2. Compare unframed bytes across catalog sizes")
    list_request = {"jsonrpc": "2.0", "id": 8, "method": "tools/list"}
    sizes = [wire_size(handle(list_request, catalog_tools(n), {})) for n in (1, 8, 24, 96)]
    report.table(["tools", "JSON bytes"], list(zip([1, 8, 24, 96], sizes, strict=True)))
    report.check(
        "the fixed envelope is amortized and the per-tool increment stays constant",
        sizes[0] < sizes[1] < sizes[2] < sizes[3]
        and (sizes[3] - sizes[2]) * 16 == (sizes[2] - sizes[1]) * 72,
    )
    report.say(
        "Bytes exclude transport framing and are not a token count. Version negotiation and SDK behavior are covered in the Wiki."
    )
    report.finish()


if __name__ == "__main__":
    main()
