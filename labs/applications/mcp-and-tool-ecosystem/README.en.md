# MCP tools: inspect discovery, calls and schema size

[中文](README.md)

[MCP and the Tool Ecosystem: How It Works and What to Do When Tools Pile Up](https://dmxsf.com/en/wiki/applications/mcp-and-tool-ecosystem/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/applications/mcp_messages.py)

Inspect offline JSON-RPC tools/list and tools/call messages and measure what narrowing the catalog sends to a client.

The lab measures compact JSON bytes. Use the Wiki experiment for the official SDK trace, transport headers and model tool-selection results.

## Run the lab

```bash
uv run dmxsf run applications/mcp-and-tool-ecosystem
```

## Exercise

Increase the number of tools from 8 to 96. Compare the encoded size for the full catalog and a single selected tool.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

```text
# MCP tools: inspect discovery, calls and schema size
  course: https://dmxsf.com/wiki/applications/mcp-and-tool-ecosystem/
          https://dmxsf.com/en/wiki/applications/mcp-and-tool-ecosystem/

## 1. Match a response to the request and validate the arguments
  [ok] tools/call preserves the request id and returns read-only content
  [ok] the server rejects a number where the schema asks for a string

## 2. Compare unframed bytes across catalog sizes
  tools  JSON bytes
  -----  ----------
      1         247
      8        1661
     24        4893
     96       19437
  [ok] the fixed envelope is amortized and the per-tool increment stays constant
Bytes exclude transport framing and are not a token count. Version negotiation and SDK behavior are covered in the Wiki.

All checks passed. (0.0s)
```
