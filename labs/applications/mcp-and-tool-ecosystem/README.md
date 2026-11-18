# MCP 工具：检查发现、调用与模式大小

[English](README.en.md)

[MCP 与工具生态：协议怎么工作，工具多了怎么办](https://dmxsf.com/wiki/applications/mcp-and-tool-ecosystem/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/applications/mcp_messages.py)

离线检查 JSON-RPC 的 tools/list 与 tools/call 报文，测量缩小候选工具目录后返回给客户端的数据。

实测的是紧凑 JSON 的 UTF-8 字节。Wiki 的官方 SDK 报文、消息头和模型选工具结果使用其正文实验。

## 运行命令

```bash
uv run dmxsf run applications/mcp-and-tool-ecosystem
```

## 练习

把工具从 8 个增加到 96 个，比较完整目录和一个候选工具的编码大小。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

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
