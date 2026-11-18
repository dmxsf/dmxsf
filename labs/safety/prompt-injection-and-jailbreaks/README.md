# 工具授权：分开数据、动作建议与权限

[English](README.en.md)

[提示注入与越狱：LLM 应用的安全边界](https://dmxsf.com/wiki/safety/prompt-injection-and-jailbreaks/) · [实验脚本](run.py) · [库代码](../../../src/dmxsf/applications/tool_authorization.py)

在模拟运行时回放正常读取与权限边界案例，验证数据内容无法授予新权限。

实验范围是应用执行权限，使用内存中的演示记录和只读授权。

## 运行命令

```bash
uv run dmxsf run safety/prompt-injection-and-jailbreaks
```

## 练习

把授权范围换成另一个工作区，确认原工作区的记录仍不可读。

## 完整运行（Apple M2 Max CPU，Python 3.12.14，种子 0）

```text
# Tool authorization: separate data, proposals and permissions
  course: https://dmxsf.com/wiki/safety/prompt-injection-and-jailbreaks/
          https://dmxsf.com/en/wiki/safety/prompt-injection-and-jailbreaks/

## 1. Confirm the authorized path
  [ok] a permitted read returns exact data without changing caller permissions

## 2. Fail closed across capabilities
  [ok] other scopes and unsupported actions have no observable effect
  [ok] an authorized read of a missing key returns a distinct result

All checks passed. (0.0s)
```
