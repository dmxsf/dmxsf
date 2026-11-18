# Tool authorization: separate data, proposals and permissions

[中文](README.md)

[Prompt Injection and Jailbreaks: The Security Boundaries of LLM Applications](https://dmxsf.com/en/wiki/safety/prompt-injection-and-jailbreaks/) · [Lab script](run.py) · [Library code](../../../src/dmxsf/applications/tool_authorization.py)

Replay a benign read and policy-boundary cases in a mocked runtime, verifying that content cannot grant new privileges.

The exercise covers application-side authorization with an in-memory store and a read-only grant.

## Run the lab

```bash
uv run dmxsf run safety/prompt-injection-and-jailbreaks
```

## Exercise

Change the permitted scope from one workspace to another. Confirm that the first workspace's record remains inaccessible.

## Full run (Apple M2 Max CPU, Python 3.12.14, seed 0)

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
