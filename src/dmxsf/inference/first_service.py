"""A loopback-only toy inference endpoint for /wiki/inference/first-inference-service/.

Wiki pages
  Chinese: https://dmxsf.com/wiki/inference/first-inference-service/
  English: https://dmxsf.com/en/wiki/inference/first-inference-service/
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

TRANSITIONS = {"one": "two", "two": "three", "three": None}


def generate(prompt: str, limit: int) -> list[str]:
    """Decode a toy transition table until stop or the output budget."""
    tokens = prompt.split()
    if not tokens or type(limit) is not int or not 1 <= limit <= 8:
        raise ValueError("need text and an integer limit from 1 to 8")
    current = tokens[-1]
    output = []
    for _ in range(limit):
        following = TRANSITIONS.get(current)
        if following is None:
            break
        output.append(following)
        current = following
    return output


class GenerateHandler(BaseHTTPRequestHandler):
    def do_POST(self) -> None:
        if self.path != "/generate":
            self.respond(404, {"error": "unknown route"})
            return
        try:
            size = int(self.headers.get("Content-Length", "-1"))
            if not 1 <= size <= 4096:
                raise ValueError("invalid body length")
            obj = json.loads(self.rfile.read(size))
            prompt = obj.get("prompt") if isinstance(obj, dict) else None
            if not isinstance(prompt, str):
                raise ValueError("prompt must be a string")
            output = generate(prompt, obj.get("max_tokens", 2))
        except (ValueError, UnicodeError):
            self.respond(400, {"error": "invalid request"})
            return
        self.respond(200, {"tokens": output})

    def respond(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args: object) -> None:
        pass


def make_server(port: int = 0) -> ThreadingHTTPServer:
    """Bind to loopback only; port zero asks the OS to choose a free port."""
    return ThreadingHTTPServer(("127.0.0.1", port), GenerateHandler)
