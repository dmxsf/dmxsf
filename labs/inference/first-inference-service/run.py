"""Send real HTTP requests to a toy decoder on an ephemeral loopback port.

Wiki pages
  Chinese: https://dmxsf.com/wiki/inference/first-inference-service/
  English: https://dmxsf.com/en/wiki/inference/first-inference-service/
"""

import json
from http.client import HTTPConnection
from threading import Thread

from dmxsf import lab
from dmxsf.inference.first_service import make_server


def request(port: int, path: str, payload: dict) -> tuple[int, dict]:
    client = HTTPConnection("127.0.0.1", port, timeout=3)
    try:
        client.request("POST", path, json.dumps(payload), {"Content-Type": "application/json"})
        response = client.getresponse()
        return response.status, json.loads(response.read())
    finally:
        client.close()


def main() -> None:
    _, report = lab.start(
        "First inference service: a local HTTP endpoint", wiki=["inference/first-inference-service"]
    )
    report.section("Loopback HTTP")
    with make_server() as server:
        thread = Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        thread.start()
        try:
            port = server.server_port
            ok = request(port, "/generate", {"prompt": "one", "max_tokens": 2})
            invalid = request(port, "/generate", {"prompt": "one", "max_tokens": True})
            absent = request(port, "/other", {"prompt": "one"})
        finally:
            server.shutdown()
            thread.join()
    report.check("decode returns two tokens", ok == (200, {"tokens": ["two", "three"]}), str(ok))
    report.check("invalid limit returns 400", invalid[0] == 400)
    report.check("unknown route returns 404", absent[0] == 404)
    report.finish()


if __name__ == "__main__":
    main()
