"""Test-only worker: captures reset requests and never imports pyuipc."""

from __future__ import annotations

import json
import os
import sys


def _send(payload: dict[str, object]) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def main() -> int:
    _send({"op": "ready", "worker": "fake-reset-capture"})
    capture = os.environ.get("FAKE_RESET_CAPTURE")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        message = json.loads(line)
        op = message.get("op")
        if op == "reset":
            request = message.get("request", {})
            if capture:
                with open(capture, "w", encoding="utf-8") as handle:
                    json.dump(request, handle)
            _send({"op": "reset_ack", "ack": {"status": "ready"}})
        elif op == "stop":
            _send({"op": "stop_ack"})
            break
        elif op == "health":
            _send({"op": "health_ack", "health": {"alive": True, "rc": None}})
        else:
            _send({"op": "error", "message": f"unknown op {op!r}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
