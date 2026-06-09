from __future__ import annotations

import json
import sys
from urllib.request import urlopen


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: _healthcheck.py <base_url>")

    base_url = sys.argv[1].rstrip("/")
    with urlopen(base_url + "/health", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return 0 if payload.get("status") == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
