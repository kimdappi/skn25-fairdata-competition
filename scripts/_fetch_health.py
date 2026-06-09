from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.request import urlopen


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: _fetch_health.py <base_url> <output_path>")

    base_url = sys.argv[1].rstrip("/")
    output_path = Path(sys.argv[2])
    with urlopen(base_url + "/health", timeout=5) as response:
        payload = json.loads(response.read().decode("utf-8"))
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
