from __future__ import annotations

# pt/logging.py
#
# Minimal structured logger used across the project.

import json
from pt.utils_time import utc_now_str


def log(evt: str, **fields):
    payload = {"ts": utc_now_str(), "evt": evt}
    payload.update(fields)
    print(json.dumps(payload, ensure_ascii=False), flush=True)
