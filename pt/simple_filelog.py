from __future__ import annotations
import os, time

class DailyFileLog:
    def __init__(self, base_dir: str = r".\logs"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

    def write(self, msg: str):
        try:
            fn = time.strftime("%Y%m%d") + ".log"
            p = os.path.join(self.base_dir, fn)
            with open(p, "a", encoding="utf-8") as f:
                f.write(msg.rstrip() + "\n")
        except Exception:
            pass
