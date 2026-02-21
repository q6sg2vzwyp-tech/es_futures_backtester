$so = @'
import os, sys, time, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

def log(where):
    msg = f"[SITE_ONLY:{where}] PID={os.getpid()} PPID={os.getppid()} ARGV={sys.argv}"
    print(msg, flush=True)
    (ROOT / "run" / "site_only.txt").open("a", encoding="utf-8").write(msg + "\n")

log("pre_import")
try:
    import sitecustomize
    log("post_import")
except Exception:
    err = traceback.format_exc()
    print(err, flush=True)
    (ROOT / "run" / "site_only.txt").open("a", encoding="utf-8").write(err + "\n")
    raise

time.sleep(20)
'@

Set-Content -Path .\run\pt_site_only.py -Value $so -Encoding UTF8
