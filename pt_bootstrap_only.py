$bo = @'
import os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

msg = f"[BOOT_ONLY] PID={os.getpid()} PPID={os.getppid()} ARGV={sys.argv} CWD={os.getcwd()} EXE={sys.executable}"
print(msg, flush=True)
(ROOT / "run" / "boot_only.txt").open("a", encoding="utf-8").write(msg + "\n")

time.sleep(20)
'@

Set-Content -Path .\run\pt_bootstrap_only.py -Value $bo -Encoding UTF8
