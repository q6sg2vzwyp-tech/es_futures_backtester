# worker.py
# Runs the trading kernel in a process whose lifecycle is controlled by runner.py.

from __future__ import annotations
import os
import sys

# Disable in-file guards (runner owns this now)
os.environ["PT_FORCE_MUTEX_SINGLETON"] = "0"
os.environ["PT_FORCE_SINGLETON"] = "0"

# Mark that we're the supervised worker (optional, but useful if paper_trader checks it)
os.environ["PT_ROLE"] = "worker"

# ---- CRITICAL: strip runner-only flags so paper_trader doesn't self-spawn ----
RUNNER_ONLY_FLAGS = {
    "--inproc",          # this is the smoking gun in your process list
    "--runner",          # if you ever add one
    "--supervisor",      # if you ever add one
}

def _sanitize_argv(argv: list[str]) -> list[str]:
    out = [argv[0]]
    skip_next = False
    for a in argv[1:]:
        if skip_next:
            skip_next = False
            continue
        # handle flags that might be like: --flag=value
        base = a.split("=", 1)[0]
        if base in RUNNER_ONLY_FLAGS:
            continue
        out.append(a)
    return out

sys.argv = _sanitize_argv(sys.argv)

def main():
    import paper_trader  # import after argv/env sanitization
    paper_trader.main()

if __name__ == "__main__":
    main()
