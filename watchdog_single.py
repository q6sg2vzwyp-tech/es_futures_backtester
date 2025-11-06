#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
watchdog_single.py — simple restart watchdog for paper_trader.py

Features
- Pass-through args to the bot (anything after `--` goes straight through)
- Optional --args-file <path> to load one line of flags from a text file
- Optional --nocaps to append "no risk caps" runtime flags (no code edits)
- Optional --patch-weekly-cap to safely neutralize weekly cap logic in paper_trader.py
- Exponential backoff restart on non-zero exit
"""

import argparse, os, sys, time, subprocess, shlex, re
from pathlib import Path

# ---------------------------
# Helpers
# ---------------------------

def read_args_file(path: Path) -> list[str]:
    """
    Reads a single line of CLI flags from a text file and splits them shell-style.
    Empty / missing file returns [].
    """
    try:
        txt = path.read_text(encoding="utf-8", errors="ignore")
        # use first non-empty line
        for line in txt.splitlines():
            line = line.strip()
            if line:
                return shlex.split(line)
    except Exception:
        pass
    return []

def make_nocaps_flags() -> list[str]:
    """
    Very large limits that effectively disable runtime caps without any code modifications.
    """
    HUGE = "1000000000"
    YEAR_SEC = str(365 * 24 * 60 * 60)
    return [
        "--day-loss-cap-R", HUGE,
        "--max-trades-per-day", "999999",
        "--max-consec-losses", "999999",
        "--day-guard-pct", "0",
        "--peak-dd-guard-pct", "0",
        "--pos-age-cap-sec", YEAR_SEC,
        "--pos-age-minR", "-" + HUGE,
    ]

def patch_weekly_cap(paper_trader_path: Path) -> bool:
    """
    Idempotent text patch: finds the weekly-cap gating line and neutralizes it.

    Looks for:  if week_R <= weekly_cap_R:
    Rewrites to: if False and (week_R <= weekly_cap_R):
    """
    try:
        src = paper_trader_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[watchdog] WARN: could not read {paper_trader_path}: {e}")
        return False

    # If already patched, do nothing
    if re.search(r"if\s+False\s+and\s*\(\s*week_R\s*<=\s*weekly_cap_R\s*\)\s*:", src):
        print("[watchdog] Weekly cap already neutralized.")
        return True

    # Conservative replace (keep original indentation)
    pattern = r"(^[ \t]*)if\s+week_R\s*<=\s*weekly_cap_R\s*:\s*$"
    repl    = r"\1if False and (week_R <= weekly_cap_R):"
    new_src, n = re.subn(pattern, repl, src, flags=re.MULTILINE)
    if n == 0:
        print("[watchdog] WARN: could not find weekly-cap line to patch; no change made.")
        return False

    try:
        backup = paper_trader_path.with_suffix(paper_trader_path.suffix + ".bak_from_watchdog")
        if not backup.exists():
            backup.write_text(src, encoding="utf-8")
        paper_trader_path.write_text(new_src, encoding="utf-8")
        print("[watchdog] Weekly cap disabled via code patch.")
        return True
    except Exception as e:
        print(f"[watchdog] ERROR: failed to write patched file: {e}")
        return False

def build_bot_cmd(py, bot, user_inline: list[str], user_file: list[str], nocaps: bool) -> list[str]:
    cmd = [str(py), str(bot)]
    if user_file:
        cmd.extend(user_file)
    if user_inline:
        cmd.extend(user_inline)
    if nocaps:
        cmd.extend(make_nocaps_flags())
    return cmd

# ---------------------------
# CLI
# ---------------------------

def parse_cli():
    ap = argparse.ArgumentParser(description="Restart watchdog for paper_trader.py")
    ap.add_argument("--python", default=str(Path(".venv") / "Scripts" / "python.exe"),
                    help="Python executable to use (default: .venv\\Scripts\\python.exe on Windows)")
    ap.add_argument("--bot", default="paper_trader.py",
                    help="Bot script to run (default: paper_trader.py)")
    ap.add_argument("--args-file", default="",
                    help="Optional text file containing a single line of flags for the bot")
    ap.add_argument("--nocaps", action="store_true",
                    help="Append very large limits to effectively disable runtime caps")
    ap.add_argument("--patch-weekly-cap", action="store_true",
                    help="Try to neutralize weekly cap inside paper_trader.py source (idempotent)")
    ap.add_argument("--success-exits", action="store_true",
                    help="Exit watchdog if child exits with code 0")
    ap.add_argument("--min-backoff", type=float, default=2.0, help="Minimum restart backoff seconds")
    ap.add_argument("--max-backoff", type=float, default=60.0, help="Maximum restart backoff seconds")
    ap.add_argument("--", dest="sep", action="store_true", help=argparse.SUPPRESS)  # separator marker
    # Everything after `--` is pass-through; argparse won't parse it. We pull from sys.argv manually.
    known, unknown = ap.parse_known_args()
    # Reconstruct pass-through from the first `--` occurrence
    passthrough = []
    if "--" in sys.argv:
        idx = sys.argv.index("--")
        passthrough = sys.argv[idx+1:]
    else:
        # If user didn't use `--`, treat 'unknown' as pass-through
        passthrough = unknown
    return known, passthrough

# ---------------------------
# Main
# ---------------------------

def main():
    cfg, inline_flags = parse_cli()

    py = Path(cfg.python)
    bot = Path(cfg.bot).resolve()
    if not bot.exists():
        print(f"[watchdog] ERROR: bot not found: {bot}")
        sys.exit(2)

    # Optional args-file
    file_flags = []
    if cfg.args_file:
        file_flags = read_args_file(Path(cfg.args_file))

    # Optional source patch for weekly cap
    if cfg.patch_weekly_cap:
        print("[watchdog] Patching weekly cap…")
        patch_weekly_cap(bot)

    backoff = float(cfg.min_backoff)
    max_backoff = float(cfg.max_backoff)

    while True:
        cmd = build_bot_cmd(py, bot, inline_flags, file_flags, cfg.nocaps)
        print(f"[watchdog] Launching: {cmd}")
        try:
            proc = subprocess.Popen(cmd)
            rc = proc.wait()
        except KeyboardInterrupt:
            print("\n[watchdog] Ctrl-C received. Exiting.")
            break
        except Exception as e:
            print(f"[watchdog] ERROR: failed to launch child: {e}")
            rc = 1

        if rc == 0 and cfg.success_exits:
            print("[watchdog] Child exited 0 (success). Stopping per --success-exits.")
            break

        print(f"[watchdog] Child exited rc={rc}; restarting in {backoff:.1f}s")
        time.sleep(backoff)
        backoff = min(max_backoff, max(backoff * 1.6, cfg.min_backoff))

if __name__ == "__main__":
    main()
