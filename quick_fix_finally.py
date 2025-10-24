import re
import sys
from pathlib import Path

p = Path("paper_trader.py")
src = p.read_text(encoding="utf-8")

# Replace the entire finally: tail with a clean, valid version
pattern = r"""
(\s*except\s+KeyboardInterrupt:\s*\n\s*log\("Interrupted; shutting down."\)\s*\n\s*finally:\s*)
.*\Z
"""
replacement = r"""\1        # Daily summary on exit
        if args.daily_summary and 'daily_trade_log' in locals() and daily_trade_log:
            try:
                write_daily_summary(current_day.strftime("%Y%m%d"), daily_trade_log, pathlib.Path("./logs"), chooser)
            except Exception as e:
                warn(f"[SUMMARY] final write failed: {e}")

        # Final save of chooser state
        if args.persist_chooser and args.chooser_model_path:
            try:
                os.makedirs(os.path.dirname(args.chooser_model_path), exist_ok=True)
                with open(args.chooser_model_path, "w", encoding="utf-8") as f:
                    json.dump(chooser.dump_state(), f)
                log(f"Saved chooser state to {args.chooser_model_path}")
            except Exception as e:
                warn(f"Could not save chooser state on exit: {e}")

        # Snapshot chooser on exit (for charts)
        try:
            if args.persist_chooser and args.chooser_model_path:
                snapshot_chooser(args.chooser_model_path)
        except Exception as e:
            warn(f"[SNAPSHOT] exit snapshot failed: {e}")

        # Cleanup connections/streams
        try:
            if 'rtb' in locals() and rtb:
                ib.cancelRealTimeBars(rtb)
        except Exception:
            pass
        try:
            ib.disconnect()
        except Exception:
            pass
        try:
            if tee:
                tee.close()
        except Exception:
            pass
"""

new = re.sub(pattern, replacement, src, flags=re.S | re.X)
if new == src:
    print("Pattern not found or already fixed; no changes written.")
    sys.exit(0)

bak = Path("paper_trader.py.autofix.bak")
bak.write_text(src, encoding="utf-8")
p.write_text(new, encoding="utf-8")
print("Fixed. Backup:", bak)
