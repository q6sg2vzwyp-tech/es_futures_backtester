"""
patch_paper_trader_for_snapshots.py
Add chooser snapshotting and enhance daily summary to include chooser stats.
Usage:
    python patch_paper_trader_for_snapshots.py path\to\paper_trader.py
This will create a backup 'paper_trader.py.bak' and write the patched file.
"""

import pathlib
import shutil
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python patch_paper_trader_for_snapshots.py path\\to\\paper_trader.py")
        sys.exit(1)
    target = pathlib.Path(sys.argv[1]).resolve()
    txt = target.read_text(encoding="utf-8")
    orig = txt

    # 1) Enhance write_daily_summary signature and body if not present
    if (
        "def write_daily_summary(day_str: str, trades: List[Dict[str, Any]], outdir: pathlib.Path):"
        in txt
    ):
        txt = txt.replace(
            "def write_daily_summary(day_str: str, trades: List[Dict[str, Any]], outdir: pathlib.Path):",
            "def write_daily_summary(day_str: str, trades: List[Dict[str, Any]], outdir: pathlib.Path, chooser: 'LinUCBChooser'=None):",
        )
        txt = txt.replace(
            'summary = {"totals": totals, "per_arm": per_arm}',
            'chooser_stats = {}\n    if chooser is not None:\n        chooser_stats = {\n            "counts": chooser.counts,\n            "reward_sum": chooser.reward_sum,\n            "avgR_by_arm": {a: (chooser.reward_sum.get(a,0.0)/max(1,chooser.counts.get(a,0))) for a in chooser.arms}\n        }\n\n    summary = {"totals": totals, "per_arm": per_arm, "chooser": chooser_stats}',
        )

    # 2) Inject snapshot_chooser() if missing
    if "def snapshot_chooser(" not in txt and "def write_daily_summary" in txt:
        insert_point = txt.find("def write_daily_summary")
        after_point = txt.find("\n# =========================", insert_point)
        if after_point == -1:
            after_point = txt.find("\ndef main()", insert_point)
        snippet = (
            "\n\n# =========================\n"
            "# NEW: Chooser state snapshot (for learning charts)\n"
            "# =========================\n"
            "def snapshot_chooser(model_path: Optional[str]):\n"
            "    if not model_path:\n"
            "        return\n"
            "    try:\n"
            "        src = pathlib.Path(model_path)\n"
            "        if not src.exists():\n"
            "            return\n"
            '        snap_dir = pathlib.Path("./logs/chooser_snapshots")\n'
            "        snap_dir.mkdir(parents=True, exist_ok=True)\n"
            "        dst = snap_dir / f\"linucb_es_{_dt.now().strftime('%Y%m%d')}.json\"\n"
            "        import shutil\n"
            "        shutil.copy(str(src), str(dst))\n"
            '        log(f"[SNAPSHOT] Saved chooser state to {dst}")\n'
            "    except Exception as e:\n"
            '        warn(f"[SNAPSHOT] Failed: {e}")\n'
        )
        txt = txt[:after_point] + snippet + txt[after_point:]

    # 3) Call snapshot_chooser at day rollover after write_daily_summary
    txt = txt.replace(
        'write_daily_summary(current_day.strftime("%Y%m%d"), daily_trade_log, pathlib.Path("./logs"))',
        'write_daily_summary(current_day.strftime("%Y%m%d"), daily_trade_log, pathlib.Path("./logs"), chooser)',
    )
    txt = txt.replace(
        'log("New trading day → resetting daily stats.")',
        'snapshot_chooser(args.chooser_model_path)\n                log("New trading day → resetting daily stats.")',
    )

    # 4) Call snapshot_chooser in finally block after final save (if not already)
    if "snapshot_chooser(args.chooser_model_path)" not in txt.split("finally:", 1)[-1]:
        txt = txt.replace(
            'log(f"Saved chooser state to {args.chooser_model_path}")\n            except Exception as e:',
            'log(f"Saved chooser state to {args.chooser_model_path}")\n            except Exception as e:\n                warn(f"Could not save chooser state on exit: {e}")\n        # Snapshot chooser on exit\n        try:\n            if args.persist_chooser and args.chooser_model_path:\n                snapshot_chooser(args.chooser_model_path)\n        except Exception as e:\n            warn(f"[SNAPSHOT] exit snapshot failed: {e}")\n        try:',
        )

    if txt == orig:
        print("No changes applied (file may already be patched).")
        return

    backup = target.with_suffix(".py.bak")
    shutil.copy(str(target), str(backup))
    target.write_text(txt, encoding="utf-8")
    print(f"Patched: {target}\nBackup:  {backup}")


if __name__ == "__main__":
    main()
