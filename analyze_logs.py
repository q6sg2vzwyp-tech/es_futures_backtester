#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse, os, json, math, csv, sys
from pathlib import Path
from datetime import datetime

def parse_line(line):
    # Accept either plain JSON line or prefix like: {"ts": ...}
    line = line.strip()
    if not line:
        return None
    # If the line has a non-JSON prefix (e.g., timestamp + HEARTBEAT ...), skip
    if line[0] not in ('{','['):
        return None
    try:
        obj = json.loads(line)
        return obj
    except Exception:
        return None

def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)

def walk_inputs(input_path):
    p = Path(input_path)
    if p.is_dir():
        for f in sorted(p.rglob("*")):
            if f.suffix.lower() in (".log",".txt",".json",".ndjson") or f.name.lower().startswith("log"):
                yield f
    elif p.is_file():
        yield p

def to_hour(ts_s):
    if not ts_s:
        return None
    try:
        # Try ISO-like "YYYY-mm-dd HH:MM:SS" (bot uses this local)
        if "T" in ts_s and ts_s.endswith("Z"):
            # e.g., 2025-10-21T13:58:24Z
            return int(ts_s[11:13])
        return int(ts_s.split()[1].split(":")[0])
    except Exception:
        return None

def get_session(session_key):
    # Expect "YYYY-mm-dd-S0" or "...-S1". Fallback to "unknown"
    if not session_key or "-S" not in str(session_key):
        return "unknown"
    try:
        idx = str(session_key).split("-S")[-1]
        return "AM" if idx == "0" else "PM"
    except Exception:
        return "unknown"

def safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

def summarize(rows, key_fields):
    # Aggregate by keys: trades, wins, pnl_sum, R_sum, avg_R, win_rate
    out = {}
    for r in rows:
        key = tuple(r.get(k) for k in key_fields)
        d = out.setdefault(key, {"trades":0, "wins":0, "pnl_sum":0.0, "R_sum":0.0})
        d["trades"] += 1
        d["pnl_sum"] += safe_float(r.get("pnl"), 0.0)
        R = safe_float(r.get("R"), 0.0)
        d["R_sum"] += R
        if R > 0:
            d["wins"] += 1
    # finalize
    res = []
    for key, d in out.items():
        rec = {k:v for k,v in zip(key_fields, key)}
        rec.update(d)
        rec["avg_R"] = (d["R_sum"]/d["trades"]) if d["trades"]>0 else 0.0
        rec["win_rate"] = (d["wins"]/d["trades"]) if d["trades"]>0 else 0.0
        res.append(rec)
    # sort by pnl_sum desc
    res.sort(key=lambda x:(-x["pnl_sum"], -x["avg_R"], -x["win_rate"]))
    return res

def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k:r.get(k,"") for k in fields})

def main():
    ap = argparse.ArgumentParser(description="Analyze ES bot logs for profitability by session/hour/arm/side")
    ap.add_argument("--in", dest="inp", required=True, help="Input file or folder (recursively scans)")
    ap.add_argument("--out", dest="out", required=True, help="Output folder for CSV reports")
    args = ap.parse_args()

    ensure_dir(args.out)

    cycle_rows = []
    hb_rows = []

    for f in walk_inputs(args.inp):
        try:
            with open(f, "r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    obj = parse_line(line)
                    if not obj or not isinstance(obj, dict):
                        continue
                    evt = obj.get("evt") or obj.get("event")
                    if evt == "flat_cycle":
                        # Flatten to normalized record
                        sess = obj.get("session") or obj.get("sess") or obj.get("session_id")
                        rec = {
                            "ts": obj.get("ts"),
                            "hour": to_hour(obj.get("ts")),
                            "session_key": sess,
                            "session": get_session(sess),
                            "arm": obj.get("arm","unknown"),
                            "side": obj.get("side","unknown"),
                            "qty": safe_int(obj.get("qty"), 0),
                            "pnl": safe_float(obj.get("pnl"), 0.0),
                            "R": safe_float(obj.get("R"), 0.0),
                            "equity": safe_float(obj.get("equity"), 0.0),
                            "adx": safe_float(obj.get("adx"), float("nan")),
                            "atrp": safe_float(obj.get("atrp"), float("nan")),
                            "symbol": obj.get("symbol","ES"),
                            "entry_ts": obj.get("entry_ts"),
                            "exit_ts": obj.get("exit_ts"),
                        }
                        cycle_rows.append(rec)
                    elif evt == "hb":
                        sess = obj.get("session")
                        hb_rows.append({
                            "ts": obj.get("ts"),
                            "hour": to_hour(obj.get("ts")),
                            "session_key": sess,
                            "session": get_session(sess),
                            "state": obj.get("state"),
                            "idle_reason": obj.get("idle_reason"),
                            "caps": ",".join(obj.get("caps", [])) if isinstance(obj.get("caps"), list) else obj.get("caps"),
                            "rt_status": obj.get("rt_status"),
                            "rt_age_sec": obj.get("rt_age_sec"),
                            "bars": obj.get("bars"),
                        })
        except Exception as e:
            print(f"[WARN] Failed {f}: {e}", file=sys.stderr)

    # Summaries
    by_session = summarize(cycle_rows, ["session"])
    by_hour = summarize(cycle_rows, ["hour"])
    by_arm = summarize(cycle_rows, ["arm"])
    by_side = summarize(cycle_rows, ["side"])
    by_session_arm = summarize(cycle_rows, ["session","arm"])
    by_session_hour = summarize(cycle_rows, ["session","hour"])

    # Write CSVs
    write_csv(os.path.join(args.out, "trades_raw.csv"), cycle_rows,
              ["ts","hour","session_key","session","arm","side","qty","pnl","R","equity","adx","atrp","symbol","entry_ts","exit_ts"])
    write_csv(os.path.join(args.out, "hb_raw.csv"), hb_rows,
              ["ts","hour","session_key","session","state","idle_reason","caps","rt_status","rt_age_sec","bars"])
    write_csv(os.path.join(args.out, "summary_by_session.csv"), by_session,
              ["session","trades","wins","win_rate","avg_R","R_sum","pnl_sum"])
    write_csv(os.path.join(args.out, "summary_by_hour.csv"), by_hour,
              ["hour","trades","wins","win_rate","avg_R","R_sum","pnl_sum"])
    write_csv(os.path.join(args.out, "summary_by_arm.csv"), by_arm,
              ["arm","trades","wins","win_rate","avg_R","R_sum","pnl_sum"])
    write_csv(os.path.join(args.out, "summary_by_side.csv"), by_side,
              ["side","trades","wins","win_rate","avg_R","R_sum","pnl_sum"])
    write_csv(os.path.join(args.out, "summary_by_session_arm.csv"), by_session_arm,
              ["session","arm","trades","wins","win_rate","avg_R","R_sum","pnl_sum"])
    write_csv(os.path.join(args.out, "summary_by_session_hour.csv"), by_session_hour,
              ["session","hour","trades","wins","win_rate","avg_R","R_sum","pnl_sum"])

    # Minimal console output
    def topn(rows, n=10):
        return rows[:n]

    print("\\n=== Top by session ===")
    for r in topn(by_session): print(r)
    print("\\n=== Top by hour ===")
    for r in topn(by_hour): print(r)
    print("\\n=== Top by arm ===")
    for r in topn(by_arm): print(r)
    print("\\nWrote CSVs to:", args.out)

if __name__ == "__main__":
    main()
