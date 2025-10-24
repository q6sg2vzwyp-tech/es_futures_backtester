# analyze_rewards.py
import csv
import datetime as dt
import pathlib
import re
import sys
from collections import defaultdict

RE = re.compile(
    r"^(?P<src>.*?):\d+:\[INFO\]\s+(?P<ts>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2})\s+\[REWARD\]\s+"
    r"arm=(?P<arm>\w+)\s+R=(?P<R>-?\d+(?:\.\d+)?)\s+slipR=(?P<slipR>-?\d+(?:\.\d+)?)\s+"
    r"hrs=(?P<hrs>\d+(?:\.\d+)?)\s+maeR=(?P<maeR>-?\d+(?:\.\d+)?)\s+reward=(?P<reward>-?\d+(?:\.\d+)?)\s+"
    r"dailyR=(?P<dailyR>-?\d+(?:\.\d+)?)"
)


def parse_line(line):
    m = RE.search(line)
    if not m:
        return None
    d = m.groupdict()
    d["ts"] = dt.datetime.strptime(d["ts"], "%Y-%m-%d %H:%M:%S")
    for k in ("R", "slipR", "hrs", "maeR", "reward", "dailyR"):
        d[k] = float(d[k])
    d["day"] = d["ts"].date().isoformat()
    d["hour"] = d["ts"].strftime("%H")
    return d


def main(infile):
    inpath = pathlib.Path(infile)
    outdir = inpath.parent
    csv_path = outdir / "rewards_parsed.csv"
    sum_path = outdir / "rewards_summary.txt"

    rows = []
    with open(inpath, encoding="utf-8", errors="ignore") as f:
        for line in f:
            d = parse_line(line)
            if d:
                rows.append(d)

    # write CSV
    cols = [
        "ts",
        "day",
        "hour",
        "arm",
        "R",
        "slipR",
        "hrs",
        "maeR",
        "reward",
        "dailyR",
        "src",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({k: (r[k] if k != "ts" else r[k].isoformat(sep=" ")) for k in cols})

    # compute summaries
    by_arm = defaultdict(list)
    by_day = defaultdict(list)
    by_hour = defaultdict(list)
    for r in rows:
        by_arm[r["arm"]].append(r)
        by_day[r["day"]].append(r)
        by_hour[r["hour"]].append(r)

    def stats(items):
        if not items:
            return {}
        R = [x["R"] for x in items]
        wins = sum(1 for x in R if x > 0)
        losses = sum(1 for x in R if x < 0)
        return {
            "n": len(R),
            "wins": wins,
            "losses": losses,
            "winrate%": round(100 * wins / len(R), 1),
            "sumR": round(sum(R), 3),
            "avgR": round(sum(R) / len(R), 3),
            "medianR": round(
                (
                    sorted(R)[len(R) // 2]
                    if len(R) % 2 == 1
                    else (sorted(R)[len(R) // 2 - 1] + sorted(R)[len(R) // 2]) / 2
                ),
                3,
            ),
            "avgMAE_R": round(sum(x["maeR"] for x in items) / len(items), 3),
            "avgHrs": round(sum(x["hrs"] for x in items) / len(items), 3),
        }

    def topk(items, k=5, key="R", reverse=True):
        return sorted(items, key=lambda x: x[key], reverse=reverse)[:k]

    with open(sum_path, "w", encoding="utf-8") as f:
        f.write(f"Parsed trades: {len(rows)}\n\n")

        f.write("== By arm ==\n")
        for arm, items in sorted(by_arm.items()):
            f.write(f"{arm}: {stats(items)}\n")
        f.write("\n== By day ==\n")
        for day, items in sorted(by_day.items()):
            f.write(f"{day}: {stats(items)}\n")

        f.write("\n== Hour-of-day (counts / avgR) ==\n")
        for hour in sorted(by_hour):
            s = stats(by_hour[hour])
            f.write(f"{hour}: n={s.get('n',0)} avgR={s.get('avgR',0)}\n")

        # outliers
        f.write("\n== Top winners ==\n")
        for x in topk(rows, 5, "R", True):
            f.write(f"{x['ts']} arm={x['arm']} R={x['R']} maeR={x['maeR']} hrs={x['hrs']}\n")
        f.write("\n== Top losers ==\n")
        for x in topk(rows, 5, "R", False):
            f.write(f"{x['ts']} arm={x['arm']} R={x['R']} maeR={x['maeR']} hrs={x['hrs']}\n")

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {sum_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_rewards.py <path to reward_lines_*.txt>")
        sys.exit(2)
    main(sys.argv[1])
