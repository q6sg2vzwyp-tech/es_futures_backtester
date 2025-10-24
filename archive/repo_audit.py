import re
import sys
from datetime import datetime
from pathlib import Path

PY_IMPORT_RE = re.compile(
    r"^\s*(?:from\s+([a-zA-Z0-9_\.]+)\s+import|import\s+([a-zA-Z0-9_\.]+))", re.M
)
PY_CALL_RE = re.compile(r"python(?:\s|\.exe\s)+([^\s]+\.py)\b", re.I)


def scan_repo(root: Path):
    py_files, bat_files, other = [], [], []
    for p in root.rglob("*"):
        if p.is_dir() or p.name.startswith("."):
            continue
        if p.suffix.lower() == ".py":
            py_files.append(p)
        elif p.suffix.lower() in (".bat", ".cmd"):
            bat_files.append(p)
        else:
            other.append(p)
    return py_files, bat_files, other


def parse_imports(py_path: Path):
    try:
        txt = py_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()
    mods = set()
    for m in PY_IMPORT_RE.finditer(txt):
        mod = m.group(1) or m.group(2)
        if mod:
            mods.add(mod.split(".")[0])
    return mods


def parse_batch_calls(bat_path: Path):
    try:
        txt = bat_path.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return set()
    calls = set()
    for m in PY_CALL_RE.finditer(txt):
        calls.add(Path(m.group(1)).name)
    return calls


def main():
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(".").resolve()
    py_files, bat_files, other = scan_repo(root)

    module_to_file = {p.stem: p for p in py_files}
    imports_by_file = {}
    imported_by = {p: set() for p in py_files}

    for p in py_files:
        imports = parse_imports(p)
        imports_by_file[p] = imports
        for mod in imports:
            if mod in module_to_file:
                imported_by[module_to_file[mod]].add(p)

    bat_refs = {p: set() for p in py_files}
    for b in bat_files:
        calls = parse_batch_calls(b)
        for call in calls:
            for p in py_files:
                if p.name == call:
                    bat_refs[p].add(b)

    entrypoints = {p for p in py_files if not imported_by[p]}

    rows = []
    for p in py_files:
        mtime = datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds")
        rows.append(
            {
                "file": str(p.relative_to(root)),
                "module": p.stem,
                "imported_by_count": len(imported_by[p]),
                "imported_by": [str(x.relative_to(root)) for x in sorted(imported_by[p])],
                "imports": sorted(list(imports_by_file[p])),
                "referenced_in_bat": [str(x.relative_to(root)) for x in sorted(bat_refs[p])],
                "is_entrypoint_candidate": p in entrypoints,
                "last_modified": mtime,
            }
        )

    LIKELY_RUNNERS = {
        "paper_trader",
        "backtest_engine",
        "strategy_runner",
        "batch_runner",
        "streamlit_dashboard",
        "optimizer",
        "report_generator",
        "ib_quote",
    }
    likely_unused = []
    for r in rows:
        stem = Path(r["file"]).stem
        if (
            r["imported_by_count"] == 0
            and len(r["referenced_in_bat"]) == 0
            and stem not in LIKELY_RUNNERS
        ):
            likely_unused.append(r)

    out1 = root / "repo_audit_files.csv"
    out2 = root / "repo_audit_unused_candidates.csv"
    with out1.open("w", newline="", encoding="utf-8") as f:
        import csv as _csv

        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    with out2.open("w", newline="", encoding="utf-8") as f:
        import csv as _csv

        w = _csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["file"])
        w.writeheader()
        for r in likely_unused:
            w.writerow(r)

    print(f"Wrote: {out1}")
    print(f"Wrote: {out2}")
    print("Done.")


if __name__ == "__main__":
    main()
