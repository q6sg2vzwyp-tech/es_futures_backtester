# tools/pt_manifest.py
import ast, os, sys, json, pathlib, traceback
from collections import defaultdict, deque

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENTRY = ROOT / "paper_trader.py"

def norm(p: pathlib.Path) -> str:
    try:
        return str(p.resolve())
    except Exception:
        return str(p)

def find_local_module(modname: str):
    # best-effort resolver for local modules in repo
    parts = modname.split(".")
    # module.py
    p1 = ROOT.joinpath(*parts).with_suffix(".py")
    if p1.exists():
        return p1
    # package/__init__.py
    p2 = ROOT.joinpath(*parts) / "__init__.py"
    if p2.exists():
        return p2
    return None

def parse_imports(pyfile: pathlib.Path):
    src = pyfile.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(src, filename=str(pyfile))
    mods = set()
    from_mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for n in node.names:
                if n.name:
                    mods.add(n.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                from_mods.add(node.module)
    return sorted(mods), sorted(from_mods)

def main():
    if not ENTRY.exists():
        raise SystemExit(f"Missing: {ENTRY}")

    seen = set()
    edges = defaultdict(list)  # file -> imported module names
    file_of = {}               # module -> file (if local)

    q = deque([ENTRY])
    while q:
        f = q.popleft()
        if f in seen:
            continue
        seen.add(f)

        try:
            imps, froms = parse_imports(f)
        except Exception:
            edges[norm(f)].append({"error": traceback.format_exc()})
            continue

        allmods = sorted(set(imps + froms))
        edges[norm(f)] = allmods

        for m in allmods:
            # only follow local modules
            local = find_local_module(m)
            if local:
                file_of[m] = norm(local)
                q.append(local)

    # also capture pip deps via import resolution (optional best-effort)
    report = {
        "root": norm(ROOT),
        "entry": norm(ENTRY),
        "local_files_count": len(seen),
        "files": sorted([norm(x) for x in seen]),
        "edges": dict(edges),
        "local_module_files": dict(sorted(file_of.items())),
    }

    out = ROOT / "run" / "pt_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[OK] wrote {out}")

if __name__ == "__main__":
    main()
