from __future__ import annotations
import ast, os, sys, json
from collections import deque

DROP_DIRS = {".venv","logs","run","results","__pycache__",".git","release","archive","_archive_broken_IGNORE","pt_v7_update","_pt_v5_tmp"}

def iter_py_files(root: str):
    for dp, dn, fn in os.walk(root):
        dn[:] = [d for d in dn if d not in DROP_DIRS and not d.startswith(".")]
        for f in fn:
            if f.endswith(".py"):
                yield os.path.join(dp, f)

def resolve_import_to_path(root: str, mod: str) -> str | None:
    p1 = os.path.join(root, mod.replace(".", os.sep) + ".py")
    if os.path.isfile(p1): return p1
    p2 = os.path.join(root, mod.replace(".", os.sep), "__init__.py")
    if os.path.isfile(p2): return p2
    return None

def parse_imports(py_path: str) -> set[str]:
    try:
        src = open(py_path, "r", encoding="utf-8").read()
        tree = ast.parse(src, filename=py_path)
    except Exception:
        return set()

    mods = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                if a.name: mods.add(a.name)
        elif isinstance(n, ast.ImportFrom):
            if n.module: mods.add(n.module)
    # include top-level parts too
    out = set()
    for m in mods:
        parts = m.split(".")
        out.add(parts[0])
        out.add(m)
    return out

def main():
    root = os.path.abspath(sys.argv[1])
    entries = [os.path.join(root, p) for p in sys.argv[2:]]

    q = deque([p for p in entries if os.path.isfile(p)])
    reachable = set(os.path.abspath(p) for p in q)

    while q:
        p = q.popleft()
        for mod in parse_imports(p):
            cand = resolve_import_to_path(root, mod)
            if cand:
                cand = os.path.abspath(cand)
                if cand not in reachable:
                    reachable.add(cand)
                    q.append(cand)

    print(json.dumps({"root": root, "reachable_paths": sorted(reachable)}, indent=2))

if __name__ == "__main__":
    main()
