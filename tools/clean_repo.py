import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

PY_EXT = ".py"

@dataclass
class FileInfo:
    path: Path
    rel: str
    sha1: str
    size: int

def sha1_file(p: Path) -> str:
    h = hashlib.sha1()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def iter_py_files(src: Path) -> List[Path]:
    out = []
    for p in src.rglob("*"):
        if not (p.is_file() and p.suffix.lower() == PY_EXT):
            continue

        parts = [x.lower() for x in p.parts]

        # skip venv, caches, and known junk/archives
        if ".venv" in parts or "venv" in parts or "__pycache__" in parts:
            continue
        if "backups" in parts:
            continue
        # skip run/hb_bundle entirely
        if "run" in parts:
            try:
                i = parts.index("run")
                if i + 1 < len(parts) and parts[i + 1] == "hb_bundle":
                    continue
            except ValueError:
                pass

        # skip these accidental PowerShell files
        if p.name.lower() in ("pt_bootstrap_only.py", "pt_site_only.py"):
            continue

        out.append(p)
    return out

def module_name_from_rel(rel_path: str) -> str:
    # convert "pt/foo/bar.py" -> "pt.foo.bar"
    rel = rel_path.replace("\\", "/")
    if rel.endswith(".py"):
        rel = rel[:-3]
    rel = rel.strip("/")
    # drop __init__ special case: "pt/__init__.py" -> "pt"
    rel = rel.replace("/__init__", "")
    return rel.replace("/", ".")

def build_manifest(src: Path) -> List[FileInfo]:
    files = []
    for p in iter_py_files(src):
        rel = str(p.relative_to(src))
        files.append(FileInfo(path=p, rel=rel, sha1=sha1_file(p), size=p.stat().st_size))
    return files

def group_exact_dupes(files: List[FileInfo]) -> Dict[str, List[FileInfo]]:
    by_hash: Dict[str, List[FileInfo]] = {}
    for fi in files:
        by_hash.setdefault(fi.sha1, []).append(fi)
    return {h: lst for h, lst in by_hash.items() if len(lst) > 1}

def choose_canonical(paths: List[FileInfo]) -> FileInfo:
    """
    Heuristic:
    1) Prefer under 'pt\\' (or 'pt/')
    2) Prefer shorter path depth
    3) Prefer larger size (often more complete)
    """
    def score(fi: FileInfo) -> Tuple[int, int, int]:
        rel = fi.rel.replace("\\", "/")
        in_pt = 1 if rel.startswith("pt/") else 0
        depth = rel.count("/")
        return (in_pt, -depth, fi.size)
    return sorted(paths, key=score, reverse=True)[0]

def load_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def write_text(p: Path, s: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(s, encoding="utf-8", newline="\n")

IMPORT_RE = re.compile(r"^(\s*)(from\s+([a-zA-Z0-9_\.]+)\s+import\s+|import\s+([a-zA-Z0-9_\.]+))", re.M)

def rewrite_imports_text(src_text: str, modmap: Dict[str, str]) -> str:
    """
    Conservative line-level rewrite that preserves original line endings.
    """
    out_lines = []
    for line in src_text.splitlines(True):  # keep line endings
        # Preserve exact newline (could be \n or \r\n)
        m_nl = re.search(r"(\r?\n)$", line)
        nl = m_nl.group(1) if m_nl else ""
        core = line[:-len(nl)] if nl else line  # strip newline for matching

        stripped = core.strip()

        # from A import B
        if stripped.startswith("from "):
            m2 = re.match(r"^(\s*)from\s+([a-zA-Z0-9_\.]+)\s+import\s+(.*)$", core)
            if m2:
                ind, mod, rest = m2.group(1), m2.group(2), m2.group(3)
                if mod in modmap:
                    out_lines.append(f"{ind}from {modmap[mod]} import {rest}{nl}")
                    continue

        # import A
        if stripped.startswith("import "):
            m3 = re.match(r"^(\s*)import\s+([a-zA-Z0-9_\.]+)(.*)$", core)
            if m3:
                ind, mod, tail = m3.group(1), m3.group(2), m3.group(3)
                if mod in modmap:
                    out_lines.append(f"{ind}import {modmap[mod]}{tail}{nl}")
                    continue

        out_lines.append(line)

    return "".join(out_lines)

def py_compile_all(repo: Path) -> Tuple[bool, List[str]]:
    import py_compile
    errors = []
    for p in iter_py_files(repo):
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as e:
            errors.append(f"{p}: {e}")
    return (len(errors) == 0, errors)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="Source repo root")
    ap.add_argument("--out", required=True, help="Output clean copy folder")
    ap.add_argument("--entry", default="", help="Entry file (optional), used for reporting only")
    args = ap.parse_args()

    src = Path(args.src).resolve()
    out = Path(args.out).resolve()

    if out.exists():
        print(f"[ERR] Output folder already exists: {out}")
        sys.exit(2)

    print(f"[1/6] Scanning: {src}")
    manifest = build_manifest(src)
    exact = group_exact_dupes(manifest)

    # Build mapping: all dup modules -> canonical module
    modmap: Dict[str, str] = {}
    removals: List[str] = []
    keepers: List[str] = []

    for h, group in exact.items():
        canon = choose_canonical(group)
        canon_mod = module_name_from_rel(canon.rel)
        keepers.append(canon.rel)
        for fi in group:
            if fi.rel == canon.rel:
                continue
            modmap[module_name_from_rel(fi.rel)] = canon_mod
            removals.append(fi.rel)

    print(f"[2/6] Creating clean copy: {out}")
    shutil.copytree(src, out, ignore=shutil.ignore_patterns(".venv", "venv", "__pycache__", "_CLEAN_COPY"))

    report = []
    report.append("# CLEAN REPORT\n")
    report.append(f"- Source: {src}\n- Output: {out}\n")
    if args.entry:
        report.append(f"- Entry: {args.entry}\n")
    report.append(f"\n## Exact duplicate groups\nFound {len(exact)} hash groups with duplicates.\n")
    report.append(f"- Removed files: {len(removals)}\n- Canonical keepers: {len(keepers)}\n")

    # Apply removals
    print(f"[3/6] Removing exact dupes ({len(removals)})")
    for rel in removals:
        p = out / rel
        if p.exists():
            p.unlink()

    # Rewrite imports in every .py
    print(f"[4/6] Rewriting imports ({len(modmap)} module mappings)")
    for p in iter_py_files(out):
        txt = load_text(p)
        new_txt = rewrite_imports_text(txt, modmap)
        if new_txt != txt:
            write_text(p, new_txt)

    # Write mapping + report
    (out / "patches").mkdir(parents=True, exist_ok=True)
    write_text(out / "mapping.json", json.dumps(modmap, indent=2, sort_keys=True))
    write_text(out / "CLEAN_REPORT.md", "".join(report))

    # Compile gate
    print("[5/6] py_compile all files")
    ok, errs = py_compile_all(out)
    if not ok:
        write_text(out / "COMPILE_ERRORS.txt", "\n".join(errs))
        print(f"[WARN] Compile errors: {len(errs)}. See COMPILE_ERRORS.txt")
    else:
        print("[OK] All files compile")

    # Summary
    print("[6/6] Done.")
    print(f"Output: {out}")
    print(f"Removed: {len(removals)}")
    print(f"Mapped:  {len(modmap)}")

if __name__ == "__main__":
    main()
