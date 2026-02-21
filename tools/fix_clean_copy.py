import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root
CLEAN = ROOT / "_CLEAN_COPY"

# Prune folders/files that should not be part of the runtime/compile surface
PRUNE_DIRS = [
    CLEAN / "backups",
    CLEAN / "run" / "hb_bundle",
]
PRUNE_FILES = [
    CLEAN / "pt_bootstrap_only.py",
    CLEAN / "pt_site_only.py",
    # self-tests / scratch often contain broken snippets
    CLEAN / "pt" / "ledger_selftest.py",
]

# Regex for extracting repeated "from X import Y" segments on one line
FROM_SEG_RE = re.compile(
    r"(from\s+[A-Za-z0-9_\.]+\s+import\s+[^#\n]+?)(?=from\s+[A-Za-z0-9_\.]+\s+import\s+|$)"
)

def fix_text(s: str) -> str:
    """
    Repairs common corruption patterns seen in this repo cleanup:
      - glued import statements (multiple imports on one physical line)
      - 'import Xfrom Y import Z' glue (missing newline between import and from)
      - delimiter artifacts from earlier passes (|from / |import / trailing |)
      - '...except Exception:' glued to import line
      - accidental "import after argv/env sanitization####" style damage
    """

    # ---- Remove prior delimiter artifacts ----
    s = re.sub(r"\|from\s+", "\nfrom ", s)
    s = re.sub(r"\|import\s+", "\nimport ", s)
    s = re.sub(r"(?m)\|\s*$", "", s)

    # ---- Fix "import after argv/env sanitization####" -> comment ----
    s = re.sub(r"(?m)^\s*import\s+after\s+argv/env\s+sanitization.*$",
               "# after argv/env sanitization", s)

    # ---- Fix "import Xfrom Y import Z" (with or without glue) ----
    # Examples:
    #   import pt.order_corefrom pt.position_core import compute_position
    #   import pt.utilsfrom pt.ib_core import connect_ib
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+?)\s*$",
        r"\1import \2\n\1from \3 import \4",
        s
    )
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+?)\s*$",
        r"\1import \2\n\1from \3 import \4",
        s
    )
    # glue without whitespace between module and from: "...utilsfrom pt.ib_core ..."
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+?)\s*$",
        r"\1import \2\n\1from \3 import \4",
        s
    )
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+?)\s*$",
        r"\1import \2\n\1from \3 import \4",
        s
    )
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+?)\s*$",
        r"\1import \2\n\1from \3 import \4",
        s
    )
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+?)\s*$",
        r"\1import \2\n\1from \3 import \4",
        s
    )
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+?)\s*$",
        r"\1import \2\n\1from \3 import \4",
        s
    )
    # Actual glue token variant: "...utilsfrom pt.ib_core" (no space before 'from')
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+",
        r"\1import \2\n\1from ",
        s
    )
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+",
        r"\1import \2\n\1from ",
        s
    )
    # And the no-space "utilsfrom" case:
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+",
        r"\1import \2\n\1from ",
        s
    )
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+",
        r"\1import \2\n\1from ",
        s
    )
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+",
        r"\1import \2\n\1from ",
        s
    )
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+",
        r"\1import \2\n\1from ",
        s
    )
    s = re.sub(
        r"(?m)^(\s*)import\s+([A-Za-z0-9_\.]+)from\s+",
        r"\1import \2\n\1from ",
        s
    )

    # ---- Convert glued "from ... import ...except Exception:" into proper try/except import ----
    # Pattern seen in pt/hb_core.py and pt/trade_bridge.py
    s = re.sub(
        r"(?m)^(\s*)from\s+([A-Za-z0-9_\.]+)\s+import\s+(.+?)except\s+Exception\s*:\s*$",
        r"\1try:\n\1    from \2 import \3\n\1except Exception:",
        s
    )

    def split_multi_imports(line: str) -> str:
        if not line.lstrip().startswith("import "):
            return line
        work = line
        work = re.sub(r"([A-Za-z0-9_\.])import\s+", r"\1|import ", work)
        work = re.sub(r"\s+import\s+", r"|import ", work)
        if "|import " not in work:
            return line
        parts = [p.strip() for p in work.split("|") if p.strip()]
        indent = re.match(r"^(\s*)", line).group(1)
        return "\n".join(indent + p for p in parts)

    def split_multi_froms(line: str) -> str:
        if not line.lstrip().startswith("from "):
            return line
        work = re.sub(r"([A-Za-z0-9_\]\)\}])from\s+", r"\1|from ", line)
        segs = [m.group(1).strip() for m in FROM_SEG_RE.finditer(work)]
        if len(segs) <= 1:
            return line
        indent = re.match(r"^(\s*)", line).group(1)
        return "\n".join(indent + seg for seg in segs)

    def split_import_and_code(one: str) -> str:
        if not (one.lstrip().startswith("import ") or one.lstrip().startswith("from ")):
            return one
        if "#" in one:
            left, right = one.split("#", 1)
            hash_part = "#" + right
        else:
            left, hash_part = one, ""
        m = re.match(r"^(\s*)(import\s+.+?|from\s+.+?\s+import\s+.+?)(\s{2,})(\S.+)$", left)
        if m:
            ind, imp, _, code = m.group(1), m.group(2), m.group(3), m.group(4)
            return f"{ind}{imp}\n{ind}{code}{hash_part}"
        return one + hash_part

    out = []
    for raw in s.splitlines(True):
        nl = "\n" if raw.endswith("\n") else ""
        line = raw[:-1] if nl else raw

        line = split_multi_froms(line)

        pieces = []
        for piece in line.split("\n"):
            pieces.append(split_multi_imports(piece))
        line = "\n".join(pieces)

        line = "\n".join(split_import_and_code(x) for x in line.split("\n"))

        line = re.sub(r"\|\s*$", "", line)
        out.append(line + nl)

    return "".join(out)

def prune_venv_like_dirs():
    # Remove any ".venv*" folders that got copied into _CLEAN_COPY (e.g. .venv_BAD_*)
    for p in CLEAN.iterdir():
        if p.is_dir() and p.name.lower().startswith(".venv"):
            shutil.rmtree(p, ignore_errors=True)
            print(f"[PRUNE] removed dir {p}")

def main():
    if not CLEAN.exists():
        raise SystemExit(f"_CLEAN_COPY not found at: {CLEAN}")

    prune_venv_like_dirs()

    for d in PRUNE_DIRS:
        if d.exists():
            shutil.rmtree(d, ignore_errors=True)
            print(f"[PRUNE] removed dir {d}")
    for f in PRUNE_FILES:
        if f.exists():
            f.unlink()
            print(f"[PRUNE] removed file {f}")

    changed = 0
    for p in CLEAN.rglob("*.py"):
        parts = [x.lower() for x in p.parts]
        if "__pycache__" in parts:
            continue
        txt = p.read_text(encoding="utf-8", errors="replace")
        new = fix_text(txt)
        if new != txt:
            p.write_text(new, encoding="utf-8", newline="\n")
            changed += 1
    print(f"[FIX] files changed: {changed}")

    import py_compile
    errors = []
    for p in CLEAN.rglob("*.py"):
        parts = [x.lower() for x in p.parts]
        if "__pycache__" in parts:
            continue
        # ignore any remaining venv-like paths defensively
        if any(part.lower().startswith(".venv") for part in p.parts):
            continue
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as e:
            errors.append(f"{p}: {e}")

    errf = CLEAN / "COMPILE_ERRORS_POSTFIX.txt"
    if errors:
        errf.write_text("\n".join(errors), encoding="utf-8")
        print(f"[WARN] compile errors remain: {len(errors)} (see COMPILE_ERRORS_POSTFIX.txt)")
    else:
        if errf.exists():
            errf.unlink()
        print("[OK] all files compile after postfix")

if __name__ == "__main__":
    main()
