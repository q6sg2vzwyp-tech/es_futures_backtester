#!/usr/bin/env python3

import re
from pathlib import Path

TARGETS = [
    Path("quick_reward_dump.py"),
    Path("summarize_gates.py"),
    Path("twoweek_review.py"),
    Path("archive/missed_reasons.py"),
    Path("archive/paper_trader_long_only.py"),
    Path("archive/run_ppt_trader_supervisor.py"),
]

EXCPT_RE = re.compile(
    r"""
    (                           # group 1: 'except Exception' header
        except\ +Exception
        (?:\s+as\s+\w+)?        # maybe already 'as e'
        \s*:\s*
    )
    (                           # group 2: body indented one level
        (?:
            [ \t]+(?:pass|continue)\s*\n
        )
    )
    """,
    re.VERBOSE,
)


def ensure_logger(text: str) -> str:
    lines = text.splitlines(True)
    has_import = any(l.startswith("import logging") or l.startswith("from logging") for l in lines)
    has_logger = any("getLogger(__name__)" in l for l in lines)

    insert_at = 0
    for i, l in enumerate(lines[:50]):
        if l.startswith("import") or l.startswith("from"):
            insert_at = i + 1

    out = lines[:]
    changed = False
    if not has_import:
        out.insert(insert_at, "import logging\n")
        insert_at += 1
        changed = True
    if not has_logger:
        out.insert(insert_at, "logger = logging.getLogger(__name__)\n")
        changed = True
    return "".join(out) if changed else text


def rewrite_excepts(text: str, relpath: str) -> str:
    def repl(m):
        head = m.group(1)
        body = m.group(2)
        # normalize 'except Exception' -> 'except Exception as e'
        head = re.sub(r"except +Exception(\s*):", r"except Exception as e:\n", head)
        # keep indentation level of body
        indent = re.match(r"([ \t]+)", body).group(1)
        if "continue" in body:
            return f"{head}{indent}logger.debug('Swallowed exception in {relpath}: %s', e)\n{indent}continue\n"
        else:
            return f"{head}{indent}logger.debug('Swallowed exception in {relpath}: %s', e)\n"

    return EXCPT_RE.sub(repl, text)


def main():
    for path in TARGETS:
        if not path.exists():
            continue
        src = path.read_text(encoding="utf-8")
        bak = path.with_suffix(path.suffix + ".bak")
        bak.write_text(src, encoding="utf-8")

        tmp = ensure_logger(src)
        tmp2 = rewrite_excepts(tmp, str(path))

        if tmp2 != src:
            path.write_text(tmp2, encoding="utf-8")
            print(f"Patched {path} (backup: {bak.name})")
        else:
            print(f"No changes needed: {path}")


if __name__ == "__main__":
    main()
