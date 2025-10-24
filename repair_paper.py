import ast, os, re, sys

PATH = r".\paper_trader.py"

def load():
    return open(PATH, "r", encoding="utf-8", errors="ignore").read()

def normalize(s: str) -> str:
    s = s.replace("\r\n","\n").replace("\r","\n")
    s = re.sub(r'[^\t\n\r\x20-\x7E]', '', s)  # strip non-ASCII (keep TAB/NL)
    return s

def strip_docstrings(s: str) -> str:
    s = re.sub(r'(?s)("""[\s\S]*?""")', '', s)
    s = re.sub(r"(?s)('''[\s\S]*?''')", '', s)
    return s

BAN_LINE_PATTERNS = [
    r'^\s*-\s',                                 # markdown bullets
    r'^\s*[A-Z]\)\s',                           # A) B) C) …
    r'^\s*Analysis paused\s*$', r'^\s*You said:\s*$', r'^\s*ChatGPT said:\s*$',
    r'^\s*Upgrade complete.*$', r'^\s*(Yes|No|Okay|Got it|Thought for \d+s)\s*$',
    r'^\s*Historical seed \+ realtime 5s bars\s*$',
    r'^\s*The script.*$', r'^\s*Windows PowerShell.*$',
]

def drop_banned_lines(s: str) -> str:
    out = []
    for ln in s.split("\n"):
        if any(re.search(p, ln) for p in BAN_LINE_PATTERNS):
            continue
        out.append(ln)
    return "\n".join(out)

def keep_probable_code(s: str) -> str:
    kept = []
    for ln in s.split("\n"):
        t = ln.strip()
        if t == "" or t.startswith("#"):
            kept.append(ln); continue
        if re.match(r'^(def|class|from|import|if|elif|else|for|while|try|except|finally|return|with|@)\b', t):
            kept.append(ln); continue
        if re.search(r'[\(\)\[\]\{\}:=,\.]', ln):
            kept.append(ln); continue
        # looks like prose -> drop
    return "\n".join(kept)

def first_raw_mismatch(s: str):
    stack = []
    pairs = {')':'(', ']':'[', '}':'{'}
    opens = set(pairs.values())
    for i,ch in enumerate(s, 1):
        if ch in opens:
            stack.append((ch,i))
        elif ch in pairs:
            if not stack or stack[-1][0] != pairs[ch]:
                line = s.count("\n",0,i)+1
                col  = i - (s.rfind("\n",0,i)+1)
                return ("mismatch", line, col)
            stack.pop()
    return (None, None, None)

def drop_line(s: str, line_no: int) -> str:
    lines = s.split("\n")
    if 1 <= line_no <= len(lines):
        # Also drop lonely ) ] } lines aggressively
        if re.match(r'^\s*[\)\]\}]\s*$', lines[line_no-1] or ""):
            lines.pop(line_no-1)
        else:
            lines.pop(line_no-1)
    return "\n".join(lines)

def ast_ok(s: str):
    try:
        ast.parse(s)
        return True, None
    except SyntaxError as e:
        return False, e

RAW = load()
txt = normalize(RAW)
txt = strip_docstrings(txt)
txt = drop_banned_lines(txt)
txt = keep_probable_code(txt)

# Extra surgical fixes for functions that often got mangled
# parse_ct_list: ensure except has 'pass'
txt = re.sub(
    r'(?ms)(def\s+parse_ct_list\([^\)]*\):.*?for\s+chunk\s+in\s+spec\.split\([^\)]*\):.*?try:\s*?\n)(\s*except\s+Exception:\s*?\n)(?!\s+pass)',
    r'\1\2            pass\n',
    txt
)
# reset_due_multi: ensure continue after today-check
txt = re.sub(
    r'(?ms)(def\s+reset_due_multi\([^\)]*\):.*?for\s+ct\s+in\s+reset_times:.*?\n\s*label\s*=\s*ct\.strftime\([^\)]*\)\s*\n\s*if\s+last_reset_marks\.get\(label\)\s*==\s*today:\s*\n)(?!\s*continue\b)',
    r'\1            continue\n',
    txt
)

# Iteratively fix raw bracket mismatches by dropping the offending line
for _ in range(20):
    kind, ln, col = first_raw_mismatch(txt)
    if kind is None:
        break
    # Print context for debugging
    lines = txt.split("\n")
    a,b = max(1,ln-2), min(len(lines), ln+2)
    sys.stdout.write(f"[RAW MISMATCH] line {ln}, col {col}\n")
    for n in range(a,b+1):
        sys.stdout.write(f"{n:6d}: {lines[n-1]}\n")
    # Drop the bad line
    txt = drop_line(txt, ln)

ok, err = ast_ok(txt)
if not ok:
    lines = txt.split("\n")
    a,b = max(1,(err.lineno or 1)-3), min(len(lines),(err.lineno or 1)+3)
    sys.stdout.write(f"[AST ERROR] {err.msg} at line {err.lineno}, col {err.offset}\n")
    for n in range(a,b+1):
        sys.stdout.write(f"{n:6d}: {lines[n-1]}\n")
    sys.exit(3)

backup = PATH + ".repair_bak"
try:
    if not os.path.exists(backup):
        open(backup, "wb").write(open(PATH,"rb").read())
except Exception:
    pass

open(PATH, "w", encoding="utf-8", newline="\n").write(txt)
print("[OK] Repaired & compiled. Backup:", backup)
