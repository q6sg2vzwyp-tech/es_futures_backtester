import sys, io

path = r".\paper_trader.py"
stack = []
pairs = {')':'(',']':'[','}':'{'}
openers = set(pairs.values())

with io.open(path,'r',encoding='utf-8',newline='') as f:
    for ln, line in enumerate(f, start=1):
        for col, ch in enumerate(line, start=1):
            if ch in openers:
                stack.append((ch, ln, col))
            elif ch in pairs:
                if not stack or stack[-1][0] != pairs[ch]:
                    print(f"Unmatched closer '{ch}' at line {ln}, col {col}")
                    sys.exit(1)
                stack.pop()

if stack:
    ch, ln, col = stack[-1]
    print(f"Unclosed opener '{ch}' opened at line {ln}, col {col}")
    sys.exit(1)

print("Brackets look balanced.")
