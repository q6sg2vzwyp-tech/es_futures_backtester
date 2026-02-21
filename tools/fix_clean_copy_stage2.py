# fix_clean_copy_stage2.py
# Auto-generated stage-2 repair tool for _CLEAN_COPY

from __future__ import annotations

from pathlib import Path
import py_compile
from typing import List

ROOT = Path(__file__).resolve().parents[1]
CLEAN = ROOT / "_CLEAN_COPY"

def quarantine(path: Path) -> bool:
    """Rename a .py file out of the import surface."""
    if not path.exists() or path.suffix.lower() != ".py":
        return False
    disabled = path.with_suffix(path.suffix + ".DISABLED")
    if disabled.exists():
        return True
    path.rename(disabled)
    return True

def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", newline="\n")

def patch_trade_bridge(tb: Path) -> bool:
    """Replace pt/trade_bridge.py with a safe minimal implementation."""
    if not tb.exists():
        return False
    content = "\n".join([
        '"""',
        'pt.trade_bridge (stage2 safe)',
        '',
        'Safe minimal implementation to bypass corrupted historical versions.',
        'Provides:',
        '  - new_trade_id()',
        '  - log_event(...)',
        '  - log_trade(...)',
        '',
        'Logs to: run/events.log and run/trades.log',
        '"""',
        '',
        'from __future__ import annotations',
        '',
        'import datetime as _dt',
        'import uuid as _uuid',
        'from pathlib import Path as _Path',
        'from typing import Any, Dict, Optional',
        '',
        '_RUN_DIR = _Path(__file__).resolve().parents[1] / "run"',
        '_RUN_DIR.mkdir(parents=True, exist_ok=True)',
        '',
        '_EVENTS_LOG = _RUN_DIR / "events.log"',
        '_TRADES_LOG = _RUN_DIR / "trades.log"',
        '',
        'def new_trade_id() -> str:',
        '    return _uuid.uuid4().hex',
        '',
        'def _ts() -> str:',
        '    return _dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")',
        '',
        'def log_event(event: str, **fields: Any) -> None:',
        '    try:',
        '        parts = [f"ts={_ts()}", f"event={event}"]',
        '        for k, v in fields.items():',
        '            parts.append(f"{k}={v!r}")',
        '        _EVENTS_LOG.open("a", encoding="utf-8").write(" ".join(parts) + "\\n")',
        '    except Exception:',
        '        pass',
        '',
        'def log_trade(trade: Dict[str, Any], *, trade_id: Optional[str] = None) -> None:',
        '    try:',
        '        tid = trade_id or trade.get("trade_id") or new_trade_id()',
        '        parts = [f"ts={_ts()}", f"trade_id={tid}"]',
        '        for k, v in trade.items():',
        '            if k == "trade_id":',
        '                continue',
        '            parts.append(f"{k}={v!r}")',
        '        _TRADES_LOG.open("a", encoding="utf-8").write(" ".join(parts) + "\\n")',
        '    except Exception:',
        '        pass',
    ])
    write_text(tb, content)
    return True

def patch_hb_core(hb: Path) -> bool:
    """Replace pt/hb_core.py with a safe minimal implementation."""
    if not hb.exists():
        return False
    content = "\n".join([
        '"""',
        'pt.hb_core (stage2 safe)',
        '',
        'Safe minimal heartbeat builder to bypass corrupted import-guard glue.',
        '"""',
        '',
        'from __future__ import annotations',
        '',
        'import datetime as dt',
        'from typing import Any, Dict, Optional',
        '',
        'def build_heartbeat_payload(*, tag: str = "HB", status: str = "OK", msg: str = "", extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:',
        '    payload: Dict[str, Any] = {',
        '        "ts": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),',
        '        "tag": tag,',
        '        "status": status,',
        '        "msg": msg,',
        '    }',
        '    if extra:',
        '        try:',
        '            payload.update(extra)',
        '        except Exception:',
        '            pass',
        '    return payload',
        '',
        'def emit_hb_snapshot(payload: Dict[str, Any]) -> None:',
        '    # Optional dependency; no-op if unavailable',
        '    try:',
        '        from pt.hb_emit import emit_hb_snapshot as _emit  # type: ignore',
        '        _emit(payload)',
        '    except Exception:',
        '        return',
    ])
    write_text(hb, content)
    return True

def compile_all(repo: Path) -> List[str]:
    errors: List[str] = []
    for p in repo.rglob("*.py"):
        if "__pycache__" in (x.lower() for x in p.parts):
            continue
        try:
            py_compile.compile(str(p), doraise=True)
        except Exception as e:
            errors.append(f"{p}: {e}")
    return errors

def main() -> int:
    if not CLEAN.exists():
        print(f"[ERR] _CLEAN_COPY not found: {CLEAN}")
        return 2

    # Quarantine optional broken modules
    if quarantine(CLEAN / "pt" / "bayes_core.py"):
        print("[QUAR] disabled pt/bayes_core.py")
    if quarantine(CLEAN / "pt" / "tiny_learner.py"):
        print("[QUAR] disabled pt/tiny_learner.py")

    # Patch runtime blockers
    if patch_trade_bridge(CLEAN / "pt" / "trade_bridge.py"):
        print("[PATCH] wrote safe pt/trade_bridge.py")
    else:
        print("[WARN] pt/trade_bridge.py not found")

    if patch_hb_core(CLEAN / "pt" / "hb_core.py"):
        print("[PATCH] wrote safe pt/hb_core.py")
    else:
        print("[WARN] pt/hb_core.py not found")

    errors = compile_all(CLEAN)
    out = CLEAN / "COMPILE_ERRORS_STAGE2.txt"
    if errors:
        out.write_text("\n".join(errors), encoding="utf-8")
        print(f"[WARN] compile errors remain: {len(errors)} (see COMPILE_ERRORS_STAGE2.txt)")
        return 1
    if out.exists():
        out.unlink()
    print("[OK] all files compile after stage2")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
