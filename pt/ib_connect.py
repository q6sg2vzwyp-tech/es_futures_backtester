
def connect_with_retries(ib, args, log):
    base = int(args.clientId)
    for i in range(max(1, int(args.connect_attempts))):
        cid = base + i
        try:
            print(f"[CONNECT] Attempt {i+1}/{args.connect_attempts} -> clientId={cid}")
            log("boot_progress", step="connecting")
            ib.connect(args.host, args.port, clientId=cid, timeout=args.connect_timeout_sec)
            ib.sleep(0.6)
            if ib.isConnected():
                print(f"Connected (clientId={cid})")
                try:
                    accts = ib.managedAccounts()
                    print(f"[POST-CONNECT] managedAccounts: {accts}")
                except Exception:
                    pass
                return cid
        except Exception as e:
            print(f"[CONNECT] Failed: {repr(e)}")
            try:
                ib.disconnect()
            except Exception:
                pass
            ib.sleep(0.5 + 0.25 * i)
    return None

# === IB_BOOTSTRAP_EXTRACT v1 BEGIN ===

from dataclasses import dataclass
from typing import Optional, Any

# NOTE:
# This section is intentionally self-contained and safe to inject into an existing pt.ib_connect module.
# It avoids importing project-specific modules other than optional logger usage to prevent circular imports.

@dataclass(frozen=True)
class IBConnSpec:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1111
    readonly: bool = True
    timeout: float = 6.0
    account: Optional[str] = None  # optional account selection

def _coerce_int(x: Any, default: int) -> int:
    try:
        return int(x)
    except Exception:
        return default

def _coerce_float(x: Any, default: float) -> float:
    try:
        return float(x)
    except Exception:
        return default

def spec_from_args(args: Any, *, defaults: Optional[IBConnSpec] = None) -> IBConnSpec:
    """
    Create an IBConnSpec from a parsed args namespace (argparse-like).

    Designed to work with common paper_trader args fields:
      args.host, args.port, args.clientId (or args.client_id), args.readonly (optional), args.timeout (optional)
    """
    d = defaults or IBConnSpec()
    host = getattr(args, "host", d.host) or d.host
    port = _coerce_int(getattr(args, "port", d.port), d.port)
    client_id = _coerce_int(getattr(args, "clientId", None) or getattr(args, "client_id", d.client_id), d.client_id)
    readonly = bool(getattr(args, "readonly", d.readonly))
    timeout = _coerce_float(getattr(args, "timeout", d.timeout), d.timeout)
    account = getattr(args, "account", None) or getattr(args, "ib_account", None) or d.account
    return IBConnSpec(host=host, port=port, client_id=client_id, readonly=readonly, timeout=timeout, account=account)

def bootstrap_ib(
    spec: IBConnSpec,
    *,
    logger: Optional[Any] = None,
    connect_fn: Optional[Any] = None,
) -> Any:
    """
    Centralized IB bootstrap used by paper_trader orchestrator.

    - spec: connection spec
    - logger: optional object with .info/.warning/.error
    - connect_fn: optional override for dependency injection/testing

    Returns:
      ib object (usually ib_insync.IB instance) that is connected.
    """
    # Lazy import to avoid import-time side effects/circular imports
    try:
        from ib_insync import IB  # type: ignore
    except Exception as e:
        raise RuntimeError("ib_insync is required for IB bootstrap but could not be imported") from e

    ib = IB() if connect_fn is None else connect_fn()

    if logger:
        logger.info(f"[IB] Connecting host={spec.host} port={spec.port} clientId={spec.client_id} readonly={spec.readonly} timeout={spec.timeout}")

    # ib_insync: IB.connect(host, port, clientId=..., readonly=..., timeout=...)
    try:
        ib.connect(spec.host, spec.port, clientId=spec.client_id, readonly=spec.readonly, timeout=spec.timeout)
    except TypeError:
        # fallback for older signatures
        ib.connect(spec.host, spec.port, clientId=spec.client_id, readonly=spec.readonly)

    # Optional account selection (safe no-op if not applicable)
    if spec.account and logger:
        logger.info(f"[IB] Account preference supplied: {spec.account}")

    if logger:
        try:
            ok = bool(getattr(ib, "isConnected")() if callable(getattr(ib, "isConnected", None)) else getattr(ib, "isConnected", False))
            logger.info(f"[IB] Connected={ok}")
        except Exception:
            logger.info("[IB] Connected (status unknown)")

    return ib

def bootstrap_ib_from_args(args: Any, *, logger: Optional[Any] = None) -> Any:
    """
    Convenience wrapper: builds IBConnSpec from args and connects.
    """
    spec = spec_from_args(args)
    return bootstrap_ib(spec, logger=logger)

# === IB_BOOTSTRAP_EXTRACT v1 END ===


# === CONNECT_EXISTING_HELPER v1 BEGIN ===
from dataclasses import dataclass, replace
from typing import Optional, Any

@dataclass(frozen=True)
class _IBConnSpec:
    host: str = "127.0.0.1"
    port: int = 4002
    client_id: int = 1111
    readonly: bool = True
    timeout: float = 6.0

def _spec_from_args(args: Any) -> _IBConnSpec:
    host = getattr(args, "host", "127.0.0.1") or "127.0.0.1"
    port = int(getattr(args, "port", 4002))
    client_id = int(getattr(args, "clientId", 1111))
    readonly = bool(getattr(args, "readonly", True))
    timeout = float(getattr(args, "connect_timeout_sec", 6.0))
    return _IBConnSpec(host=host, port=port, client_id=client_id, readonly=readonly, timeout=timeout)

def connect_existing_ib_from_args(ib: Any, args: Any, *, client_id: Optional[int] = None, logger: Optional[Any] = None) -> Any:
    spec = _spec_from_args(args)
    if client_id is not None:
        spec = replace(spec, client_id=int(client_id))
    # NOTE: logger is expected to be your existing log(...) function (safe if absent/mismatched)
    if logger:
        try:
            logger("boot_progress", step="connecting", host=spec.host, port=spec.port, clientId=spec.client_id)
        except Exception:
            pass
    try:
        ib.connect(spec.host, spec.port, clientId=spec.client_id, timeout=spec.timeout)
    except TypeError:
        ib.connect(spec.host, spec.port, clientId=spec.client_id)
    return ib
# === CONNECT_EXISTING_HELPER v1 END ===
