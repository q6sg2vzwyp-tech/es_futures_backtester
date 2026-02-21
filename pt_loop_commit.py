def commit_ctx(ctx: dict, **fields) -> dict:
    try:
        ctx.update(fields)
    except Exception:
        pass
    return ctx
