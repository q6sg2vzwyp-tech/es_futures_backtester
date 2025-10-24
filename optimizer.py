from __future__ import annotations

import warnings
from typing import Any

import numpy as np

# Hide noisy duplicate-point warnings from skopt
warnings.filterwarnings(
    "ignore",
    message="The objective has been evaluated at point",
    category=UserWarning,
)

try:
    from skopt import gp_minimize
    from skopt.space import Integer, Real

    HAVE_SKOPT = True
    warnings.filterwarnings(
        "ignore",
        message="The objective has been evaluated at point",
        category=UserWarning,
    )
except Exception:
    HAVE_SKOPT = False


def _build_dims(search_space: dict[str, tuple[float, float]]):
    names: list[str] = []
    dims: list[Any] = []
    is_int: list[bool] = []
    for k, (lo, hi) in search_space.items():
        names.append(k)
        lo_f, hi_f = float(lo), float(hi)
        if lo_f.is_integer() and hi_f.is_integer():
            is_int.append(True)
            dims.append(Integer(int(lo_f), int(hi_f)) if HAVE_SKOPT else (int(lo_f), int(hi_f)))
        else:
            is_int.append(False)
            dims.append(Real(lo_f, hi_f) if HAVE_SKOPT else (lo_f, hi_f))
    return names, dims, is_int


def _vector_to_params(names: list[str], is_int: list[bool], x: list[float]) -> dict[str, Any]:
    return {
        name: (round(v) if as_int else float(v))
        for name, as_int, v in zip(names, is_int, x, strict=True)
    }


def run_bayesian_opt(
    objective_fn,
    search_space: dict[str, tuple[float, float]],
    n_calls: int = 30,
    random_state: int | None = 42,
) -> dict[str, Any]:
    if not search_space:
        _ = objective_fn({})
        return {}

    names, dims, is_int = _build_dims(search_space)
    seen: dict[tuple[float, ...], float] = {}

    def _obj(x):
        key = tuple(x)
        if key in seen:
            return seen[key]
        params = _vector_to_params(names, is_int, x)
        try:
            val = float(objective_fn(params))
            if np.isnan(val) or np.isinf(val):
                val = 1e9
        except Exception:
            val = 1e9
        seen[key] = val
        return val

    if HAVE_SKOPT:
        res = gp_minimize(
            _obj,
            dimensions=dims,
            n_calls=max(10, int(n_calls)),
            random_state=random_state,
            acq_func="gp_hedge",
            n_initial_points=min(15, max(5, int(n_calls / 2))),
            initial_point_generator="lhs",
            xi=0.005,  # smaller exploration knob for better exploitation
        )
        return _vector_to_params(names, is_int, res.x)

    rng = np.random.default_rng(random_state)

    def _sample_one():
        vec = []
        for (lo, hi), as_int in zip(dims, is_int, strict=True):
            if as_int:
                vec.append(int(rng.integers(int(lo), int(hi) + 1)))
            else:
                vec.append(float(rng.uniform(float(lo), float(hi))))
        return vec

    best_x, best_score = None, float("inf")
    trials = max(30, int(n_calls))
    for _ in range(trials):
        x = _sample_one()
        try:
            score = float(_obj(x))
        except Exception:
            score = float("inf")
        if score < best_score:
            best_score, best_x = score, x

    if best_x is None:
        mids = []
        for (lo, hi), as_int in zip(dims, is_int, strict=True):
            mid = (float(lo) + float(hi)) / 2.0
            mids.append(round(mid) if as_int else float(mid))
        best_x = mids

    return _vector_to_params(names, is_int, best_x)
