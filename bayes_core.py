# bayes_core.py
import os
import json
import tempfile
from typing import Dict, Any, Tuple, Optional, List
from learner_bayes import run_eod_bayes_opt
import utils

def build_bayes_training_set(src_csv: str, dst_csv: str, logger) -> str:
    # (your existing implementation)
    ...

def maybe_apply_bayes_best(args, logger, best_params_path: str) -> str:
    """
    Returns BAYES_SOURCE string ("cli" or "bayes_best") so paper_trader
    doesn't need the global.
    """
    BAYES_SOURCE = "cli"
    # (your existing logic, but use best_params_path instead of LEARN_BAYES_BEST)
    return BAYES_SOURCE

def run_eod_bayes_opt_filtered(
    trades_csv: str,
    best_params_path: str,
    param_space: Dict[str, Tuple[Any, Any]],
    ignore_reasons: Optional[List[str]] = None,
):
    # (your existing wrapper, calling run_eod_bayes_opt)
    ...

