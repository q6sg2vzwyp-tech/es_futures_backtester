# config_core.py
from __future__ import annotations

import os
import datetime as dt
from dataclasses import dataclass
from typing import Optional, Set
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class Config:
    # Base dirs
    BASE_DIR: str
    LOG_DIR: str
    LEARN_DIR: str

    # Files
    HB_PATH: str
    LEARN_MODEL_PATH: str
    LEARN_BAYES_BEST: str
    TRADE_LOG_CSV: str
    BAYES_TRAIN_CSV: str
    DAY_STATE_JSON: str
    DAILY_RESTART_JSON: str
    RUNTIME_STATE_JSON: str
    SHADOW_TRADE_LOG: str
    SHADOW_ROUNDTRIP_LOG: str
    SHADOW_MODEL_JSON: str

    # TZs
    CT_TZ: ZoneInfo
    UTC_TZ: ZoneInfo

    # Times / cooldowns
    DAILY_RESTART_CT: dt.time
    ORPHAN_SWEEP_COOLDOWN: float
    IB_ERROR_DECAY_SEC: float

    # Shadow window
    SHADOW_START_CT: dt.time
    SHADOW_END_CT: dt.time

    # Holidays
    US_MARKET_HOLIDAYS: Set[dt.date]


def build_config(script_path: str) -> Config:
    base_dir = os.path.abspath(os.path.dirname(script_path))

    log_dir = os.path.join(base_dir, "logs")
    learn_dir = os.path.join(base_dir, "learn")

    hb_path = os.path.join(base_dir, "run", "heartbeat.txt")
    learn_model_path = os.path.join(learn_dir, "bandit_state.json")
    learn_bayes_best = os.path.join(learn_dir, "bayes_best_params.json")

    trade_log_csv = os.path.join(base_dir, "results", "trades.csv")
    bayes_train_csv = os.path.join(base_dir, "results", "trades_bayes_clean.csv")

    day_state_json = os.path.join(base_dir, "data", "state", "day_guard.json")
    daily_restart_json = os.path.join(base_dir, "data", "state", "daily_restart.json")
    runtime_state_json = os.path.join(base_dir, "data", "state", "runtime_state.json")

    shadow_trade_log = os.path.join(base_dir, "results", "shadow_trades.csv")
    shadow_roundtrip_log = os.path.join(base_dir, "results", "shadow_roundtrips.csv")
    shadow_model_json = os.path.join(learn_dir, "shadow_model.json")

    ct_tz = ZoneInfo("America/Chicago")
    utc_tz = ZoneInfo("UTC")

    # NOTE: keep your exact holiday set here (you can extend later)
    holidays = {
        # 2026 CME Globex U.S. holidays (observed dates)
        dt.date(2026, 1, 1),   # New Year's Day
        dt.date(2026, 1, 19),  # Martin Luther King, Jr. Day
        dt.date(2026, 2, 16),  # Presidents Day
        dt.date(2026, 4, 3),   # Good Friday
        dt.date(2026, 5, 25),  # Memorial Day
        dt.date(2026, 6, 19),  # Juneteenth
        dt.date(2026, 7, 3),   # Independence Day (observed)
        dt.date(2026, 9, 7),   # Labor Day
        dt.date(2026, 11, 26), # Thanksgiving Day
        dt.date(2026, 12, 25), # Christmas Day

        # Included in CME’s 2026 schedule window
        dt.date(2027, 1, 1),   # New Year's Day 2027
    }


    return Config(
        BASE_DIR=base_dir,
        LOG_DIR=log_dir,
        LEARN_DIR=learn_dir,
        HB_PATH=hb_path,
        LEARN_MODEL_PATH=learn_model_path,
        LEARN_BAYES_BEST=learn_bayes_best,
        TRADE_LOG_CSV=trade_log_csv,
        BAYES_TRAIN_CSV=bayes_train_csv,
        DAY_STATE_JSON=day_state_json,
        DAILY_RESTART_JSON=daily_restart_json,
        RUNTIME_STATE_JSON=runtime_state_json,
        SHADOW_TRADE_LOG=shadow_trade_log,
        SHADOW_ROUNDTRIP_LOG=shadow_roundtrip_log,
        SHADOW_MODEL_JSON=shadow_model_json,
        CT_TZ=ct_tz,
        UTC_TZ=utc_tz,
        DAILY_RESTART_CT=dt.time(16, 30),
        ORPHAN_SWEEP_COOLDOWN=30.0,
        IB_ERROR_DECAY_SEC=120.0,
        SHADOW_START_CT=dt.time(8, 30),
        SHADOW_END_CT=dt.time(15, 15),
        US_MARKET_HOLIDAYS=holidays,
    )
