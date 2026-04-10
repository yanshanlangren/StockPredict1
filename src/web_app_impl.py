#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Backward-compatible facade for legacy imports.

Deprecated: use `src.web_runtime`, `src.services.*`, and `src.web_routes.*` directly.
"""

from __future__ import annotations

from flask import request

import src.web_runtime as runtime
from src.services.company_service import get_company_info_api, get_relevance_graph_api
from src.services.dataset_helpers import (
    load_dataframe_by_path as _load_dataframe_by_path,
    resolve_processed_dataset_path as _resolve_processed_dataset_path,
)
from src.services.dataset_service import (
    build_dataset_api,
    get_dataset_build_status_api,
    get_dataset_info_api,
    get_dataset_sample_replay_api,
)
from src.services.evaluation_service import (
    get_latest_backtest_report_raw_api,
    get_latest_evaluation_report_raw_api,
    run_cross_section_backtest_api,
    run_offline_evaluation_api,
)
from src.services.model_service import (
    get_baseline_train_status_api,
    get_latest_baseline_report_raw_api,
    get_model_info_api,
    get_train_status_api,
    health_check_api,
    train_baseline_model_api,
    train_model_api,
)
from src.services.system_service import (
    get_market_update_status_api,
    get_stocks_api,
    start_market_update_api,
)

logger = runtime.logger


def init_components():
    return runtime.init_components()


def initialize_if_needed():
    return runtime.initialize_if_needed()


def handle_exception(exc):
    return runtime.handle_exception(exc)


def handle_404(exc):
    return runtime.handle_404(exc)


def handle_500(exc):
    return runtime.handle_500(exc)


def _safe_int_param(value, default: int, min_value: int = None, max_value: int = None):
    return runtime.safe_int_param(value, default, min_value=min_value, max_value=max_value)


def _safe_float_param(value, default: float, min_value: float = None, max_value: float = None):
    return runtime.safe_float_param(value, default, min_value=min_value, max_value=max_value)


def _parse_bool_param(value, default: bool = True):
    return runtime.parse_bool_param(value, default)


def health_check():
    return health_check_api()


def get_model_info():
    return get_model_info_api()


def train_model():
    return train_model_api(request.json or {})


def get_train_status():
    return get_train_status_api()


def get_baseline_train_status():
    return get_baseline_train_status_api()


def get_latest_baseline_report_raw():
    return get_latest_baseline_report_raw_api()


def train_baseline_model():
    return train_baseline_model_api(request.json or {})


def get_latest_evaluation_report_raw():
    return get_latest_evaluation_report_raw_api()


def run_offline_evaluation():
    return run_offline_evaluation_api(request.json or {})


def get_latest_backtest_report_raw():
    return get_latest_backtest_report_raw_api()


def run_cross_section_backtest():
    return run_cross_section_backtest_api(request.json or {})


def get_dataset_info():
    return get_dataset_info_api()


def get_dataset_sample_replay(stock_code):
    return get_dataset_sample_replay_api(
        stock_code=stock_code,
        trade_date=request.args.get("trade_date"),
        news_limit=request.args.get("news_limit", 20, type=int),
    )


def get_dataset_build_status():
    return get_dataset_build_status_api()


def build_dataset():
    return build_dataset_api(request.json or {})


def get_stocks():
    return get_stocks_api(force_refresh=request.args.get("refresh", "false").lower() == "true")


def get_market_update_status():
    return get_market_update_status_api()


def start_market_update():
    return start_market_update_api(request.json or {})


def get_company_info(stock_code):
    return get_company_info_api(stock_code)


def get_relevance_graph(stock_code):
    return get_relevance_graph_api(stock_code)

def get_data_manager():
    return runtime.data_manager


data_manager = runtime.data_manager


training_status = runtime.training_status
training_lock = runtime.training_lock
baseline_train_status = runtime.baseline_train_status
baseline_train_lock = runtime.baseline_train_lock
dataset_build_status = runtime.dataset_build_status
dataset_build_lock = runtime.dataset_build_lock
market_update_status = runtime.market_update_status
market_update_lock = runtime.market_update_lock
news_sync_status = runtime.news_sync_status
news_sync_lock = runtime.news_sync_lock
batch_predict_status = runtime.batch_predict_status
batch_predict_lock = runtime.batch_predict_lock
