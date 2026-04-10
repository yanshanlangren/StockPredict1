#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core API route registration (routing only, no business logic)."""

from __future__ import annotations

from flask import request

from src.services.company_service import get_company_info_api, get_relevance_graph_api
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


def register_core_api_routes(app):
    """Register core API routes to service handlers."""

    app.add_url_rule("/api/health", view_func=health_check_api, methods=["GET"])
    app.add_url_rule("/api/model/info", view_func=get_model_info_api, methods=["GET"])
    app.add_url_rule(
        "/api/model/train",
        endpoint="api_model_train",
        view_func=lambda: train_model_api(request.json or {}),
        methods=["POST"],
    )
    app.add_url_rule("/api/model/train/status", view_func=get_train_status_api, methods=["GET"])

    app.add_url_rule(
        "/api/model/train-baseline/status",
        view_func=get_baseline_train_status_api,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/model/train-baseline/report/latest",
        view_func=get_latest_baseline_report_raw_api,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/model/train-baseline",
        endpoint="api_model_train_baseline",
        view_func=lambda: train_baseline_model_api(request.json or {}),
        methods=["POST"],
    )

    app.add_url_rule(
        "/api/model/evaluate/report/latest",
        view_func=get_latest_evaluation_report_raw_api,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/model/evaluate",
        endpoint="api_model_evaluate",
        view_func=lambda: run_offline_evaluation_api(request.json or {}),
        methods=["POST"],
    )

    app.add_url_rule(
        "/api/backtest/cross-section/report/latest",
        view_func=get_latest_backtest_report_raw_api,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/backtest/cross-section",
        endpoint="api_backtest_cross_section",
        view_func=lambda: run_cross_section_backtest_api(request.json or {}),
        methods=["POST"],
    )

    app.add_url_rule("/api/dataset/info", view_func=get_dataset_info_api, methods=["GET"])
    app.add_url_rule(
        "/api/dataset/sample/<stock_code>",
        endpoint="api_dataset_sample",
        view_func=lambda stock_code: get_dataset_sample_replay_api(
            stock_code=stock_code,
            trade_date=request.args.get("trade_date"),
            news_limit=request.args.get("news_limit", 20, type=int),
        ),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/dataset/build/status",
        view_func=get_dataset_build_status_api,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/dataset/build",
        endpoint="api_dataset_build",
        view_func=lambda: build_dataset_api(request.json or {}),
        methods=["POST"],
    )

    app.add_url_rule(
        "/api/stocks",
        endpoint="api_stocks_list",
        view_func=lambda: get_stocks_api(
            force_refresh=request.args.get("refresh", "false").lower() == "true"
        ),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/system/market-update/status",
        view_func=get_market_update_status_api,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/system/market-update",
        endpoint="api_system_market_update",
        view_func=lambda: start_market_update_api(request.json or {}),
        methods=["POST"],
    )

    app.add_url_rule(
        "/api/company/<stock_code>",
        view_func=get_company_info_api,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/analysis/relevance-graph/<stock_code>",
        view_func=get_relevance_graph_api,
        methods=["GET"],
    )
