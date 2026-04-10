#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""News and prediction route registration (routing only)."""

from __future__ import annotations

from flask import request

from src.services.news_feature_service import (
    analyze_news_impact_api,
    get_news_api,
    get_news_daily_features_api,
)
from src.services.news_sync_service import (
    get_news_sync_status_api,
    start_news_sync_api,
)
from src.services.prediction_service import (
    get_batch_predict_status_api,
    get_stock_info_api,
    predict_batch_api,
    predict_multimodal_api,
)


def register_news_predict_routes(app):
    """Register routes for news and prediction related APIs."""

    app.add_url_rule(
        "/api/system/news-sync/status",
        view_func=get_news_sync_status_api,
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/system/news-sync",
        endpoint="api_system_news_sync",
        view_func=lambda: start_news_sync_api(request.json or {}),
        methods=["POST"],
    )

    app.add_url_rule(
        "/api/stock/<stock_code>",
        endpoint="api_stock_info",
        view_func=lambda stock_code: get_stock_info_api(
            stock_code=stock_code,
            days=request.args.get("days", 100, type=int),
        ),
        methods=["GET"],
    )

    app.add_url_rule(
        "/api/predict/multimodal/<stock_code>",
        endpoint="api_predict_multimodal",
        view_func=lambda stock_code: predict_multimodal_api(stock_code, request.json or {}),
        methods=["POST"],
    )

    app.add_url_rule(
        "/api/predict/batch",
        endpoint="api_predict_batch",
        view_func=lambda: predict_batch_api(request.json or {}),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/predict/batch/status",
        endpoint="api_predict_batch_status",
        view_func=get_batch_predict_status_api,
        methods=["GET"],
    )

    app.add_url_rule(
        "/api/news",
        endpoint="api_news_list",
        view_func=lambda: get_news_api(
            stock_code=request.args.get("stock_code"),
            limit=request.args.get("limit", 50, type=int),
            source=request.args.get("source", "eastmoney"),
            max_age_hours_raw=request.args.get("max_age_hours", default="72"),
        ),
        methods=["GET"],
    )

    app.add_url_rule(
        "/api/features/news/<stock_code>",
        endpoint="api_news_daily_features",
        view_func=lambda stock_code: get_news_daily_features_api(
            stock_code=stock_code,
            trade_date=request.args.get("trade_date"),
        ),
        methods=["GET"],
    )

    app.add_url_rule(
        "/api/analysis/news-impact/<stock_code>",
        view_func=analyze_news_impact_api,
        methods=["GET"],
    )
