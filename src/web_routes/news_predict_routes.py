#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""News and prediction route registration (routing only)."""

from __future__ import annotations

from flask import request

from src.news_source_registry import get_news_source_registry
from src.services.news_feature_service import (
    analyze_news_impact_api,
    get_news_api,
    get_news_daily_features_api,
)
from src.services.news_source_analyzer_service import analyze_news_source_api
from src.services.news_source_service import (
    create_news_source_api,
    delete_news_source_api,
    list_news_sources_api,
    update_news_source_api,
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
            source=request.args.get("source", get_news_source_registry().get_default_source_id()),
            max_age_hours_raw=request.args.get("max_age_hours", default="72"),
        ),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/news/sources",
        endpoint="api_news_sources_list",
        view_func=lambda: list_news_sources_api(
            include_disabled_raw=request.args.get("include_disabled", default="1")
        ),
        methods=["GET"],
    )
    app.add_url_rule(
        "/api/news/sources",
        endpoint="api_news_sources_create",
        view_func=lambda: create_news_source_api(request.json or {}),
        methods=["POST"],
    )
    app.add_url_rule(
        "/api/news/sources/<source_id>",
        endpoint="api_news_sources_update",
        view_func=lambda source_id: update_news_source_api(source_id, request.json or {}),
        methods=["PUT"],
    )
    app.add_url_rule(
        "/api/news/sources/<source_id>",
        endpoint="api_news_sources_delete",
        view_func=lambda source_id: delete_news_source_api(source_id),
        methods=["DELETE"],
    )
    app.add_url_rule(
        "/api/news/sources/analyze",
        endpoint="api_news_sources_analyze",
        view_func=lambda: analyze_news_source_api(request.json or {}),
        methods=["POST"],
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
