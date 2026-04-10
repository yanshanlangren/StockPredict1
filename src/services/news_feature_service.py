#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""News list, daily feature and impact analysis APIs."""

from __future__ import annotations

from flask import jsonify

import src.web_runtime as runtime
from src.services.dataset_helpers import load_dataframe_by_path, resolve_processed_dataset_path
from src.services.news_common import parse_optional_age_hours, to_plain_value


def get_news_api(stock_code: str | None, limit: int, source: str, max_age_hours_raw):
    """Fetch news list and statistics."""
    try:
        from src.news_crawler import get_news_crawler

        max_news_age_hours = parse_optional_age_hours(max_age_hours_raw)
        crawler = get_news_crawler()
        if not crawler.is_available():
            return jsonify({"success": False, "message": "新闻爬虫不可用，请安装akshare"}), 400

        news_list = crawler.get_news(
            stock_code=stock_code,
            limit=limit,
            source=source,
            max_news_age_hours=max_news_age_hours,
        )
        stats = crawler.get_news_statistics(news_list)

        return jsonify(
            {
                "success": True,
                "data": {
                    "news": news_list,
                    "statistics": stats,
                    "max_news_age_hours": max_news_age_hours,
                },
            }
        )
    except Exception as exc:
        runtime.logger.error("获取新闻失败: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500


def get_news_daily_features_api(stock_code: str, trade_date: str | None):
    """Query news_daily_features by stock/date."""
    try:
        import pandas as pd

        normalized_code = str(stock_code).strip().zfill(6)

        news_feature_path = resolve_processed_dataset_path("news_daily_features")
        if not news_feature_path:
            return (
                jsonify({"success": False, "message": "未找到 news_daily_features 数据集，请先构建数据集"}),
                404,
            )

        df = load_dataframe_by_path(news_feature_path)
        if df is None or df.empty:
            return jsonify({"success": False, "message": "news_daily_features 数据为空"}), 404

        if "stock_code" not in df.columns or "trade_date" not in df.columns:
            return (
                jsonify({"success": False, "message": "news_daily_features 缺少必要字段(stock_code/trade_date)"}),
                500,
            )

        frame = df.copy()
        frame["stock_code"] = frame["stock_code"].astype(str).str.zfill(6)
        frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce")
        frame = frame.dropna(subset=["trade_date"])
        frame = frame[frame["stock_code"] == normalized_code].sort_values("trade_date")

        if frame.empty:
            return jsonify(
                {
                    "success": True,
                    "data": {
                        "stock_code": normalized_code,
                        "available": False,
                        "message": "该股票在 news_daily_features 中暂无数据",
                        "available_dates": [],
                    },
                }
            )

        if trade_date:
            target_date = pd.to_datetime(trade_date, errors="coerce")
            if pd.isna(target_date):
                return jsonify({"success": False, "message": "trade_date 格式错误，请使用 YYYY-MM-DD"}), 400
            target_date = target_date.normalize()
            selected = frame[frame["trade_date"].dt.normalize() == target_date]
            if selected.empty:
                return jsonify(
                    {
                        "success": True,
                        "data": {
                            "stock_code": normalized_code,
                            "available": False,
                            "message": f"{target_date.date()} 无新闻特征数据",
                            "available_dates": frame["trade_date"].dt.strftime("%Y-%m-%d").tail(60).tolist(),
                        },
                    }
                )
            row = selected.iloc[-1]
        else:
            row = frame.iloc[-1]

        record = {key: to_plain_value(value) for key, value in row.to_dict().items()}
        return jsonify(
            {
                "success": True,
                "data": {
                    "stock_code": normalized_code,
                    "available": True,
                    "dataset_path": news_feature_path,
                    "trade_date": row["trade_date"].strftime("%Y-%m-%d"),
                    "feature_count": len(record),
                    "features": record,
                    "available_dates": frame["trade_date"].dt.strftime("%Y-%m-%d").tail(60).tolist(),
                },
            }
        )
    except Exception as exc:
        runtime.logger.error("查询新闻特征失败: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500


def analyze_news_impact_api(stock_code: str):
    """Analyze news impact report for stock."""
    try:
        from src.news_crawler import get_news_crawler
        from src.news_impact_analyzer import get_news_impact_analyzer

        crawler = get_news_crawler()
        analyzer = get_news_impact_analyzer()

        news_list = crawler.get_news(stock_code=stock_code, limit=120, source="all")
        if not news_list:
            return jsonify(
                {
                    "success": True,
                    "data": {"stock_code": stock_code, "total_news": 0, "message": "暂无相关新闻"},
                }
            )

        report = analyzer.generate_impact_report(news_list, stock_code)
        return jsonify({"success": True, "data": report})
    except Exception as exc:
        runtime.logger.error("分析新闻影响失败: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500
