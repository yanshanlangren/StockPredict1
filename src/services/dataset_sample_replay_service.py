#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Sample replay helper for stock_code + trade_date."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.schedule_utils import build_trade_close_timestamp, get_previous_trade_date
from src.services.dataset_storage_service import (
    load_dataframe_by_path,
    resolve_processed_dataset_path,
    to_plain_json_value,
)


def build_dataset_sample_replay(
    stock_code: str,
    trade_date: str | None = None,
    news_limit: int = 20,
) -> dict[str, Any]:
    """Replay one stock_code + trade_date sample and return alignment diagnostics."""
    normalized_code = str(stock_code).strip().zfill(6)
    news_limit = max(1, min(int(news_limit), 200))

    model_path = resolve_processed_dataset_path("model_dataset")
    if not model_path:
        raise FileNotFoundError("未找到 model_dataset，请先构建离线数据集")

    model_df = load_dataframe_by_path(model_path)
    if model_df is None or model_df.empty:
        raise ValueError("model_dataset 为空")

    required_model_cols = {"stock_code", "trade_date", "label_up_5d"}
    missing = [col for col in required_model_cols if col not in model_df.columns]
    if missing:
        raise ValueError(f"model_dataset 缺少字段: {', '.join(missing)}")

    model_frame = model_df.copy()
    model_frame["stock_code"] = model_frame["stock_code"].astype(str).str.zfill(6)
    model_frame["trade_date"] = pd.to_datetime(model_frame["trade_date"], errors="coerce").dt.normalize()
    model_frame = model_frame.dropna(subset=["trade_date"])
    model_frame = model_frame[model_frame["stock_code"] == normalized_code].sort_values("trade_date")

    if model_frame.empty:
        return {
            "stock_code": normalized_code,
            "available": False,
            "message": "该股票在 model_dataset 中暂无样本",
            "available_dates": [],
        }

    if trade_date:
        target_date = pd.to_datetime(trade_date, errors="coerce")
        if pd.isna(target_date):
            raise ValueError("trade_date 格式错误，请使用 YYYY-MM-DD")
        target_date = target_date.normalize()
        selected = model_frame[model_frame["trade_date"] == target_date]
        if selected.empty:
            return {
                "stock_code": normalized_code,
                "available": False,
                "message": f"{target_date.date()} 无样本数据",
                "available_dates": model_frame["trade_date"].dt.strftime("%Y-%m-%d").tail(120).tolist(),
            }
        sample_row = selected.iloc[-1]
    else:
        sample_row = model_frame.iloc[-1]
        target_date = sample_row["trade_date"]

    market_snapshot = None
    trade_calendar = []
    market_path = resolve_processed_dataset_path("market_daily")
    if market_path:
        market_df = load_dataframe_by_path(market_path)
        if market_df is not None and not market_df.empty and {"stock_code", "trade_date"}.issubset(market_df.columns):
            market_frame = market_df.copy()
            market_frame["stock_code"] = market_frame["stock_code"].astype(str).str.zfill(6)
            market_frame["trade_date"] = pd.to_datetime(
                market_frame["trade_date"], errors="coerce"
            ).dt.normalize()
            market_frame = market_frame.dropna(subset=["trade_date"])
            market_frame = market_frame[market_frame["stock_code"] == normalized_code].sort_values("trade_date")
            if not market_frame.empty:
                trade_calendar = sorted(market_frame["trade_date"].dropna().unique().tolist())
                market_selected = market_frame[market_frame["trade_date"] == target_date]
                if not market_selected.empty:
                    market_snapshot = {
                        key: to_plain_json_value(val)
                        for key, val in market_selected.iloc[-1].to_dict().items()
                    }

    news_daily_snapshot = None
    news_daily_path = resolve_processed_dataset_path("news_daily_features")
    if news_daily_path:
        news_daily_df = load_dataframe_by_path(news_daily_path)
        if (
            news_daily_df is not None
            and not news_daily_df.empty
            and {"stock_code", "trade_date"}.issubset(news_daily_df.columns)
        ):
            news_daily_frame = news_daily_df.copy()
            news_daily_frame["stock_code"] = news_daily_frame["stock_code"].astype(str).str.zfill(6)
            news_daily_frame["trade_date"] = pd.to_datetime(
                news_daily_frame["trade_date"], errors="coerce"
            ).dt.normalize()
            news_daily_frame = news_daily_frame.dropna(subset=["trade_date"])
            daily_selected = news_daily_frame[
                (news_daily_frame["stock_code"] == normalized_code)
                & (news_daily_frame["trade_date"] == target_date)
            ]
            if not daily_selected.empty:
                news_daily_snapshot = {
                    key: to_plain_json_value(val)
                    for key, val in daily_selected.iloc[-1].to_dict().items()
                }

    previous_trade_date = (
        get_previous_trade_date(trade_calendar, target_date, include_current=False)
        if trade_calendar
        else None
    )
    window_start_ts = (
        build_trade_close_timestamp(previous_trade_date, market_close_hour=15, market_close_minute=0)
        if previous_trade_date is not None
        else None
    )
    window_end_ts = build_trade_close_timestamp(target_date, market_close_hour=15, market_close_minute=0)

    news_used = []
    future_news_count = 0
    raw_news_path = resolve_processed_dataset_path("news_raw")
    if raw_news_path:
        news_raw_df = load_dataframe_by_path(raw_news_path)
        if (
            news_raw_df is not None
            and not news_raw_df.empty
            and {"stock_code", "publish_time"}.issubset(news_raw_df.columns)
        ):
            raw_frame = news_raw_df.copy()
            raw_frame["stock_code"] = raw_frame["stock_code"].astype(str).str.zfill(6)
            raw_frame["publish_time"] = pd.to_datetime(raw_frame["publish_time"], errors="coerce")
            raw_frame = raw_frame.dropna(subset=["publish_time"])
            raw_frame = raw_frame[raw_frame["stock_code"] == normalized_code]

            if window_end_ts is not None:
                future_news_count = int((raw_frame["publish_time"] > window_end_ts).sum())
                used_mask = raw_frame["publish_time"] <= window_end_ts
                if window_start_ts is not None:
                    used_mask &= raw_frame["publish_time"] > window_start_ts
                used_news = raw_frame[used_mask].sort_values("publish_time", ascending=False).head(news_limit)
            else:
                used_news = raw_frame.sort_values("publish_time", ascending=False).head(news_limit)

            for _, row in used_news.iterrows():
                news_used.append(
                    {
                        "publish_time": row["publish_time"].strftime("%Y-%m-%d %H:%M:%S"),
                        "source": str(row.get("source", "") or ""),
                        "title": str(row.get("title", "") or ""),
                        "sentiment": to_plain_json_value(row.get("sentiment")),
                        "importance": to_plain_json_value(row.get("importance")),
                    }
                )

    used_latest_publish_time = news_used[0]["publish_time"] if news_used else None
    time_check_passed = True
    if news_used and window_end_ts is not None:
        try:
            max_used_publish = max(pd.to_datetime(item["publish_time"]) for item in news_used)
            time_check_passed = bool(max_used_publish <= window_end_ts)
        except Exception:
            time_check_passed = False

    excluded_fields = {"stock_code", "stock_name", "trade_date", "label_up_5d", "future_ret_5d", "date"}
    feature_snapshot = {}
    for key, value in sample_row.to_dict().items():
        if key in excluded_fields:
            continue
        feature_snapshot[key] = to_plain_json_value(value)

    return {
        "stock_code": normalized_code,
        "trade_date": target_date.strftime("%Y-%m-%d"),
        "available": True,
        "dataset_path": model_path,
        "label_up_5d": int(to_plain_json_value(sample_row.get("label_up_5d")) or 0),
        "future_ret_5d": to_plain_json_value(sample_row.get("future_ret_5d")),
        "feature_count": len(feature_snapshot),
        "feature_snapshot": feature_snapshot,
        "market_snapshot": market_snapshot,
        "news_daily_snapshot": news_daily_snapshot,
        "news_window": {
            "start": window_start_ts.strftime("%Y-%m-%d %H:%M:%S") if window_start_ts is not None else None,
            "end": window_end_ts.strftime("%Y-%m-%d %H:%M:%S") if window_end_ts is not None else None,
            "used_news_count": len(news_used),
            "used_latest_publish_time": used_latest_publish_time,
            "future_news_count_after_close": future_news_count,
            "strict_time_check_passed": bool(time_check_passed),
        },
        "used_news": news_used,
        "available_dates": model_frame["trade_date"].dt.strftime("%Y-%m-%d").tail(120).tolist(),
    }
