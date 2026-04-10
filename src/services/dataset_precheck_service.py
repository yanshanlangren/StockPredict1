#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset precheck helpers for multimodal training."""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from src.services.dataset_storage_service import (
    load_dataset_metadata,
    resolve_processed_dataset_path,
)
from src.web_runtime import MULTIMODAL_TF_AVAILABLE, logger


MULTIMODAL_TECH_REQUIRED_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "rsi",
    "macd",
    "bb_position",
    "ma5_gt_ma10",
    "ma10_gt_ma20",
    "volume_ratio",
]

MULTIMODAL_NEWS_SENTIMENT_COLUMNS = [
    "news_count",
    "avg_sentiment",
    "weighted_sentiment",
    "news_impact_total",
]


def load_dataset_columns(dataset_path: str | None) -> list[str]:
    """Load dataset column names only."""
    if not dataset_path or not os.path.exists(dataset_path):
        return []

    try:
        if dataset_path.endswith(".parquet"):
            frame = pd.read_parquet(dataset_path)
        else:
            frame = pd.read_csv(dataset_path, nrows=1)
        return list(frame.columns)
    except Exception as exc:
        logger.warning("读取数据集字段失败: %s, error=%s", dataset_path, exc)
        return []


def build_multimodal_feature_status(model_dataset_path: str | None) -> dict[str, Any]:
    """Build feature-group readiness for multimodal training."""
    columns = load_dataset_columns(model_dataset_path)
    column_set = set(columns)

    tech_missing = [col for col in MULTIMODAL_TECH_REQUIRED_COLUMNS if col not in column_set]
    news_missing = [col for col in MULTIMODAL_NEWS_SENTIMENT_COLUMNS if col not in column_set]

    has_sector = any(col.startswith("sector_") for col in columns)
    has_relevance = any(
        col.startswith("relevance_")
        or col.startswith("static_")
        or col.startswith("stock_sector_")
        for col in columns
    )

    feature_groups = [
        {
            "name": "news_sentiment",
            "label": "新闻情感",
            "ready": len(news_missing) == 0,
            "missing": news_missing,
            "hint": "news_count / avg_sentiment / weighted_sentiment / news_impact_total",
        },
        {
            "name": "sector_impact",
            "label": "领域影响",
            "ready": bool(has_sector),
            "missing": [] if has_sector else ["sector_*"],
            "hint": "需要 sector_* 字段",
        },
        {
            "name": "technical",
            "label": "技术指标",
            "ready": len(tech_missing) == 0,
            "missing": tech_missing,
            "hint": "需要 ret/rsi/macd/bb/ma/volume 等技术面字段",
        },
        {
            "name": "relevance",
            "label": "相关性矩阵",
            "ready": bool(has_relevance),
            "missing": [] if has_relevance else ["relevance_* / static_* / stock_sector_*"],
            "hint": "需要相关性或静态关联字段",
        },
    ]

    return {
        "ready": all(item["ready"] for item in feature_groups),
        "feature_groups": feature_groups,
        "column_count": len(columns),
    }


def build_multimodal_precheck(metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build multimodal training precheck for required datasets and feature groups."""
    metadata = metadata or load_dataset_metadata() or {}
    row_counts = metadata.get("row_counts", {}) if isinstance(metadata, dict) else {}
    paths = metadata.get("paths", {}) if isinstance(metadata, dict) else {}

    required_inputs = ["model_dataset", "market_daily", "news_raw"]
    inputs = []
    resolved_path_map = {}

    for name in required_inputs:
        raw_path = paths.get(name)
        path = raw_path if raw_path and os.path.exists(raw_path) else resolve_processed_dataset_path(name)
        exists = bool(path and os.path.exists(path))
        resolved_path_map[name] = path
        inputs.append(
            {
                "name": name,
                "label": name,
                "ready": exists,
                "path": path,
                "row_count": int(row_counts.get(name, 0)) if row_counts.get(name) is not None else 0,
            }
        )

    feature_status = build_multimodal_feature_status(resolved_path_map.get("model_dataset"))

    return {
        "ready": all(item["ready"] for item in inputs),
        "inputs": inputs,
        "feature_status": feature_status,
        "tensorflow_available": bool(MULTIMODAL_TF_AVAILABLE),
    }
