#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset storage and report loading helpers."""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_DIR, RESULT_DIR
from src.web_runtime import logger


def load_dataset_metadata() -> dict[str, Any] | None:
    """Load latest dataset metadata."""
    metadata_path = os.path.join(PROCESSED_DATA_DIR, "dataset_metadata.json")
    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except Exception as exc:
        logger.error("读取数据集元数据失败: %s", exc)
        return None


def load_dataset_preview(dataset_path: str | None, limit: int = 5) -> dict[str, Any]:
    """Load sampled rows for dataset preview."""
    if not dataset_path or not os.path.exists(dataset_path):
        return {"columns": [], "rows": []}

    try:
        if dataset_path.endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path, nrows=limit)

        preferred_columns = [
            "stock_code",
            "stock_name",
            "trade_date",
            "close",
            "ret_5d",
            "rsi",
            "news_count",
            "weighted_sentiment",
            "news_impact_total",
            "label_up_5d",
            "future_ret_5d",
        ]
        preview_columns = [col for col in preferred_columns if col in df.columns]
        if not preview_columns:
            preview_columns = list(df.columns[:10])

        preview_df = df[preview_columns].head(limit).copy()
        for column in preview_df.columns:
            if pd.api.types.is_datetime64_any_dtype(preview_df[column]):
                preview_df[column] = preview_df[column].astype(str)

        preview_df = preview_df.replace({np.nan: None})
        return {
            "columns": preview_columns,
            "rows": json.loads(preview_df.to_json(orient="records", force_ascii=False)),
        }
    except Exception as exc:
        logger.error("读取数据集预览失败: %s", exc)
        return {"columns": [], "rows": []}


def resolve_processed_dataset_path(dataset_name: str) -> str | None:
    """Resolve dataset path from metadata then processed directory fallback."""
    metadata = load_dataset_metadata()
    if metadata:
        path = metadata.get("paths", {}).get(dataset_name)
        if path and os.path.exists(path):
            return path

    parquet_path = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}.parquet")
    csv_path = os.path.join(PROCESSED_DATA_DIR, f"{dataset_name}.csv")
    if os.path.exists(parquet_path):
        return parquet_path
    if os.path.exists(csv_path):
        return csv_path
    return None


def load_dataframe_by_path(path: str | None) -> pd.DataFrame | None:
    """Load dataframe based on suffix."""
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_latest_json_report(filename: str) -> dict[str, Any] | None:
    """Load report json under results directory."""
    report_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, "r", encoding="utf-8") as file_obj:
            return json.load(file_obj)
    except Exception as exc:
        logger.error("读取报告失败 %s: %s", filename, exc)
        return None


def load_latest_baseline_report() -> dict[str, Any] | None:
    return load_latest_json_report("structured_baseline_report_latest.json")


def to_plain_json_value(value: Any):
    """Convert pandas/numpy scalars into JSON-safe values."""
    if value is None:
        return None
    if isinstance(value, (list, dict, tuple)):
        return value
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, np.generic):
        return value.item()
    return value
