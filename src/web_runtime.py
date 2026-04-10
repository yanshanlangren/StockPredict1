#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Web runtime context: global state, initialization and common helpers."""

from __future__ import annotations

import logging
import math
import threading
from datetime import datetime
from typing import Any

from flask import jsonify

from src.data_source_manager import DataSource, DataSourceManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MULTIMODAL_MODEL_AVAILABLE = False
MULTIMODAL_TF_AVAILABLE = False
multimodal_predictor = None


def _try_load_multimodal_predictor() -> None:
    """Try to load multimodal predictor and runtime availability."""
    global MULTIMODAL_MODEL_AVAILABLE, MULTIMODAL_TF_AVAILABLE, multimodal_predictor

    try:
        from src.multimodal_model import (
            TENSORFLOW_AVAILABLE as tf_available,
            get_multimodal_predictor,
        )

        MULTIMODAL_TF_AVAILABLE = bool(tf_available)
        multimodal_predictor = get_multimodal_predictor()
        model_info = multimodal_predictor.get_model_info()
        MULTIMODAL_MODEL_AVAILABLE = bool(model_info.get("available", False))

        if MULTIMODAL_MODEL_AVAILABLE:
            logger.info("✓ 多模态模型已加载")
        else:
            logger.info("ℹ️  多模态模型未找到")
            logger.info("提示: 训练多模态模型请运行: python train_multimodal_model.py")
    except Exception as exc:
        MULTIMODAL_MODEL_AVAILABLE = False
        MULTIMODAL_TF_AVAILABLE = False
        multimodal_predictor = None
        logger.warning("多模态模型加载失败: %s", exc)


def refresh_multimodal_predictor() -> dict[str, Any]:
    """Reload predictor instance and return latest model info."""
    global MULTIMODAL_MODEL_AVAILABLE, multimodal_predictor

    from src.multimodal_model import get_multimodal_predictor

    multimodal_predictor = get_multimodal_predictor()
    model_info = multimodal_predictor.get_model_info()
    MULTIMODAL_MODEL_AVAILABLE = bool(model_info.get("available", False))
    return model_info


def get_multimodal_predictor_instance():
    return multimodal_predictor


_try_load_multimodal_predictor()


data_manager = None

training_status = {
    "is_training": False,
    "progress": 0,
    "message": "",
    "start_time": None,
    "end_time": None,
    "error": None,
    "current_epoch": 0,
    "total_epochs": 0,
    "result": None,
    "params": None,
}
training_lock = threading.Lock()

dataset_build_status = {
    "is_building": False,
    "progress": 0,
    "message": "",
    "start_time": None,
    "end_time": None,
    "error": None,
    "result": None,
    "params": None,
    "detail": None,
}
dataset_build_lock = threading.Lock()

baseline_train_status = {
    "is_training": False,
    "progress": 0,
    "message": "",
    "start_time": None,
    "end_time": None,
    "error": None,
    "result": None,
    "params": None,
}
baseline_train_lock = threading.Lock()

market_update_status = {
    "is_running": False,
    "progress": 0,
    "message": "",
    "start_time": None,
    "end_time": None,
    "error": None,
    "result": None,
    "params": None,
    "detail": None,
}
market_update_lock = threading.Lock()

news_sync_status = {
    "is_running": False,
    "progress": 0,
    "message": "",
    "start_time": None,
    "end_time": None,
    "error": None,
    "result": None,
    "params": None,
    "detail": None,
}
news_sync_lock = threading.Lock()

batch_predict_status = {
    "is_running": False,
    "progress": 0,
    "message": "",
    "start_time": None,
    "end_time": None,
    "error": None,
    "result": None,
    "params": None,
    "detail": None,
}
batch_predict_lock = threading.Lock()


def init_components() -> bool:
    """Initialize data source manager."""
    global data_manager

    try:
        logger.info("初始化数据源管理器...")
        data_manager = DataSourceManager(preferred_source=DataSource.TENCENT)
        logger.info("✓ 数据源管理器初始化成功")
        return True
    except Exception as exc:
        logger.error("组件初始化失败: %s", exc)
        return False


def initialize_if_needed() -> None:
    """Lazy initialize components before first request."""
    global data_manager
    if data_manager is None:
        init_components()


def handle_exception(exc):
    """Fallback JSON error handler for unhandled exceptions."""
    logger.error("未处理的异常: %s", exc)
    return (
        jsonify(
            {
                "success": False,
                "message": f"服务器错误: {str(exc)}",
                "error_type": type(exc).__name__,
            }
        ),
        500,
    )


def handle_404(_):
    return jsonify({"success": False, "message": "请求的资源不存在"}), 404


def handle_500(_):
    return jsonify({"success": False, "message": "服务器内部错误"}), 500


def safe_int_param(
    value: Any,
    default: int,
    min_value: int | None = None,
    max_value: int | None = None,
) -> int:
    """Safely parse int parameter and clamp to bounds."""
    parsed = int(default)
    try:
        if value is None or isinstance(value, bool):
            raise ValueError("invalid int value")
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("empty int value")
            numeric = float(text)
        else:
            numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non-finite int value")
        parsed = int(numeric)
    except Exception:
        parsed = int(default)

    if min_value is not None:
        parsed = max(int(min_value), parsed)
    if max_value is not None:
        parsed = min(int(max_value), parsed)
    return parsed


def safe_float_param(
    value: Any,
    default: float,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    """Safely parse float parameter and clamp to bounds."""
    parsed = float(default)
    try:
        if value is None or isinstance(value, bool):
            raise ValueError("invalid float value")
        if isinstance(value, str):
            text = value.strip()
            if not text:
                raise ValueError("empty float value")
            numeric = float(text)
        else:
            numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError("non-finite float value")
        parsed = float(numeric)
    except Exception:
        parsed = float(default)

    if min_value is not None:
        parsed = max(float(min_value), parsed)
    if max_value is not None:
        parsed = min(float(max_value), parsed)
    return parsed


def parse_bool_param(value: Any, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return bool(default)
    return bool(value)


def now_iso() -> str:
    return datetime.now().isoformat()
