#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model information and health APIs."""

from __future__ import annotations

from datetime import datetime

from flask import jsonify

import src.web_runtime as runtime


def health_check_api():
    """System health check."""
    model_info = {}
    if runtime.MULTIMODAL_MODEL_AVAILABLE and runtime.get_multimodal_predictor_instance() is not None:
        try:
            model_info = runtime.get_multimodal_predictor_instance().get_model_info()
        except Exception:
            model_info = {}

    return jsonify(
        {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "data_manager": runtime.data_manager is not None,
                "multimodal_model": runtime.MULTIMODAL_MODEL_AVAILABLE,
            },
            "model": model_info if model_info else None,
        }
    )


def get_model_info_api():
    """Get model information."""
    result = {"model_available": runtime.MULTIMODAL_MODEL_AVAILABLE}

    predictor = runtime.get_multimodal_predictor_instance()
    if predictor is None:
        result["model_error"] = "多模态模型组件不可用"
        return jsonify(result)

    try:
        model_info = predictor.get_model_info()
        result["model"] = model_info
        result["model_available"] = bool(model_info.get("available", False))
    except Exception as exc:
        result["model_error"] = str(exc)

    return jsonify(result)
