#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Model service aggregator (kept for stable imports)."""

from __future__ import annotations

from src.services.baseline_train_service import (
    get_baseline_train_status_api,
    get_latest_baseline_report_raw_api,
    train_baseline_model_api,
)
from src.services.model_info_service import get_model_info_api, health_check_api
from src.services.multimodal_train_service import get_train_status_api, train_model_api

__all__ = [
    "health_check_api",
    "get_model_info_api",
    "train_model_api",
    "get_train_status_api",
    "get_baseline_train_status_api",
    "get_latest_baseline_report_raw_api",
    "train_baseline_model_api",
]
