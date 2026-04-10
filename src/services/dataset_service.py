#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset APIs: info, sample replay and dataset build task."""

from __future__ import annotations

import threading
from datetime import datetime

from flask import jsonify

import src.web_runtime as runtime
from src.services.dataset_helpers import (
    build_dataset_info,
    build_dataset_sample_replay,
)


def get_dataset_info_api():
    """Get current offline dataset information."""
    return jsonify({"success": True, "data": build_dataset_info()})


def get_dataset_sample_replay_api(stock_code: str, trade_date: str | None, news_limit: int):
    """Replay dataset sample by stock_code + trade_date."""
    try:
        replay = build_dataset_sample_replay(
            stock_code=stock_code,
            trade_date=trade_date,
            news_limit=news_limit,
        )
        return jsonify({"success": True, "data": replay})
    except FileNotFoundError as exc:
        return jsonify({"success": False, "message": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        runtime.logger.error("样本回放失败: %s", exc)
        return jsonify({"success": False, "message": f"样本回放失败: {str(exc)}"}), 500


def get_dataset_build_status_api():
    """Get offline dataset build status."""
    with runtime.dataset_build_lock:
        status = dict(runtime.dataset_build_status)

    return jsonify({"success": True, "status": status})


def _parse_build_dataset_params(params: dict):
    stocks = max(1, min(int(params.get("stocks", 50)), 5000))
    days = max(80, min(int(params.get("days", 240)), 1500))
    horizon = max(1, min(int(params.get("horizon", 5)), 20))
    label_threshold = float(params.get("label_threshold", 0.01))
    force_refresh = bool(params.get("force_refresh", False))
    refresh_news = bool(params.get("refresh_news", False))
    return stocks, days, horizon, label_threshold, force_refresh, refresh_news


def build_dataset_api(params: dict):
    """Start offline dataset build task."""
    with runtime.dataset_build_lock:
        if runtime.dataset_build_status.get("is_building"):
            return jsonify({"success": False, "message": "数据集正在构建中，请稍候..."}), 400

        stocks, days, horizon, label_threshold, force_refresh, refresh_news = _parse_build_dataset_params(params)

        runtime.dataset_build_status.clear()
        runtime.dataset_build_status.update(
            {
                "is_building": True,
                "progress": 0,
                "message": "初始化数据集构建任务...",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "error": None,
                "result": None,
                "detail": None,
                "params": {
                    "stocks": stocks,
                    "days": days,
                    "horizon": horizon,
                    "label_threshold": label_threshold,
                    "force_refresh": force_refresh,
                    "refresh_news": refresh_news,
                },
            }
        )

    def update_dataset_progress(progress, message, extra=None):
        with runtime.dataset_build_lock:
            runtime.dataset_build_status["progress"] = progress
            runtime.dataset_build_status["message"] = message
            runtime.dataset_build_status["detail"] = extra or None

    def run_dataset_build():
        try:
            from src.dataset_builder import DatasetBuilder

            builder = DatasetBuilder()
            result = builder.build(
                stock_limit=stocks,
                days=days,
                future_horizon=horizon,
                label_threshold=label_threshold,
                force_refresh=force_refresh,
                refresh_news=refresh_news,
                progress_callback=update_dataset_progress,
            )

            with runtime.dataset_build_lock:
                runtime.dataset_build_status["is_building"] = False
                runtime.dataset_build_status["progress"] = 100
                runtime.dataset_build_status["message"] = "离线数据集已更新"
                runtime.dataset_build_status["end_time"] = datetime.now().isoformat()
                runtime.dataset_build_status["result"] = result
                runtime.dataset_build_status["detail"] = {
                    "processed_stocks": len(result.get("processed_stocks", [])),
                    "row_counts": result.get("row_counts", {}),
                }
        except Exception as exc:
            runtime.logger.error("数据集构建失败: %s", exc)
            with runtime.dataset_build_lock:
                runtime.dataset_build_status["is_building"] = False
                runtime.dataset_build_status["error"] = str(exc)
                runtime.dataset_build_status["message"] = f"数据集构建失败: {str(exc)}"
                runtime.dataset_build_status["end_time"] = datetime.now().isoformat()

    thread = threading.Thread(target=run_dataset_build)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "success": True,
            "message": "数据集构建已启动，请通过 /api/dataset/build/status 查看进度",
            "params": {
                "stocks": stocks,
                "days": days,
                "horizon": horizon,
                "label_threshold": label_threshold,
                "force_refresh": force_refresh,
                "refresh_news": refresh_news,
            },
        }
    )
