#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Structured baseline training APIs."""

from __future__ import annotations

import json
import threading
from datetime import datetime

from flask import current_app, jsonify

import src.web_runtime as runtime
from src.services.dataset_helpers import load_latest_baseline_report


def get_baseline_train_status_api():
    """Get baseline training status."""
    with runtime.baseline_train_lock:
        status = dict(runtime.baseline_train_status)
    return jsonify({"success": True, "status": status})


def get_latest_baseline_report_raw_api():
    """Return latest baseline report as formatted JSON."""
    report = load_latest_baseline_report()
    if not report:
        return jsonify({"success": False, "message": "暂无可用的基线评估报告"}), 404

    return current_app.response_class(
        response=json.dumps(report, ensure_ascii=False, indent=2),
        status=200,
        mimetype="application/json",
    )


def _parse_baseline_train_params(params: dict):
    model_type = str(params.get("model_type", "logistic")).strip().lower()
    top_k = int(params.get("top_k", 20))
    valid_ratio = float(params.get("valid_ratio", 0.15))
    test_ratio = float(params.get("test_ratio", 0.15))
    dataset_path = params.get("dataset_path")
    if isinstance(dataset_path, str):
        dataset_path = dataset_path.strip() or None
    return model_type, top_k, valid_ratio, test_ratio, dataset_path


def train_baseline_model_api(params: dict):
    """Start structured baseline model training."""
    with runtime.baseline_train_lock:
        if runtime.baseline_train_status.get("is_training"):
            return jsonify({"success": False, "message": "基线模型正在训练中，请稍候..."}), 400

        try:
            model_type, top_k, valid_ratio, test_ratio, dataset_path = _parse_baseline_train_params(params)
        except (TypeError, ValueError):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "参数格式错误，请检查 model_type/top_k/valid_ratio/test_ratio",
                    }
                ),
                400,
            )

        if model_type not in ["logistic", "random_forest", "lightgbm"]:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "model_type 只支持 logistic / random_forest / lightgbm",
                    }
                ),
                400,
            )

        top_k = max(1, min(top_k, 200))
        valid_ratio = max(0.05, min(valid_ratio, 0.4))
        test_ratio = max(0.05, min(test_ratio, 0.4))

        if valid_ratio + test_ratio >= 0.8:
            return jsonify({"success": False, "message": "valid_ratio + test_ratio 必须小于 0.8"}), 400

        runtime.baseline_train_status.clear()
        runtime.baseline_train_status.update(
            {
                "is_training": True,
                "progress": 0,
                "message": "初始化基线训练任务...",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "error": None,
                "result": None,
                "params": {
                    "model_type": model_type,
                    "top_k": top_k,
                    "valid_ratio": valid_ratio,
                    "test_ratio": test_ratio,
                    "dataset_path": dataset_path,
                },
            }
        )

    def run_baseline_training():
        try:
            from src.baseline_model import BaselineModelTrainer

            with runtime.baseline_train_lock:
                runtime.baseline_train_status["progress"] = 15
                runtime.baseline_train_status["message"] = "正在加载离线数据集..."

            trainer = BaselineModelTrainer(
                dataset_path=dataset_path,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio,
            )

            with runtime.baseline_train_lock:
                runtime.baseline_train_status["progress"] = 45
                runtime.baseline_train_status["message"] = "正在训练结构化基线模型..."

            report = trainer.run(model_type=model_type, top_k=top_k)

            with runtime.baseline_train_lock:
                runtime.baseline_train_status["is_training"] = False
                runtime.baseline_train_status["progress"] = 100
                runtime.baseline_train_status["message"] = "基线模型训练完成"
                runtime.baseline_train_status["end_time"] = datetime.now().isoformat()
                runtime.baseline_train_status["result"] = {
                    "model_type": model_type,
                    "report_path": report.get("report_path"),
                    "latest_path": report.get("latest_path"),
                    "metrics": report.get("metrics", {}),
                    "model_files": report.get("model_files", {}),
                }
        except Exception as exc:
            runtime.logger.error("基线训练失败: %s", exc)
            with runtime.baseline_train_lock:
                runtime.baseline_train_status["is_training"] = False
                runtime.baseline_train_status["error"] = str(exc)
                runtime.baseline_train_status["message"] = f"基线训练失败: {str(exc)}"
                runtime.baseline_train_status["end_time"] = datetime.now().isoformat()

    thread = threading.Thread(target=run_baseline_training)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "success": True,
            "message": "基线训练已启动，请通过 /api/model/train-baseline/status 查看进度",
            "params": {
                "model_type": model_type,
                "top_k": top_k,
                "valid_ratio": valid_ratio,
                "test_ratio": test_ratio,
                "dataset_path": dataset_path,
            },
        }
    )
