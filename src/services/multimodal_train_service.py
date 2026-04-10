#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Multimodal training APIs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
from datetime import datetime

from flask import jsonify

import src.web_runtime as runtime
from config import BASE_DIR
from src.services.dataset_helpers import (
    build_multimodal_precheck,
    load_dataset_metadata,
    resolve_processed_dataset_path,
)


def _parse_train_model_params(params: dict):
    stocks = runtime.safe_int_param(params.get("stocks"), default=50, min_value=10, max_value=5000)
    days = runtime.safe_int_param(params.get("days"), default=200, min_value=60, max_value=1500)
    epochs = runtime.safe_int_param(params.get("epochs"), default=50, min_value=5, max_value=500)
    dataset_path = params.get("dataset_path")
    market_path = params.get("market_path")
    news_raw_path = params.get("news_raw_path")

    dataset_path = dataset_path.strip() or None if isinstance(dataset_path, str) else None
    market_path = market_path.strip() or None if isinstance(market_path, str) else None
    news_raw_path = news_raw_path.strip() or None if isinstance(news_raw_path, str) else None

    return stocks, days, epochs, dataset_path, market_path, news_raw_path


def train_model_api(params: dict):
    """Start multimodal model training task."""
    with runtime.training_lock:
        if runtime.training_status.get("is_training"):
            return jsonify({"success": False, "message": "模型正在训练中，请稍候..."}), 400

        stocks, days, epochs, dataset_path, market_path, news_raw_path = _parse_train_model_params(params)

        if not runtime.MULTIMODAL_TF_AVAILABLE:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "当前环境未启用 TensorFlow，多模态训练只允许 TensorFlow 模式",
                    }
                ),
                400,
            )

        resolved_input_paths = {
            "model_dataset": dataset_path or resolve_processed_dataset_path("model_dataset"),
            "market_daily": market_path or resolve_processed_dataset_path("market_daily"),
            "news_raw": news_raw_path or resolve_processed_dataset_path("news_raw"),
        }
        missing_inputs = [
            name
            for name, path in resolved_input_paths.items()
            if not path or not os.path.exists(path)
        ]
        if missing_inputs:
            precheck = build_multimodal_precheck(load_dataset_metadata())
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "训练前请先在“模型训练”页面构建离线数据集，"
                        f"缺失输入: {', '.join(missing_inputs)}",
                        "precheck": precheck,
                    }
                ),
                400,
            )

        dataset_path = resolved_input_paths["model_dataset"]
        market_path = resolved_input_paths["market_daily"]
        news_raw_path = resolved_input_paths["news_raw"]

        runtime.training_status.clear()
        runtime.training_status.update(
            {
                "is_training": True,
                "progress": 0,
                "message": "初始化训练环境...",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "error": None,
                "current_epoch": 0,
                "total_epochs": epochs,
                "result": None,
                "params": {
                    "stocks": stocks,
                    "days": days,
                    "epochs": epochs,
                    "dataset_path": dataset_path,
                    "market_path": market_path,
                    "news_raw_path": news_raw_path,
                },
            }
        )

    def run_training():
        try:
            with runtime.training_lock:
                runtime.training_status["progress"] = 10
                runtime.training_status["message"] = f"正在收集 {stocks} 只股票的数据（heavy 模式）..."

            cmd = [
                sys.executable,
                "train_multimodal_model.py",
                "--stocks",
                str(stocks),
                "--days",
                str(days),
                "--epochs",
                str(epochs),
                "--model-tier",
                "heavy",
            ]

            if dataset_path:
                cmd.extend(["--dataset-path", dataset_path])
            if market_path:
                cmd.extend(["--market-path", market_path])
            if news_raw_path:
                cmd.extend(["--news-raw-path", news_raw_path])

            runtime.logger.info("启动训练: %s", " ".join(cmd))

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=BASE_DIR,
            )

            if process.stdout is not None:
                for line in process.stdout:
                    line = line.strip()
                    runtime.logger.info("[训练] %s", line)

                    if "准备训练数据" in line or "获取股票" in line:
                        with runtime.training_lock:
                            runtime.training_status["progress"] = 20
                            runtime.training_status["message"] = "正在准备训练数据..."
                    elif "加载标准数据集" in line:
                        with runtime.training_lock:
                            runtime.training_status["progress"] = 20
                            runtime.training_status["message"] = "正在加载标准数据集..."
                    elif "解析特征字段" in line:
                        with runtime.training_lock:
                            runtime.training_status["progress"] = 30
                            runtime.training_status["message"] = "正在解析多模态特征..."
                    elif "构建新闻文本对齐映射" in line:
                        with runtime.training_lock:
                            runtime.training_status["progress"] = 40
                            runtime.training_status["message"] = "正在构建新闻文本向量..."
                    elif "处理股票" in line:
                        with runtime.training_lock:
                            runtime.training_status["progress"] = 40
                            runtime.training_status["message"] = "正在处理股票数据..."
                    elif "开始训练" in line:
                        with runtime.training_lock:
                            runtime.training_status["progress"] = 60
                            runtime.training_status["message"] = f"正在训练模型（第 0/{epochs} 轮）..."
                    elif "Epoch" in line:
                        match = re.search(r"Epoch\s+(\d+)\s*/\s*(\d+)", line, flags=re.IGNORECASE)
                        if match:
                            current_epoch = int(match.group(1))
                            total_epochs = int(match.group(2))
                            epoch_progress = 60 + int((current_epoch / max(total_epochs, 1)) * 25)
                            with runtime.training_lock:
                                runtime.training_status["progress"] = max(
                                    runtime.training_status.get("progress", 0),
                                    min(epoch_progress, 85),
                                )
                                runtime.training_status["message"] = (
                                    f"正在训练模型（第 {current_epoch}/{total_epochs} 轮）..."
                                )
                                runtime.training_status["current_epoch"] = current_epoch
                                runtime.training_status["total_epochs"] = total_epochs
                        else:
                            with runtime.training_lock:
                                runtime.training_status["progress"] = max(
                                    runtime.training_status.get("progress", 0),
                                    60,
                                )
                                total_epochs = int(runtime.training_status.get("total_epochs") or epochs)
                                current_epoch = int(runtime.training_status.get("current_epoch") or 0)
                                runtime.training_status["message"] = (
                                    f"正在训练模型（第 {current_epoch}/{total_epochs} 轮）..."
                                )
                    elif "保存" in line:
                        with runtime.training_lock:
                            runtime.training_status["progress"] = 90
                            runtime.training_status["message"] = "正在保存模型..."
                    elif "训练完成" in line or "模型已保存" in line:
                        with runtime.training_lock:
                            runtime.training_status["progress"] = 95
                            runtime.training_status["message"] = "训练完成，正在加载模型..."

            return_code = process.wait()
            if return_code != 0:
                raise RuntimeError(f"训练脚本返回错误码: {return_code}")

            train_result = {
                "production_model_replaced": None,
                "candidate_only": None,
                "model_available": None,
                "runtime_backend": None,
            }
            final_message = "训练完成，正在刷新模型状态..."

            try:
                model_info = runtime.refresh_multimodal_predictor()
                replaced = bool(model_info.get("production_model_replaced", True))
                candidate_only = bool(model_info.get("candidate_only", not replaced))
                model_available = bool(model_info.get("available", False))
                runtime_backend = model_info.get("runtime_backend") or model_info.get("backend") or "unknown"
                train_result = {
                    "production_model_replaced": replaced,
                    "candidate_only": candidate_only,
                    "model_available": model_available,
                    "runtime_backend": runtime_backend,
                }
                if replaced:
                    final_message = "训练完成，线上模型已更新"
                elif model_available:
                    final_message = "训练完成，新模型未超过基线，已保存候选模型（继续使用当前线上模型）"
                else:
                    final_message = "训练完成，新模型未超过基线且当前无可用线上模型，请先检查数据集质量评估"
                runtime.logger.info("✓ 模型重新加载成功")
            except Exception as exc:
                runtime.logger.error("模型重新加载失败: %s", exc)
                train_result = {
                    "production_model_replaced": None,
                    "candidate_only": None,
                    "model_available": bool(runtime.MULTIMODAL_MODEL_AVAILABLE),
                    "runtime_backend": "unknown",
                }
                final_message = f"训练完成，但模型状态刷新失败: {str(exc)}"

            with runtime.training_lock:
                runtime.training_status["progress"] = 100
                runtime.training_status["message"] = final_message
                runtime.training_status["end_time"] = datetime.now().isoformat()
                runtime.training_status["is_training"] = False
                runtime.training_status["current_epoch"] = int(runtime.training_status.get("total_epochs") or epochs)
                runtime.training_status["result"] = train_result
        except Exception as exc:
            runtime.logger.error("训练失败: %s", exc)
            with runtime.training_lock:
                runtime.training_status["is_training"] = False
                runtime.training_status["error"] = str(exc)
                runtime.training_status["message"] = f"训练失败: {str(exc)}"
                runtime.training_status["end_time"] = datetime.now().isoformat()
                runtime.training_status["result"] = None

    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "success": True,
            "message": "训练已启动，请通过 /api/model/train/status 查看进度",
            "params": {
                "stocks": stocks,
                "days": days,
                "epochs": epochs,
                "dataset_path": dataset_path,
                "market_path": market_path,
                "news_raw_path": news_raw_path,
            },
        }
    )


def get_train_status_api():
    """Get multimodal training status."""
    with runtime.training_lock:
        status = dict(runtime.training_status)
    return jsonify({"success": True, "status": status})
