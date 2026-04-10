#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Offline evaluation and backtest APIs."""

from __future__ import annotations

import json

from flask import current_app, jsonify

import src.web_runtime as runtime
from src.services.dataset_helpers import load_latest_json_report


def get_latest_evaluation_report_raw_api():
    """Return latest offline evaluation report JSON."""
    report = load_latest_json_report("offline_evaluation_report_latest.json")
    if not report:
        return jsonify({"success": False, "message": "暂无可用的离线评估报告"}), 404

    return current_app.response_class(
        response=json.dumps(report, ensure_ascii=False, indent=2),
        status=200,
        mimetype="application/json",
    )


def run_offline_evaluation_api(params: dict):
    """Run offline time-split evaluation."""
    try:
        model_type = str(params.get("model_type", "logistic")).strip().lower()
        top_k = int(params.get("top_k", 20))
        train_ratio = float(params.get("train_ratio", 0.70))
        valid_ratio = float(params.get("valid_ratio", 0.15))
        test_ratio = float(params.get("test_ratio", 0.15))
        train_days = int(params.get("train_days", 180))
        valid_days = int(params.get("valid_days", 30))
        test_days = int(params.get("test_days", 30))
        step_days = int(params.get("step_days", 20))
        rolling_windows = int(params.get("rolling_windows", 3))
        dataset_path = params.get("dataset_path")
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "参数格式错误，请检查评估参数类型"}), 400

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

    if isinstance(dataset_path, str):
        dataset_path = dataset_path.strip() or None

    try:
        from src.evaluator import OfflineEvaluator

        evaluator = OfflineEvaluator(dataset_path=dataset_path)
        report = evaluator.run(
            model_type=model_type,
            top_k=max(1, min(top_k, 200)),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            train_days=max(60, train_days),
            valid_days=max(10, valid_days),
            test_days=max(10, test_days),
            step_days=max(5, step_days),
            rolling_windows=max(1, min(rolling_windows, 12)),
        )

        return jsonify(
            {
                "success": True,
                "message": "离线评估完成",
                "data": {
                    "report_path": report.get("report_path"),
                    "latest_path": report.get("latest_path"),
                    "holdout_results": report.get("holdout_results", {}),
                    "holdout_uplift_full_vs_tech": report.get("holdout_uplift_full_vs_tech", {}),
                    "rolling_summary": report.get("rolling_summary", {}),
                    "rolling_windows": len(report.get("rolling_results", [])),
                },
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        runtime.logger.error("离线评估失败: %s", exc)
        return jsonify({"success": False, "message": f"离线评估失败: {str(exc)}"}), 500


def get_latest_backtest_report_raw_api():
    """Return latest cross-section backtest report JSON."""
    report = load_latest_json_report("cross_section_backtest_latest.json")
    if not report:
        return jsonify({"success": False, "message": "暂无可用的截面回测报告"}), 404

    return current_app.response_class(
        response=json.dumps(report, ensure_ascii=False, indent=2),
        status=200,
        mimetype="application/json",
    )


def run_cross_section_backtest_api(params: dict):
    """Run cross-section stock-selection backtest."""
    try:
        model_type = str(params.get("model_type", "logistic")).strip().lower()
        feature_set = str(params.get("feature_set", "all_features")).strip().lower()
        top_n = int(params.get("top_n", 20))
        hold_days = int(params.get("hold_days", 5))
        train_ratio = float(params.get("train_ratio", 0.70))
        valid_ratio = float(params.get("valid_ratio", 0.15))
        test_ratio = float(params.get("test_ratio", 0.15))
        commission_rate = float(params.get("commission_rate", 0.0003))
        stamp_tax_rate = float(params.get("stamp_tax_rate", 0.001))
        slippage_rate = float(params.get("slippage_rate", 0.0002))
        dataset_path = params.get("dataset_path")
    except (TypeError, ValueError):
        return jsonify({"success": False, "message": "参数格式错误，请检查回测参数类型"}), 400

    if model_type not in ["logistic", "random_forest"]:
        return (
            jsonify({"success": False, "message": "model_type 只支持 logistic 或 random_forest"}),
            400,
        )

    if feature_set not in ["all_features", "technical_only"]:
        return (
            jsonify(
                {
                    "success": False,
                    "message": "feature_set 只支持 all_features 或 technical_only",
                }
            ),
            400,
        )

    if isinstance(dataset_path, str):
        dataset_path = dataset_path.strip() or None

    try:
        from src.backtest_engine import CrossSectionBacktestEngine

        engine = CrossSectionBacktestEngine(dataset_path=dataset_path)
        report = engine.run(
            model_type=model_type,
            feature_set=feature_set,
            top_n=max(1, min(top_n, 200)),
            hold_days=max(1, min(hold_days, 20)),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            commission_rate=max(0.0, commission_rate),
            stamp_tax_rate=max(0.0, stamp_tax_rate),
            slippage_rate=max(0.0, slippage_rate),
        )

        return jsonify(
            {
                "success": True,
                "message": "截面回测完成",
                "data": {
                    "report_path": report.get("report_path"),
                    "latest_path": report.get("latest_path"),
                    "summary": report.get("summary", {}),
                    "split": report.get("split", {}),
                    "sector_distribution": report.get("sector_distribution", {}),
                    "equity_curve_points": len(report.get("equity_curve", [])),
                    "holding_days": len(report.get("daily_holdings", [])),
                    "equity_curve": report.get("equity_curve", []),
                    "daily_holdings": report.get("daily_holdings", []),
                    "config": report.get("config", {}),
                },
            }
        )
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        runtime.logger.error("截面回测失败: %s", exc)
        return jsonify({"success": False, "message": f"截面回测失败: {str(exc)}"}), 500
