#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stock and prediction APIs."""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Any

import numpy as np
from flask import jsonify

import src.web_runtime as runtime
from src.services.news_common import parse_optional_age_hours


def get_stock_info_api(stock_code: str, days: int):
    """Get stock price stats and kline."""
    try:
        if runtime.data_manager is None:
            return jsonify({"success": False, "message": "数据组件未初始化"}), 503

        df = runtime.data_manager.get_stock_kline(stock_code, days=days)
        if df.empty:
            return jsonify({"success": False, "message": f"获取股票 {stock_code} 数据失败"}), 500

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change = (latest["close"] - prev["close"]) / prev["close"] * 100 if prev["close"] != 0 else 0

        stats = {
            "current": round(latest["close"], 2),
            "change": round(change, 2),
            "high": round(df["high"].max(), 2),
            "low": round(df["low"].min(), 2),
            "volume": int(df["volume"].sum()),
            "avg_volume": int(df["volume"].mean()),
            "days": len(df),
        }

        kline_data = []
        for date, row in df.iterrows():
            kline_data.append(
                {
                    "date": date.strftime("%Y-%m-%d"),
                    "open": round(row["open"], 2),
                    "high": round(row["high"], 2),
                    "low": round(row["low"], 2),
                    "close": round(row["close"], 2),
                    "volume": int(row["volume"]),
                }
            )

        return jsonify({"success": True, "data": {"code": stock_code, "stats": stats, "kline": kline_data}})
    except Exception as exc:
        runtime.logger.error("获取股票信息失败: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500


def predict_multimodal_api(stock_code: str, params: dict):
    """Run single-stock multimodal prediction."""
    try:
        if runtime.data_manager is None:
            return jsonify({"success": False, "message": "数据组件未初始化"}), 503

        from src.multimodal_model import get_multimodal_predictor
        from src.news_crawler import get_news_crawler
        from src.news_impact_analyzer import get_news_impact_analyzer
        from src.relevance_graph import get_relevance_graph

        days = runtime.safe_int_param(params.get("days"), default=100, min_value=60, max_value=600)
        use_news = runtime.parse_bool_param(params.get("use_news", True), True)
        use_relevance = runtime.parse_bool_param(params.get("use_relevance", True), True)
        max_news_age_hours = parse_optional_age_hours(params.get("max_news_age_hours", 72))

        df = runtime.data_manager.get_stock_kline(stock_code, days=days)
        if df.empty or len(df) < 60:
            return (
                jsonify(
                    {
                        "success": False,
                        "message": f"数据不足，至少需要60天数据（当前：{len(df) if not df.empty else 0}天）",
                    }
                ),
                400,
            )

        news_list = []
        sector_impact = {}
        if use_news:
            try:
                crawler = get_news_crawler()
                news_list = crawler.get_news(
                    stock_code=stock_code,
                    limit=20,
                    source="all",
                    max_news_age_hours=max_news_age_hours,
                )
                analyzer = get_news_impact_analyzer()
                sector_impact = analyzer.get_sector_impact_vector(news_list)
            except Exception as exc:
                runtime.logger.warning("获取新闻数据失败: %s", exc)

        relevance_matrix = None
        stock_idx = 0
        if use_relevance:
            try:
                graph = get_relevance_graph()
                matrix_data = graph.get_relevance_matrix()
                relevance_matrix = np.array(matrix_data["matrix"])

                stock_codes = matrix_data["stock_codes"]
                if stock_code in stock_codes:
                    stock_idx = stock_codes.index(stock_code)
            except Exception as exc:
                runtime.logger.warning("获取相关性数据失败: %s", exc)

        predictor = get_multimodal_predictor()
        result = predictor.predict_stock(
            stock_code=stock_code,
            kline_df=df,
            news_list=news_list,
            sector_impact=sector_impact,
            relevance_matrix=relevance_matrix,
            stock_idx=stock_idx,
        )

        return jsonify({"success": True, "data": result})
    except Exception as exc:
        runtime.logger.error("多模态预测失败: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500


def get_batch_predict_status_api():
    """Get background stock recommendation task status."""
    with runtime.batch_predict_lock:
        status = dict(runtime.batch_predict_status)
    return jsonify({"success": True, "status": status})


def _build_batch_predict_context(params: dict) -> dict[str, Any]:
    if runtime.data_manager is None:
        raise RuntimeError("数据组件未初始化")

    from src.multimodal_model import get_multimodal_predictor

    predictor = get_multimodal_predictor()
    model_info = {}
    try:
        model_info = predictor.get_model_info() or {}
    except Exception:
        model_info = {}

    runtime_backend = str(model_info.get("runtime_backend", "")).lower()
    model_ready = bool(model_info.get("available", False))
    optimized_profile = runtime_backend == "tensorflow" and model_ready

    default_top_n = 30 if optimized_profile else 20
    default_kline_days = 240 if optimized_profile else 120
    default_news_limit = 30 if optimized_profile else 20
    default_min_price = 2.0 if optimized_profile else 0.0
    default_max_price = 300.0 if optimized_profile else 1000.0

    top_n = runtime.safe_int_param(params.get("top_n"), default=default_top_n, min_value=1, max_value=200)
    kline_days = runtime.safe_int_param(
        params.get("kline_days"),
        default=default_kline_days,
        min_value=60,
        max_value=600,
    )
    news_limit = runtime.safe_int_param(
        params.get("news_limit"),
        default=default_news_limit,
        min_value=1,
        max_value=2000,
    )
    max_news_age_hours = parse_optional_age_hours(params.get("max_news_age_hours", 72))
    use_news = runtime.parse_bool_param(params.get("use_news", True), True)
    use_relevance = runtime.parse_bool_param(params.get("use_relevance", True), True)
    min_price = runtime.safe_float_param(
        params.get("min_price"),
        default=default_min_price,
        min_value=0.0,
        max_value=100000.0,
    )
    max_price = runtime.safe_float_param(
        params.get("max_price"),
        default=default_max_price,
        min_value=0.0,
        max_value=100000.0,
    )

    if max_price < min_price:
        raise ValueError("max_price 不能小于 min_price")

    return {
        "top_n": top_n,
        "kline_days": kline_days,
        "news_limit": news_limit,
        "max_news_age_hours": max_news_age_hours,
        "use_news": use_news,
        "use_relevance": use_relevance,
        "min_price": min_price,
        "max_price": max_price,
        "optimized_profile": optimized_profile,
        "runtime_backend": runtime_backend,
        "model_ready": model_ready,
    }


def _execute_batch_prediction(config: dict[str, Any], update_status):
    from src.multimodal_model import get_multimodal_predictor
    from src.news_crawler import get_news_crawler
    from src.news_impact_analyzer import get_news_impact_analyzer
    from src.relevance_graph import get_relevance_graph

    predictor = get_multimodal_predictor()
    top_n = config["top_n"]
    kline_days = config["kline_days"]
    news_limit = config["news_limit"]
    max_news_age_hours = config["max_news_age_hours"]
    use_news = config["use_news"]
    use_relevance = config["use_relevance"]
    min_price = config["min_price"]
    max_price = config["max_price"]

    runtime.logger.info(
        "开始股票推荐，stock_scope=all_cache, top_n=%s, kline_days=%s, use_news=%s, use_relevance=%s, max_news_age_hours=%s",
        top_n,
        kline_days,
        use_news,
        use_relevance,
        max_news_age_hours,
    )

    update_status(5, "正在加载股票池...")
    stock_list = runtime.data_manager.get_stock_list()
    if stock_list.empty:
        raise ValueError("无法获取股票列表")

    total_stocks = int(len(stock_list))
    update_status(
        12,
        "股票池加载完成，正在准备新闻与关联特征...",
        detail={"total_stocks": total_stocks, "current": 0, "analyzed": 0, "predictions": 0},
    )

    crawler = get_news_crawler()
    analyzer = get_news_impact_analyzer()

    relevance_matrix = None
    stock_idx_map = {}
    if use_relevance:
        try:
            matrix_data = get_relevance_graph().get_relevance_matrix()
            raw_matrix = matrix_data.get("matrix")
            raw_codes = matrix_data.get("stock_codes", [])
            if raw_matrix is not None and raw_codes:
                relevance_matrix = np.array(raw_matrix)
                stock_codes = [str(code).zfill(6) for code in raw_codes]
                stock_idx_map = {code: idx for idx, code in enumerate(stock_codes)}
        except Exception as exc:
            runtime.logger.warning("读取相关性矩阵失败，将降级为无相关性特征: %s", exc)
            relevance_matrix = None
            stock_idx_map = {}

    global_news = []
    if use_news:
        try:
            if crawler.is_available():
                update_status(18, "正在加载全市场新闻缓存...")
                global_news = crawler.get_news(
                    stock_code=None,
                    limit=max(300, min(5000, news_limit * 30)),
                    source="all",
                    use_cache=True,
                    max_news_age_hours=max_news_age_hours,
                )
            else:
                runtime.logger.warning("新闻爬虫不可用，股票推荐将降级为无新闻特征")
        except Exception as exc:
            runtime.logger.warning("获取全量新闻失败，将降级为无新闻特征: %s", exc)
            global_news = []

    stock_news_map = {}
    if use_news and global_news:
        update_status(24, "正在为股票映射相关新闻...")
        for idx, (_, stock) in enumerate(stock_list.iterrows(), 1):
            code = str(stock.get("code", "")).zfill(6)
            matches = crawler.filter_news_by_stock(global_news, code, limit=news_limit)
            stock_news_map[code] = matches
            if idx % 200 == 0 or idx == total_stocks:
                progress = 24 + int(idx / max(total_stocks, 1) * 8)
                update_status(
                    min(progress, 32),
                    f"正在为股票映射相关新闻...{idx}/{total_stocks}",
                    detail={
                        "total_stocks": total_stocks,
                        "current": idx,
                        "analyzed": 0,
                        "predictions": 0,
                    },
                )

    predictions = []
    analyzed_count = 0
    for idx, (_, stock) in enumerate(stock_list.iterrows(), 1):
        stock_code_iter = str(stock["code"]).zfill(6)
        stock_name = stock.get("name", stock_code_iter)

        try:
            df = runtime.data_manager.get_stock_kline(stock_code_iter, days=kline_days)
            if df.empty or len(df) < 60:
                continue

            latest_price = float(df["close"].iloc[-1])
            if latest_price < min_price or latest_price > max_price:
                continue

            news_list = stock_news_map.get(stock_code_iter, []) if use_news else []
            sector_impact = analyzer.get_sector_impact_vector(news_list) if news_list else {}

            relevance_matrix_input = None
            stock_idx = 0
            if use_relevance and relevance_matrix is not None and stock_code_iter in stock_idx_map:
                relevance_matrix_input = relevance_matrix
                stock_idx = stock_idx_map[stock_code_iter]

            result = predictor.predict_stock(
                stock_code=stock_code_iter,
                kline_df=df,
                news_list=news_list,
                sector_impact=sector_impact,
                relevance_matrix=relevance_matrix_input,
                stock_idx=stock_idx,
            )

            if result.get("success"):
                predictions.append(
                    {
                        "stock_code": stock_code_iter,
                        "stock_name": stock_name,
                        "latest_price": latest_price,
                        "prediction": result["prediction"],
                        "prediction_text": result["prediction_text"],
                        "probability": float(round(result["probability"], 4)),
                        "confidence": float(round(result["confidence"], 4)),
                        "expected_return": float(round(result.get("expected_return", 0.0), 2)),
                        "predicted_price": float(round(result.get("predicted_price", latest_price), 2)),
                        "predict_horizon_days": 5,
                        "news_count": int(result.get("news_count", len(news_list))),
                        "backend": result.get("backend", "rule_based"),
                    }
                )
                analyzed_count += 1
        except Exception as exc:
            runtime.logger.debug("预测股票 %s 失败: %s", stock_code_iter, exc)
            continue

        if idx % 5 == 0 or idx == total_stocks:
            progress = 32 + int(idx / max(total_stocks, 1) * 63)
            update_status(
                min(progress, 95),
                f"正在进行股票推荐...{idx}/{total_stocks}",
                detail={
                    "total_stocks": total_stocks,
                    "current": idx,
                    "current_stock": stock_code_iter,
                    "analyzed": analyzed_count,
                    "predictions": len(predictions),
                },
            )

    predictions.sort(key=lambda x: x["expected_return"], reverse=True)
    top_predictions = predictions[:top_n]
    for idx, pred in enumerate(top_predictions, 1):
        pred["rank"] = idx

    up_count = sum(1 for p in predictions if p["prediction"] == 1)
    down_count = len(predictions) - up_count

    runtime.logger.info(
        "股票推荐完成，分析 %s 只股票，上涨%s只，下跌%s只",
        analyzed_count,
        up_count,
        down_count,
    )

    return {
        "predictions": top_predictions,
        "total_analyzed": analyzed_count,
        "total_predictions": len(predictions),
        "total_cached_stocks": total_stocks,
        "up_count": up_count,
        "down_count": down_count,
        "analysis_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "predict_horizon_days": 5,
        "input_params": {
            "top_n": top_n,
            "kline_days": kline_days,
            "news_limit": news_limit,
            "max_news_age_hours": max_news_age_hours,
            "use_news": use_news,
            "use_relevance": use_relevance,
            "min_price": min_price,
            "max_price": max_price,
            "optimized_profile": bool(config.get("optimized_profile", False)),
        },
        "model_info": {
            "model_name": "multimodal_stock_predictor",
            "predict_horizon_days": 5,
            "runtime_backend": config.get("runtime_backend") or "unknown",
            "model_available": bool(config.get("model_ready", False)),
        },
    }


def predict_batch_api(params: dict):
    """Start background stock recommendation task."""
    with runtime.batch_predict_lock:
        if runtime.batch_predict_status.get("is_running"):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "股票推荐任务正在运行中，请稍候...",
                        "status": dict(runtime.batch_predict_status),
                    }
                ),
                400,
            )

    try:
        config = _build_batch_predict_context(params)
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        runtime.logger.error("初始化股票推荐任务失败: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500

    with runtime.batch_predict_lock:
        if runtime.batch_predict_status.get("is_running"):
            return (
                jsonify(
                    {
                        "success": False,
                        "message": "股票推荐任务正在运行中，请稍候...",
                        "status": dict(runtime.batch_predict_status),
                    }
                ),
                400,
            )

        runtime.batch_predict_status.clear()
        runtime.batch_predict_status.update(
            {
                "is_running": True,
                "progress": 0,
                "message": "初始化股票推荐任务...",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "error": None,
                "result": None,
                "params": {
                    "top_n": config["top_n"],
                    "kline_days": config["kline_days"],
                    "news_limit": config["news_limit"],
                    "max_news_age_hours": config["max_news_age_hours"],
                    "use_news": config["use_news"],
                    "use_relevance": config["use_relevance"],
                    "min_price": config["min_price"],
                    "max_price": config["max_price"],
                },
                "detail": None,
            }
        )

    def update_status(progress: int, message: str, detail: dict[str, Any] | None = None):
        with runtime.batch_predict_lock:
            runtime.batch_predict_status["progress"] = int(max(0, min(progress, 100)))
            runtime.batch_predict_status["message"] = message
            runtime.batch_predict_status["detail"] = detail or None

    def run_background_task():
        try:
            result = _execute_batch_prediction(config, update_status)
            with runtime.batch_predict_lock:
                runtime.batch_predict_status["is_running"] = False
                runtime.batch_predict_status["progress"] = 100
                runtime.batch_predict_status["message"] = "股票推荐完成"
                runtime.batch_predict_status["result"] = result
                runtime.batch_predict_status["error"] = None
                runtime.batch_predict_status["end_time"] = datetime.now().isoformat()
                runtime.batch_predict_status["detail"] = {
                    "total_stocks": result.get("total_cached_stocks", 0),
                    "analyzed": result.get("total_analyzed", 0),
                    "predictions": result.get("total_predictions", 0),
                }
        except Exception as exc:
            runtime.logger.error("股票推荐失败: %s", exc)
            with runtime.batch_predict_lock:
                runtime.batch_predict_status["is_running"] = False
                runtime.batch_predict_status["error"] = str(exc)
                runtime.batch_predict_status["message"] = f"股票推荐失败: {str(exc)}"
                runtime.batch_predict_status["end_time"] = datetime.now().isoformat()

    thread = threading.Thread(target=run_background_task)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "success": True,
            "message": "股票推荐任务已启动，请通过 /api/predict/batch/status 查看进度",
            "params": dict(runtime.batch_predict_status.get("params") or {}),
        }
    )
