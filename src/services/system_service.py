#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""System APIs: stock list and market update task."""

from __future__ import annotations

import threading
from datetime import datetime

from flask import jsonify

import src.web_runtime as runtime


def get_stocks_api(force_refresh: bool = False):
    """Get stock list."""
    try:
        if runtime.data_manager is None:
            return jsonify({"success": False, "message": "数据组件未初始化"}), 503

        if force_refresh:
            runtime.logger.info("强制刷新股票列表...")
            stock_list = runtime.data_manager.get_stock_list(force_refresh=True)
        else:
            stock_list = runtime.data_manager.get_stock_list()

        if stock_list.empty:
            return jsonify({"success": False, "message": "获取股票列表失败"}), 500

        if "code" in stock_list.columns:
            stock_list = stock_list.copy()
            stock_list["code"] = stock_list["code"].astype(str)

        return jsonify(
            {
                "success": True,
                "data": stock_list.to_dict("records"),
                "total": len(stock_list),
                "refreshed": force_refresh,
            }
        )
    except Exception as exc:
        runtime.logger.error("获取股票列表失败: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500


def get_market_update_status_api():
    """Get full-market update task status."""
    with runtime.market_update_lock:
        status = dict(runtime.market_update_status)

    return jsonify({"success": True, "status": status})


def _parse_market_update_params(params: dict):
    days = max(120, min(int(params.get("days", 400)), 2500))
    refresh_stock_list = runtime.parse_bool_param(params.get("refresh_stock_list", True), True)
    return days, refresh_stock_list


def start_market_update_api(params: dict):
    """Start background update for all stock market data."""
    with runtime.market_update_lock:
        if runtime.market_update_status.get("is_running"):
            return (
                jsonify({"success": False, "message": "全量行情更新任务正在运行中，请稍候..."}),
                400,
            )

        days, refresh_stock_list = _parse_market_update_params(params)

        runtime.market_update_status.clear()
        runtime.market_update_status.update(
            {
                "is_running": True,
                "progress": 0,
                "message": "初始化全量行情更新任务...",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "error": None,
                "result": None,
                "detail": None,
                "params": {"days": days, "refresh_stock_list": refresh_stock_list},
            }
        )

    def update_market_status(progress, message, detail=None):
        with runtime.market_update_lock:
            runtime.market_update_status["progress"] = progress
            runtime.market_update_status["message"] = message
            runtime.market_update_status["detail"] = detail or None

    def run_market_update():
        try:
            if runtime.data_manager is None:
                raise RuntimeError("数据源管理器未初始化")

            update_market_status(5, "正在拉取股票列表...")
            stock_df = runtime.data_manager.get_stock_list(force_refresh=refresh_stock_list)
            if stock_df is None or stock_df.empty:
                raise ValueError("未获取到股票列表，无法执行全量行情更新")
            if "code" not in stock_df.columns:
                raise ValueError("股票列表缺少 code 字段")

            codes = stock_df["code"].astype(str).str.zfill(6).tolist()
            total = len(codes)
            success_count = 0
            fail_count = 0
            failed_codes = []
            latest_trade_date = None

            for idx, stock_code in enumerate(codes, 1):
                try:
                    df = runtime.data_manager.get_stock_kline(stock_code, days=days, force_refresh=True)
                    if df is not None and not df.empty:
                        success_count += 1
                        last_dt = df.index[-1]
                        if hasattr(last_dt, "strftime"):
                            last_date_str = last_dt.strftime("%Y-%m-%d")
                            if latest_trade_date is None or last_date_str > latest_trade_date:
                                latest_trade_date = last_date_str
                    else:
                        fail_count += 1
                        if len(failed_codes) < 30:
                            failed_codes.append(stock_code)
                except Exception as stock_error:
                    runtime.logger.warning("更新股票 %s 行情失败: %s", stock_code, stock_error)
                    fail_count += 1
                    if len(failed_codes) < 30:
                        failed_codes.append(stock_code)

                progress = 10 + int(idx / max(total, 1) * 88)
                update_market_status(
                    min(progress, 98),
                    f"更新行情进度 {idx}/{total}（当前: {stock_code}）",
                    {
                        "current": idx,
                        "total": total,
                        "current_stock": stock_code,
                        "success_count": success_count,
                        "fail_count": fail_count,
                        "latest_trade_date": latest_trade_date,
                    },
                )

            result = {
                "total_stocks": total,
                "success_count": success_count,
                "fail_count": fail_count,
                "failed_codes_preview": failed_codes,
                "latest_trade_date": latest_trade_date,
                "updated_at": datetime.now().isoformat(),
                "target_date": datetime.now().strftime("%Y-%m-%d"),
            }

            with runtime.market_update_lock:
                runtime.market_update_status["is_running"] = False
                runtime.market_update_status["progress"] = 100
                runtime.market_update_status["message"] = "全量行情更新完成"
                runtime.market_update_status["end_time"] = datetime.now().isoformat()
                runtime.market_update_status["result"] = result
                runtime.market_update_status["detail"] = {
                    "current": total,
                    "total": total,
                    "success_count": success_count,
                    "fail_count": fail_count,
                    "latest_trade_date": latest_trade_date,
                }
        except Exception as exc:
            runtime.logger.error("全量行情更新失败: %s", exc)
            with runtime.market_update_lock:
                runtime.market_update_status["is_running"] = False
                runtime.market_update_status["error"] = str(exc)
                runtime.market_update_status["message"] = f"全量行情更新失败: {str(exc)}"
                runtime.market_update_status["end_time"] = datetime.now().isoformat()

    thread = threading.Thread(target=run_market_update)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "success": True,
            "message": "全量行情更新任务已启动，请通过 /api/system/market-update/status 查看进度",
            "params": {"days": days, "refresh_stock_list": refresh_stock_list},
        }
    )
