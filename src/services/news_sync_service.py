#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""News synchronization task APIs."""

from __future__ import annotations

import threading
from datetime import datetime

from flask import jsonify

import src.web_runtime as runtime
from src.services.news_common import parse_optional_age_hours


def get_news_sync_status_api():
    """Get all-source news sync status."""
    with runtime.news_sync_lock:
        status = dict(runtime.news_sync_status)
    return jsonify({"success": True, "status": status})


def _normalize_sources(requested_sources):
    default_sources = ["eastmoney", "sina", "tencent"]
    if isinstance(requested_sources, list) and requested_sources:
        allowed = {"eastmoney", "sina", "tencent"}
        sources = [
            str(item).strip().lower()
            for item in requested_sources
            if str(item).strip().lower() in allowed
        ]
        if sources:
            return sources
    return default_sources


def start_news_sync_api(params: dict):
    """Start all-source news synchronization."""
    with runtime.news_sync_lock:
        if runtime.news_sync_status.get("is_running"):
            return (
                jsonify({"success": False, "message": "全源新闻同步任务正在运行中，请稍候..."}),
                400,
            )

        limit_per_source = runtime.safe_int_param(
            params.get("limit_per_source"),
            default=300,
            min_value=50,
            max_value=2000,
        )
        max_news_age_hours = parse_optional_age_hours(params.get("max_news_age_hours", None))
        sources = _normalize_sources(params.get("sources"))

        runtime.news_sync_status.clear()
        runtime.news_sync_status.update(
            {
                "is_running": True,
                "progress": 0,
                "message": "初始化全源新闻同步任务...",
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "error": None,
                "result": None,
                "detail": None,
                "params": {
                    "sources": sources,
                    "limit_per_source": limit_per_source,
                    "max_news_age_hours": max_news_age_hours,
                },
            }
        )

    def update_news_status(progress, message, detail=None):
        with runtime.news_sync_lock:
            runtime.news_sync_status["progress"] = progress
            runtime.news_sync_status["message"] = message
            runtime.news_sync_status["detail"] = detail or None

    def run_news_sync():
        try:
            from src.news_crawler import get_news_crawler

            crawler = get_news_crawler()
            if not crawler.is_available():
                raise RuntimeError("新闻爬虫不可用，请先安装或启用 akshare")

            source_counts = {}
            all_news_items = []
            total_sources = len(sources)

            for idx, source in enumerate(sources, 1):
                begin_progress = 5 + int((idx - 1) / max(total_sources, 1) * 80)
                update_news_status(
                    begin_progress,
                    f"正在同步 {source} 新闻源...",
                    {
                        "current_source": source,
                        "source_index": idx,
                        "source_total": total_sources,
                        "source_counts": source_counts,
                    },
                )

                try:
                    fetched = crawler.get_news(
                        stock_code=None,
                        limit=limit_per_source,
                        use_cache=False,
                        source=source,
                        max_news_age_hours=max_news_age_hours,
                    )
                except Exception as source_error:
                    runtime.logger.warning("同步新闻源 %s 失败: %s", source, source_error)
                    fetched = []

                source_counts[source] = len(fetched)
                all_news_items.extend(fetched)

                end_progress = 5 + int(idx / max(total_sources, 1) * 80)
                update_news_status(
                    end_progress,
                    f"已完成 {source} 新闻源，同步 {len(fetched)} 条",
                    {
                        "current_source": source,
                        "source_index": idx,
                        "source_total": total_sources,
                        "source_counts": source_counts,
                    },
                )

            deduped = crawler.deduplicate_news(all_news_items)
            cache_file = crawler.save_news_cache(deduped, stock_code=None, source="all")
            statistics = crawler.get_news_statistics(deduped)
            result = {
                "sources": sources,
                "source_counts": source_counts,
                "total_raw": len(all_news_items),
                "total_unique": len(deduped),
                "dedup_removed": max(len(all_news_items) - len(deduped), 0),
                "max_news_age_hours": max_news_age_hours,
                "cache_file": cache_file,
                "statistics": statistics,
                "synced_at": datetime.now().isoformat(),
            }

            with runtime.news_sync_lock:
                runtime.news_sync_status["is_running"] = False
                runtime.news_sync_status["progress"] = 100
                runtime.news_sync_status["message"] = "全源新闻同步完成"
                runtime.news_sync_status["end_time"] = datetime.now().isoformat()
                runtime.news_sync_status["result"] = result
                runtime.news_sync_status["detail"] = {
                    "source_index": total_sources,
                    "source_total": total_sources,
                    "source_counts": source_counts,
                    "total_unique": len(deduped),
                    "max_news_age_hours": max_news_age_hours,
                }
        except Exception as exc:
            runtime.logger.error("全源新闻同步失败: %s", exc)
            with runtime.news_sync_lock:
                runtime.news_sync_status["is_running"] = False
                runtime.news_sync_status["error"] = str(exc)
                runtime.news_sync_status["message"] = f"全源新闻同步失败: {str(exc)}"
                runtime.news_sync_status["end_time"] = datetime.now().isoformat()

    thread = threading.Thread(target=run_news_sync)
    thread.daemon = True
    thread.start()

    return jsonify(
        {
            "success": True,
            "message": "全源新闻同步任务已启动，请通过 /api/system/news-sync/status 查看进度",
            "params": {
                "sources": sources,
                "limit_per_source": limit_per_source,
                "max_news_age_hours": max_news_age_hours,
            },
        }
    )
