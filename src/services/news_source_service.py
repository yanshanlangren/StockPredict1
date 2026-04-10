#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""News source management APIs."""

from __future__ import annotations

from flask import jsonify

import src.web_runtime as runtime
from src.news_source_registry import SUPPORTED_NEWS_ADAPTERS, get_news_source_registry


def list_news_sources_api(include_disabled_raw):
    """List news source configs for dropdown and management."""
    include_disabled = runtime.parse_bool_param(include_disabled_raw, True)
    registry = get_news_source_registry()
    sources = registry.list_sources(include_disabled=include_disabled)

    return jsonify(
        {
            "success": True,
            "data": {
                "sources": sources,
                "enabled_source_ids": registry.list_enabled_source_ids(),
                "adapters": list(SUPPORTED_NEWS_ADAPTERS),
                "adapter_templates": registry.get_adapter_templates(),
                "config_file": registry.config_file,
            },
        }
    )


def create_news_source_api(params: dict):
    """Create a custom news source config."""
    registry = get_news_source_registry()
    try:
        source = registry.create_source(params or {})
        return jsonify({"success": True, "message": "新闻源创建成功", "data": source})
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        runtime.logger.error("创建新闻源失败: %s", exc)
        return jsonify({"success": False, "message": f"创建新闻源失败: {str(exc)}"}), 500


def update_news_source_api(source_id: str, params: dict):
    """Update existing news source config."""
    registry = get_news_source_registry()
    try:
        source = registry.update_source(source_id, params or {})
        return jsonify({"success": True, "message": "新闻源更新成功", "data": source})
    except KeyError as exc:
        return jsonify({"success": False, "message": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        runtime.logger.error("更新新闻源失败: %s", exc)
        return jsonify({"success": False, "message": f"更新新闻源失败: {str(exc)}"}), 500


def delete_news_source_api(source_id: str):
    """Delete news source config."""
    registry = get_news_source_registry()
    try:
        registry.delete_source(source_id)
        return jsonify({"success": True, "message": "新闻源删除成功"})
    except KeyError as exc:
        return jsonify({"success": False, "message": str(exc)}), 404
    except ValueError as exc:
        return jsonify({"success": False, "message": str(exc)}), 400
    except Exception as exc:
        runtime.logger.error("删除新闻源失败: %s", exc)
        return jsonify({"success": False, "message": f"删除新闻源失败: {str(exc)}"}), 500
