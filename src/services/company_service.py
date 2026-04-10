#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Company and relevance graph APIs."""

from __future__ import annotations

from flask import jsonify

import src.web_runtime as runtime


def get_company_info_api(stock_code: str):
    """Get company profile + financial + business analysis."""
    try:
        from src.company_info_engine import get_company_info_engine

        engine = get_company_info_engine()
        company_info = engine.get_company_info(stock_code)
        financial_data = engine.get_financial_data(stock_code)
        business_analysis = engine.analyze_business_structure(stock_code)

        return jsonify(
            {
                "success": True,
                "data": {
                    "info": company_info,
                    "financial": financial_data,
                    "business": business_analysis,
                },
            }
        )
    except Exception as exc:
        runtime.logger.error("获取公司信息失败: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500


def get_relevance_graph_api(stock_code: str):
    """Get stock relevance graph."""
    try:
        from src.relevance_graph import get_relevance_graph

        graph = get_relevance_graph()
        graph_data = graph.get_stock_relevance_graph(stock_code, depth=2)

        return jsonify({"success": True, "data": graph_data})
    except Exception as exc:
        runtime.logger.error("获取相关性图谱失败: %s", exc)
        return jsonify({"success": False, "message": str(exc)}), 500
