#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""股票交易AI系统 - Flask 应用入口（仅路由与初始化）。"""

from __future__ import annotations

from flask import Flask

import src.web_runtime as runtime
from src.web_routes import (
    register_core_api_routes,
    register_news_predict_routes,
    register_page_routes,
)


app = Flask(
    __name__,
    template_folder="app/templates",
    static_folder="app/static",
)

app.register_error_handler(Exception, runtime.handle_exception)
app.register_error_handler(404, runtime.handle_404)
app.register_error_handler(500, runtime.handle_500)
app.before_request(runtime.initialize_if_needed)

register_page_routes(app)
register_core_api_routes(app)
register_news_predict_routes(app)


if __name__ == "__main__":
    runtime.logger.info("=" * 50)
    runtime.logger.info("股票交易AI系统（多模态模型版）启动中...")
    runtime.logger.info("=" * 50)

    runtime.init_components()
    app.run(host="0.0.0.0", port=5001, debug=True)
