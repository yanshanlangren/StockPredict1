#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Page route registration."""

from __future__ import annotations

from flask import render_template


def register_page_routes(app):
    """Register html page routes."""

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/overview")
    def overview_page():
        return render_template("overview.html")

    @app.route("/stock")
    def stock_page():
        return render_template("stock.html")

    @app.route("/news")
    def news_page():
        return render_template("news.html")

    @app.route("/predict")
    def predict_page():
        return render_template("predict.html")

    @app.route("/dataset")
    def dataset_page():
        return render_template("dataset.html")
