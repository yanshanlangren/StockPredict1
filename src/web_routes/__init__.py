"""Web route registration modules."""

from src.web_routes.core_api_routes import register_core_api_routes
from src.web_routes.news_predict_routes import register_news_predict_routes
from src.web_routes.page_routes import register_page_routes

__all__ = [
    "register_core_api_routes",
    "register_news_predict_routes",
    "register_page_routes",
]
