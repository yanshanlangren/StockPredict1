#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新闻爬取引擎 - 财经新闻数据采集

功能：
1. 从多个财经网站爬取新闻
2. 新闻分类和情感分析
3. 新闻缓存管理
"""

import os
import json
import logging
import hashlib
import math
from typing import Any, Optional, List, Dict, Tuple, Set
from datetime import datetime, timedelta
from urllib.parse import urlparse, quote, urljoin
import re

import pandas as pd
import numpy as np

from src.news_source_registry import get_news_source_registry

logger = logging.getLogger(__name__)

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
    logger.info("✓ akshare库已加载，新闻爬取功能可用")
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.error("✗ akshare库未安装，新闻爬取功能不可用")

try:
    from curl_cffi import requests as curl_requests
except ImportError:
    curl_requests = None

try:
    import requests as std_requests
except ImportError:
    std_requests = None

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


class NewsCrawler:
    """财经新闻爬取引擎"""

    def __init__(self, cache_dir: str = "data/cache/news"):
        """
        初始化新闻爬虫

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.cache_ttl_hours = 1.0

        # 新闻类别映射
        self.category_map = {
            "财经": ["财经", "经济", "金融", "投资", "理财"],
            "股票": ["股票", "A股", "港股", "美股", "股市"],
            "基金": ["基金", "ETF", "公募", "私募"],
            "债券": ["债券", "国债", "企业债"],
            "期货": ["期货", "大宗", "商品"],
            "外汇": ["外汇", "汇率", "人民币", "美元"],
            "银行": ["银行", "存款", "贷款", "利率"],
            "保险": ["保险", "寿险", "财险"],
            "宏观": ["宏观", "GDP", "CPI", "PMI", "政策"],
            "公司": ["公司", "企业", "财报", "业绩"],
        }

        # 领域关键词映射
        self.sector_keywords = {
            "银行": ["银行", "贷款", "存款", "利率", "LPR"],
            "证券": ["证券", "券商", "投行", "IPO", "上市"],
            "保险": ["保险", "寿险", "财险", "养老金"],
            "地产": ["房地产", "楼盘", "房价", "土地", "万科"],
            "汽车": ["汽车", "新能源车", "电动车", "比亚迪", "特斯拉"],
            "科技": ["科技", "人工智能", "AI", "芯片", "半导体"],
            "医药": ["医药", "医疗", "生物", "疫苗", "创新药"],
            "消费": ["消费", "零售", "电商", "白酒", "食品"],
            "能源": ["能源", "石油", "天然气", "新能源", "光伏"],
            "军工": ["军工", "国防", "航空", "航天"],
            "基建": ["基建", "建筑", "工程", "水泥", "钢铁"],
            "传媒": ["传媒", "影视", "游戏", "广告"],
            "教育": ["教育", "培训", "学校"],
            "农业": ["农业", "粮食", "种子", "养殖"],
        }

        self.positive_words = {
            "上涨": 1.1,
            "增长": 1.0,
            "盈利": 1.0,
            "突破": 1.0,
            "创新高": 1.2,
            "利好": 1.2,
            "增持": 0.9,
            "回购": 1.0,
            "分红": 0.8,
            "超预期": 1.2,
            "领涨": 0.9,
            "涨停": 1.0,
            "高景气": 1.1,
            "订单充足": 1.0,
            "业绩预增": 1.0,
        }
        self.negative_words = {
            "下跌": 1.1,
            "亏损": 1.2,
            "暴跌": 1.3,
            "利空": 1.2,
            "减持": 1.0,
            "质押": 0.8,
            "违约": 1.3,
            "退市": 1.4,
            "熊市": 1.0,
            "跌停": 1.1,
            "下滑": 1.0,
            "预警": 0.9,
            "风险": 0.8,
            "诉讼": 0.8,
            "处罚": 1.0,
            "调查": 0.8,
            "爆雷": 1.3,
        }
        self.semantic_positive_patterns = [
            "同比增长",
            "环比增长",
            "扭亏为盈",
            "业绩超预期",
            "签订大单",
            "获批",
            "政策支持",
            "回购计划",
        ]
        self.semantic_negative_patterns = [
            "同比下滑",
            "环比下滑",
            "业绩不及预期",
            "暂停上市",
            "被立案调查",
            "终止合作",
            "大幅亏损",
            "减值损失",
        ]
        self.negation_words = ["不", "未", "无", "并非", "否认", "难以", "不能", "没有"]
        self.intensify_words = ["显著", "大幅", "强劲", "持续", "明显", "急剧", "再度", "全面"]
        self.uncertain_words = ["或", "可能", "预计", "或将", "传闻", "据悉", "传", "拟"]

        self._stock_alias_map: Dict[str, List[str]] = {}
        self._stock_alias_cache_time: Optional[datetime] = None
        self._stock_alias_ttl_hours = 6.0
        self._static_stock_name_map = {
            "600000": ["浦发银行", "浦发"],
            "600519": ["贵州茅台", "茅台"],
            "600036": ["招商银行", "招行"],
            "000001": ["平安银行", "平安"],
            "000002": ["万科A", "万科"],
            "600690": ["海尔智家", "海尔"],
            "000858": ["五粮液"],
            "600030": ["中信证券"],
        }
        self._last_fetch_meta = {
            "bad_timestamp_count": 0,
            "dedup_removed_count": 0,
            "age_filtered_count": 0,
            "stock_filtered_out_count": 0,
        }

    def is_available(self) -> bool:
        """检查爬虫是否可用"""
        return AKSHARE_AVAILABLE

    def get_cache_file_path(self, stock_code: Optional[str] = None, source: str = "all") -> str:
        """返回缓存文件路径（供外部任务复用）。"""
        return self._cache_file_path(stock_code=stock_code, source=source)

    def deduplicate_news(self, news_list: List[Dict]) -> List[Dict]:
        """对新闻列表执行统一去重。"""
        deduped, _ = self._deduplicate_news(news_list)
        return deduped

    def save_news_cache(self, news_list: List[Dict], stock_code: Optional[str] = None, source: str = "all") -> str:
        """保存新闻缓存并返回缓存文件路径。"""
        self._save_to_cache(news_list=news_list, stock_code=stock_code, source=source)
        return self._cache_file_path(stock_code=stock_code, source=source)

    def filter_news_by_stock(self, news_list: List[Dict], stock_code: str, limit: Optional[int] = None) -> List[Dict]:
        """按股票筛选新闻并按时间降序输出。"""
        filtered = self._filter_by_stock(news_list, stock_code)
        filtered = sorted(
            filtered,
            key=lambda item: self._publish_timestamp_or_min(item.get("publish_time")),
            reverse=True,
        )
        if limit is not None:
            try:
                limit_value = max(int(limit), 1)
                return filtered[:limit_value]
            except Exception:
                pass
        return filtered

    def _list_enabled_source_configs(self) -> List[Dict]:
        registry = get_news_source_registry()
        source_items = registry.list_enabled_sources()
        if source_items:
            return source_items

        fallback_id = registry.get_default_source_id()
        fallback = registry.get_source(fallback_id)
        if fallback:
            return [fallback]

        return [
            {
                "source_id": "eastmoney",
                "name": "东方财富",
                "adapter": "json_api",
                "keyword": "财经新闻",
                "enabled": True,
                "adapter_config": {"preset": "eastmoney_search"},
            }
        ]

    def _resolve_fetch_source_configs(self, source: str) -> List[Dict]:
        normalized_source = self._normalize_source(source)
        enabled_sources = self._list_enabled_source_configs()
        source_map = {
            str(item.get("source_id", "")).strip(): item
            for item in enabled_sources
            if str(item.get("source_id", "")).strip()
        }
        if normalized_source == "all":
            return enabled_sources
        if normalized_source in source_map:
            return [source_map[normalized_source]]
        return enabled_sources[:1]

    def get_news(
        self,
        stock_code: Optional[str] = None,
        limit: int = 50,
        use_cache: bool = True,
        source: str = "eastmoney",
        max_news_age_hours: Optional[int] = 72,
    ) -> List[Dict]:
        """
        获取财经新闻

        Args:
            stock_code: 股票代码（可选，用于筛选相关新闻）
            limit: 返回数量限制
            use_cache: 是否使用缓存
            source: 新闻源 source_id 或 "all"
            max_news_age_hours: 新闻时效窗口（小时），None/<=0 表示不过滤

        Returns:
            新闻列表
        """
        if not AKSHARE_AVAILABLE:
            logger.error("akshare库不可用，无法获取新闻")
            return []

        try:
            limit_value = max(int(limit), 1)
        except Exception:
            limit_value = 50

        normalized_source = self._normalize_source(source)
        source_configs = self._resolve_fetch_source_configs(normalized_source)
        source_keys = [str(item.get("source_id", "")).strip() for item in source_configs]
        source_keys = [item for item in source_keys if item]
        cache_source_key = normalized_source if normalized_source == "all" else (
            source_keys[0] if source_keys else normalized_source
        )
        normalized_stock_code = self._normalize_stock_code(stock_code)
        max_age_hours = self._normalize_max_age_hours(max_news_age_hours)

        self._last_fetch_meta = {
            "bad_timestamp_count": 0,
            "dedup_removed_count": 0,
            "age_filtered_count": 0,
            "stock_filtered_out_count": 0,
        }

        if use_cache:
            cached = self._load_from_cache(
                stock_code=normalized_stock_code,
                source=cache_source_key,
                max_age_hours=max_age_hours,
            )
            if cached:
                if len(cached) >= limit_value:
                    logger.info(
                        "从缓存加载 %s 条新闻 (stock=%s, source=%s)",
                        len(cached),
                        normalized_stock_code or "all",
                        cache_source_key,
                    )
                    return cached[:limit_value]
                logger.info(
                    "缓存条数不足(%s < %s)，触发实时补拉 (stock=%s, source=%s)",
                    len(cached),
                    limit_value,
                    normalized_stock_code or "all",
                    cache_source_key,
                )

        try:
            fetched_news: List[Dict] = []
            for source_cfg in source_configs:
                source_key = str(source_cfg.get("source_id", "")).strip()
                if not source_key:
                    continue
                source_label = str(source_cfg.get("name") or source_key).strip() or source_key
                adapter_key = str(source_cfg.get("adapter") or source_key).strip().lower() or "eastmoney"
                keyword = str(source_cfg.get("keyword") or source_label).strip() or source_label

                try:
                    if normalized_source == "all":
                        target_count = min(max(limit_value, 120), 800)
                    else:
                        target_count = min(max(limit_value, 60), 2000)
                    rows = self._fetch_source_rows(source_cfg=source_cfg, max_items=target_count)
                    for row in rows:
                        news_item = self._parse_news_row(row)
                        if not news_item:
                            continue
                        news_item["source"] = source_key
                        news_item["source_name"] = source_label
                        fetched_news.append(news_item)
                except Exception as exc:
                    logger.warning("获取 %s(%s) 新闻失败: %s", source_key, adapter_key, exc)

            if normalized_stock_code and fetched_news:
                before_filter = len(fetched_news)
                fetched_news = self._filter_by_stock(fetched_news, normalized_stock_code)
                self._last_fetch_meta["stock_filtered_out_count"] = max(before_filter - len(fetched_news), 0)

            deduped_news, dedup_removed = self._deduplicate_news(fetched_news)
            self._last_fetch_meta["dedup_removed_count"] = dedup_removed

            sorted_news = sorted(
                deduped_news,
                key=lambda item: self._publish_timestamp_or_min(item.get("publish_time")),
                reverse=True,
            )

            # 缓存保存完整去重结果，便于不同时效窗口复用
            if sorted_news:
                self._save_to_cache(
                    news_list=sorted_news,
                    stock_code=normalized_stock_code,
                    source=cache_source_key,
                )

            filtered_news, age_filtered = self._apply_freshness_filter(sorted_news, max_age_hours)
            self._last_fetch_meta["age_filtered_count"] = age_filtered

            logger.info(
                "成功获取 %s 条新闻 (stock=%s, source=%s, dedup_removed=%s, age_filtered=%s, bad_ts=%s)",
                len(filtered_news),
                normalized_stock_code or "all",
                cache_source_key,
                dedup_removed,
                age_filtered,
                self._last_fetch_meta["bad_timestamp_count"],
            )
            return filtered_news[:limit_value]

        except Exception as exc:
            logger.error("获取新闻失败: %s", exc)
            return []

    def _fetch_source_rows(self, source_cfg: Dict, max_items: int) -> List[Dict]:
        target_count = max(int(max_items), 1)
        source_key = str(source_cfg.get("source_id", "")).strip()
        source_name = str(source_cfg.get("name", source_key) or source_key).strip() or source_key
        keyword = str(source_cfg.get("keyword", source_name) or source_name).strip() or source_name
        adapter = str(source_cfg.get("adapter", "json_api") or "json_api").strip().lower()
        adapter_config = source_cfg.get("adapter_config", {})
        if not isinstance(adapter_config, dict):
            adapter_config = {}

        rows: List[Dict] = []
        if adapter == "json_api":
            rows = self._fetch_json_api_rows(
                source_key=source_key,
                source_name=source_name,
                keyword=keyword,
                max_items=target_count,
                adapter_config=adapter_config,
            )
        elif adapter == "html_selector":
            rows = self._fetch_html_selector_rows(
                source_name=source_name,
                keyword=keyword,
                max_items=target_count,
                adapter_config=adapter_config,
            )
        elif adapter == "rss":
            rows = self._fetch_rss_rows(
                source_name=source_name,
                keyword=keyword,
                max_items=target_count,
                adapter_config=adapter_config,
            )
        else:
            logger.warning("未知新闻源 adapter: source=%s adapter=%s", source_key, adapter)
            return []

        deduped, _ = self._deduplicate_news_records(rows)
        return deduped[:target_count]

    def _config_int(self, payload: Dict, key: str, default: int, min_value: int, max_value: int) -> int:
        try:
            value = int(float(payload.get(key, default)))
        except Exception:
            value = int(default)
        if value < min_value:
            return min_value
        if value > max_value:
            return max_value
        return value

    def _fetch_json_api_rows(
        self,
        source_key: str,
        source_name: str,
        keyword: str,
        max_items: int,
        adapter_config: Dict,
    ) -> List[Dict]:
        target_count = max(int(max_items), 1)
        preset = str(adapter_config.get("preset", "") or "").strip().lower()

        if preset == "tencent_hot":
            rows = self._fetch_tencent_hot_rows(max_items=target_count)
            fallback_search = bool(adapter_config.get("fallback_search", True))
            if fallback_search and len(rows) < target_count:
                rows.extend(
                    self._fetch_eastmoney_search_rows(
                        keyword=keyword,
                        max_items=max(target_count - len(rows), 100),
                    )
                )
            return rows[:target_count]

        if preset == "eastmoney_search":
            rows = self._fetch_eastmoney_search_rows(
                keyword=keyword,
                max_items=max(target_count, 100),
            )
            if rows:
                return rows[:target_count]
            return self._fallback_akshare_rows(source_key=source_key, keyword=keyword, max_items=target_count)

        url = str(adapter_config.get("url", "") or "").strip()
        if not url:
            return []

        base_params = adapter_config.get("params", {})
        if not isinstance(base_params, dict):
            base_params = {}
        headers = adapter_config.get("headers", {})
        if not isinstance(headers, dict):
            headers = {}
        field_map = adapter_config.get("field_map", {})
        if not isinstance(field_map, dict):
            field_map = {}

        list_path = str(adapter_config.get("list_path", "") or "").strip()
        query_param = str(adapter_config.get("query_param", "") or "").strip()
        page_param = str(adapter_config.get("page_param", "") or "").strip()
        page_start = self._config_int(adapter_config, "page_start", 1, 0, 1000000)
        page_size_param = str(adapter_config.get("page_size_param", "") or "").strip()
        page_size = self._config_int(adapter_config, "page_size", 50, 1, 1000)
        limit_param = str(adapter_config.get("limit_param", "") or "").strip()
        max_pages = self._config_int(adapter_config, "max_pages", 1, 1, 30)
        timeout = self._config_int(adapter_config, "timeout", 20, 3, 120)
        response_format = str(adapter_config.get("response_format", "") or "").strip().lower()
        is_jsonp = bool(adapter_config.get("jsonp", False)) or response_format == "jsonp"

        rows: List[Dict] = []
        for page_index in range(max_pages):
            params = dict(base_params)
            if query_param and keyword:
                params[query_param] = keyword
            if page_param:
                params[page_param] = page_start + page_index
            if page_size_param:
                params[page_size_param] = page_size
            if limit_param and limit_param not in params:
                params[limit_param] = min(target_count, page_size)

            if is_jsonp:
                text = self._http_get_text(url=url, params=params, headers=headers, timeout=timeout)
                payload = self._parse_jsonp_payload(text)
            else:
                payload = self._http_get_json(url=url, params=params, headers=headers, timeout=timeout)

            items = self._extract_payload_items(payload, list_path)
            if not items:
                if page_index == 0:
                    continue
                break

            for item in items:
                row = self._build_row_from_json_item(
                    item=item,
                    source_name=source_name,
                    keyword=keyword,
                    field_map=field_map,
                )
                if row:
                    rows.append(row)
                if len(rows) >= target_count:
                    return rows[:target_count]

            if len(items) < page_size:
                break

        if rows:
            return rows[:target_count]

        if bool(adapter_config.get("akshare_fallback", False)):
            return self._fallback_akshare_rows(source_key=source_key, keyword=keyword, max_items=target_count)
        return []

    def _fetch_html_selector_rows(
        self,
        source_name: str,
        keyword: str,
        max_items: int,
        adapter_config: Dict,
    ) -> List[Dict]:
        if BeautifulSoup is None:
            logger.warning("BeautifulSoup 不可用，html_selector adapter 无法执行")
            return []

        url = str(adapter_config.get("url", "") or "").strip()
        if not url:
            return []

        headers = adapter_config.get("headers", {})
        if not isinstance(headers, dict):
            headers = {}
        base_params = adapter_config.get("params", {})
        if not isinstance(base_params, dict):
            base_params = {}
        field_map = adapter_config.get("field_map", {})
        if not isinstance(field_map, dict):
            field_map = {}

        item_selector = str(adapter_config.get("item_selector", "") or "").strip() or "article"
        query_param = str(adapter_config.get("query_param", "") or "").strip()
        page_param = str(adapter_config.get("page_param", "") or "").strip()
        page_start = self._config_int(adapter_config, "page_start", 1, 0, 1000000)
        max_pages = self._config_int(adapter_config, "max_pages", 1, 1, 20)
        timeout = self._config_int(adapter_config, "timeout", 20, 3, 120)
        link_base = str(adapter_config.get("link_base", "") or "").strip() or url

        rows: List[Dict] = []
        for page_index in range(max_pages):
            params = dict(base_params)
            if query_param and keyword:
                params[query_param] = keyword
            if page_param:
                params[page_param] = page_start + page_index

            html_text = self._http_get_text(url=url, params=params, headers=headers, timeout=timeout)
            if not html_text:
                if page_index == 0:
                    continue
                break

            try:
                soup = BeautifulSoup(html_text, "lxml")
            except Exception:
                soup = BeautifulSoup(html_text, "html.parser")
            nodes = soup.select(item_selector) if item_selector else []
            if not nodes:
                break

            for node in nodes:
                row = self._build_row_from_html_item(
                    node=node,
                    source_name=source_name,
                    keyword=keyword,
                    field_map=field_map,
                    link_base=link_base,
                )
                if row:
                    rows.append(row)
                if len(rows) >= max_items:
                    return rows[:max_items]
        return rows[:max_items]

    def _fetch_rss_rows(
        self,
        source_name: str,
        keyword: str,
        max_items: int,
        adapter_config: Dict,
    ) -> List[Dict]:
        if BeautifulSoup is None:
            logger.warning("BeautifulSoup 不可用，rss adapter 无法执行")
            return []

        raw_url = str(adapter_config.get("url", "") or adapter_config.get("feed_url", "")).strip()
        if not raw_url:
            return []
        feed_url = raw_url.replace("{keyword}", quote(keyword))

        headers = adapter_config.get("headers", {})
        if not isinstance(headers, dict):
            headers = {}
        timeout = self._config_int(adapter_config, "timeout", 20, 3, 120)
        item_tag = str(adapter_config.get("item_tag", "") or "").strip()

        xml_text = self._http_get_text(url=feed_url, params={}, headers=headers, timeout=timeout)
        if not xml_text:
            return []

        try:
            soup = BeautifulSoup(xml_text, "xml")
        except Exception:
            soup = BeautifulSoup(xml_text, "html.parser")
        items = soup.find_all(item_tag) if item_tag else soup.find_all(["item", "entry"])
        rows: List[Dict] = []

        for item in items:
            title = self._find_xml_field_text(item, ["title"])
            content = self._find_xml_field_text(item, ["description", "content", "summary", "content:encoded"])
            publish_time = self._find_xml_field_text(item, ["pubDate", "published", "updated", "dc:date"])
            source = self._find_xml_field_text(item, ["source", "author"]) or source_name

            link = ""
            link_node = item.find("link")
            if link_node is not None:
                href = str(link_node.get("href", "") or "").strip()
                text_link = link_node.get_text(strip=True)
                link = href or text_link
            if not link:
                link = self._find_xml_field_text(item, ["guid", "id"])
            if link and not str(link).startswith(("http://", "https://")):
                link = urljoin(feed_url, str(link))

            if not title:
                continue
            if not publish_time:
                publish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            rows.append(
                {
                    "关键词": keyword,
                    "新闻标题": title,
                    "新闻内容": content,
                    "发布时间": publish_time,
                    "来源": source,
                    "新闻链接": link,
                }
            )
            if len(rows) >= max_items:
                break

        return rows[:max_items]

    def _extract_payload_items(self, payload: Any, list_path: str) -> List[Any]:
        if payload is None:
            return []
        if list_path:
            value = self._extract_path_value(payload, list_path)
            if isinstance(value, list):
                return value
            if isinstance(value, dict):
                return [value]
            return []
        if isinstance(payload, list):
            return payload
        if isinstance(payload, dict):
            for key in ("items", "list", "data", "result", "rows"):
                candidate = payload.get(key)
                if isinstance(candidate, list):
                    return candidate
            return []
        return []

    def _extract_path_value(self, payload: Any, path: str) -> Any:
        text = str(path or "").strip()
        if not text:
            return payload
        tokens = re.findall(r"[^.\[\]]+", text)
        current = payload
        for token in tokens:
            if isinstance(current, dict):
                if token in current:
                    current = current[token]
                    continue
                return None
            if isinstance(current, list):
                try:
                    index = int(token)
                except Exception:
                    return None
                if index < 0 or index >= len(current):
                    return None
                current = current[index]
                continue
            return None
        return current

    def _extract_json_field(self, item: Any, rule: Any, fallback_paths: List[str]) -> str:
        candidates: List[Any] = []
        default_value = ""
        if isinstance(rule, str):
            candidates.append(rule)
        elif isinstance(rule, list):
            candidates.extend(rule)
        elif isinstance(rule, dict):
            if "value" in rule:
                return str(rule.get("value", "") or "").strip()
            path = rule.get("path", rule.get("field", ""))
            if path:
                candidates.append(path)
            paths = rule.get("paths", [])
            if isinstance(paths, list):
                candidates.extend(paths)
            default_value = str(rule.get("default", "") or "")

        if not candidates:
            candidates.extend(fallback_paths)

        for candidate in candidates:
            value = self._extract_path_value(item, str(candidate or ""))
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return default_value

    def _build_row_from_json_item(
        self,
        item: Any,
        source_name: str,
        keyword: str,
        field_map: Dict[str, Any],
    ) -> Optional[Dict]:
        if not isinstance(item, dict):
            return None

        title = self._extract_json_field(item, field_map.get("title"), ["title", "新闻标题"])
        if not title:
            return None
        content = self._extract_json_field(item, field_map.get("content"), ["content", "summary", "新闻内容"])
        publish_time = self._extract_json_field(
            item,
            field_map.get("publish_time"),
            ["publish_time", "pub_time", "date", "time", "发布时间"],
        )
        url = self._extract_json_field(item, field_map.get("url"), ["url", "link", "news_url", "新闻链接"])
        source = self._extract_json_field(item, field_map.get("source"), ["source", "mediaName", "来源"]) or source_name

        if not publish_time:
            publish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return {
            "关键词": keyword,
            "新闻标题": title,
            "新闻内容": content,
            "发布时间": publish_time,
            "来源": source,
            "新闻链接": url,
        }

    def _extract_html_field(self, node: Any, rule: Any) -> str:
        if rule is None:
            return ""
        if isinstance(rule, str):
            selector = rule
            attr = "text"
            default_value = ""
        elif isinstance(rule, dict):
            if "value" in rule:
                return str(rule.get("value", "") or "").strip()
            selector = str(rule.get("selector", "") or "").strip()
            attr = str(rule.get("attr", "text") or "text").strip().lower()
            default_value = str(rule.get("default", "") or "")
        else:
            return ""

        target = node
        if selector:
            target = node.select_one(selector)
        if target is None:
            return default_value

        if attr in {"text", "innertext", "content"}:
            return target.get_text(" ", strip=True)
        if attr == "html":
            return str(target)
        return str(target.get(attr, default_value) or "").strip()

    def _build_row_from_html_item(
        self,
        node: Any,
        source_name: str,
        keyword: str,
        field_map: Dict[str, Any],
        link_base: str,
    ) -> Optional[Dict]:
        title = self._extract_html_field(node, field_map.get("title", {"selector": "a", "attr": "text"}))
        if not title:
            return None
        content = self._extract_html_field(node, field_map.get("content", {"selector": "p", "attr": "text"}))
        publish_time = self._extract_html_field(
            node,
            field_map.get("publish_time", {"selector": "time", "attr": "text"}),
        )
        if not publish_time:
            publish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        source = self._extract_html_field(node, field_map.get("source", {"value": source_name})) or source_name
        url = self._extract_html_field(node, field_map.get("url", {"selector": "a", "attr": "href"}))
        if url and not url.startswith(("http://", "https://")):
            url = urljoin(link_base, url)

        return {
            "关键词": keyword,
            "新闻标题": title,
            "新闻内容": content,
            "发布时间": publish_time,
            "来源": source,
            "新闻链接": url,
        }

    def _find_xml_field_text(self, node: Any, names: List[str]) -> str:
        for name in names:
            found = node.find(name)
            if found is None:
                continue
            text = found.get_text(" ", strip=True)
            if text:
                return text
        return ""

    def _fallback_akshare_rows(self, source_key: str, keyword: str, max_items: int) -> List[Dict]:
        fallback_rows: List[Dict] = []
        try:
            df = ak.stock_news_em(symbol=keyword)
            if df is not None and not df.empty:
                for _, row in df.iterrows():
                    fallback_rows.append(dict(row))
        except Exception as exc:
            logger.warning("source=%s 回退接口抓取失败: %s", source_key, exc)
        if not fallback_rows:
            return []
        deduped_fallback, _ = self._deduplicate_news_records(fallback_rows)
        return deduped_fallback[: max(int(max_items), 1)]

    def _fetch_tencent_hot_rows(self, max_items: int = 100) -> List[Dict]:
        target_count = max(1, min(int(max_items), 200))
        url = "https://r.inews.qq.com/gw/event/hot_ranking_list"
        params = {"page_size": max(50, target_count)}
        headers = {
            "accept": "application/json, text/plain, */*",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/142.0.0.0 Safari/537.36"
            ),
            "referer": "https://new.qq.com/ch/finance/",
        }

        payload = self._http_get_json(url=url, params=params, headers=headers, timeout=20)
        if not payload:
            return []

        id_list = payload.get("idlist", []) if isinstance(payload, dict) else []
        if not id_list:
            return []

        news_list = id_list[0].get("newslist", []) if isinstance(id_list[0], dict) else []
        rows: List[Dict] = []
        for item in news_list:
            if not isinstance(item, dict):
                continue
            title = self._strip_html_markup(item.get("title", ""))
            if not title:
                continue
            publish_time = str(item.get("time", "") or "").strip()
            if not publish_time:
                ts_value = item.get("timestamp")
                try:
                    publish_time = datetime.fromtimestamp(float(ts_value)).strftime("%Y-%m-%d %H:%M:%S")
                except Exception:
                    publish_time = ""

            if not publish_time:
                continue

            article_url = str(item.get("url", "") or "").strip()
            article_id = str(item.get("id", "") or "").strip()
            if not article_url and article_id:
                article_url = f"https://view.inews.qq.com/a/{article_id}"

            rows.append(
                {
                    "关键词": "腾讯财经",
                    "新闻标题": title,
                    "新闻内容": self._strip_html_markup(item.get("abstract", "")),
                    "发布时间": publish_time,
                    "来源": str(item.get("source", "") or "").strip() or "腾讯新闻",
                    "新闻链接": article_url,
                }
            )
            if len(rows) >= target_count:
                break
        return rows

    def _fetch_eastmoney_search_rows(self, keyword: str, max_items: int = 200) -> List[Dict]:
        target_count = max(int(max_items), 1)
        page_size = min(100, max(10, target_count))
        max_pages = max(1, min(20, int(math.ceil(target_count / float(page_size))) + 1))

        url = "https://search-api-web.eastmoney.com/search/jsonp"
        headers = {
            "accept": "*/*",
            "cache-control": "no-cache",
            "pragma": "no-cache",
            "user-agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/142.0.0.0 Safari/537.36"
            ),
            "referer": f"https://so.eastmoney.com/news/s?keyword={quote(str(keyword or ''))}",
        }

        collected: List[Dict] = []
        for page_index in range(1, max_pages + 1):
            inner_param = {
                "uid": "",
                "keyword": keyword,
                "type": ["cmsArticleWebOld"],
                "client": "web",
                "clientType": "web",
                "clientVersion": "curr",
                "param": {
                    "cmsArticleWebOld": {
                        "searchScope": "default",
                        "sort": "default",
                        "pageIndex": page_index,
                        "pageSize": page_size,
                        "preTag": "<em>",
                        "postTag": "</em>",
                    }
                },
            }
            params = {
                "cb": "jQuery35101792940631092459_1764599530165",
                "param": json.dumps(inner_param, ensure_ascii=False),
                "_": "1764599530176",
            }

            text = self._http_get_text(url=url, params=params, headers=headers, timeout=20)
            if not text:
                break

            payload = self._parse_jsonp_payload(text)
            result = payload.get("result", {}) if isinstance(payload, dict) else {}
            items = result.get("cmsArticleWebOld", []) if isinstance(result, dict) else []
            if not items:
                break

            for item in items:
                if not isinstance(item, dict):
                    continue
                title = self._strip_html_markup(item.get("title", ""))
                content = self._strip_html_markup(item.get("content", ""))
                if not title:
                    continue
                collected.append(
                    {
                        "关键词": keyword,
                        "新闻标题": title,
                        "新闻内容": content,
                        "发布时间": str(item.get("date", "") or "").strip(),
                        "来源": str(item.get("mediaName", "") or "").strip() or "未知",
                        "新闻链接": self._build_eastmoney_url(item),
                    }
                )
                if len(collected) >= target_count:
                    return collected[:target_count]

            if len(items) < page_size:
                break

        return collected[:target_count]

    def _build_eastmoney_url(self, item: Dict) -> str:
        direct_url = str(item.get("url", "") or "").strip()
        if direct_url:
            return direct_url
        code = str(item.get("code", "") or "").strip()
        if code:
            return f"http://finance.eastmoney.com/a/{code}.html"
        return ""

    def _parse_jsonp_payload(self, text: str) -> Dict:
        raw = str(text or "").strip()
        if not raw:
            return {}
        if raw.startswith("{"):
            try:
                return json.loads(raw)
            except Exception:
                return {}

        left = raw.find("(")
        right = raw.rfind(")")
        if left < 0 or right <= left:
            return {}
        inner = raw[left + 1 : right]
        try:
            return json.loads(inner)
        except Exception:
            return {}

    def _http_get_text(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: int = 20,
    ) -> str:
        query_params = params if isinstance(params, dict) else {}
        request_headers = headers if isinstance(headers, dict) else {}

        if curl_requests is not None:
            try:
                resp = curl_requests.get(url, params=query_params, headers=request_headers, timeout=timeout)
                if getattr(resp, "status_code", 0) == 200:
                    return str(resp.text or "")
            except Exception:
                pass

        if std_requests is not None:
            try:
                resp = std_requests.get(url, params=query_params, headers=request_headers, timeout=timeout)
                if resp.status_code == 200:
                    return str(resp.text or "")
            except Exception:
                pass

        return ""

    def _http_get_json(
        self,
        url: str,
        params: Optional[Dict] = None,
        headers: Optional[Dict] = None,
        timeout: int = 20,
    ) -> Dict:
        text = self._http_get_text(url=url, params=params, headers=headers, timeout=timeout)
        if not text:
            return {}
        try:
            return json.loads(text)
        except Exception:
            return {}

    def _deduplicate_news_records(self, rows: List[Dict]) -> Tuple[List[Dict], int]:
        if not rows:
            return [], 0
        selected: List[Dict] = []
        seen = set()
        for row in rows:
            if not isinstance(row, dict):
                continue
            title = self._strip_html_markup(row.get("新闻标题", row.get("标题", "")))
            publish = str(row.get("发布时间", row.get("时间", "")) or "").strip()
            link = str(row.get("新闻链接", row.get("链接", "")) or "").strip()
            dedup_key = f"{title}|{publish}|{link}"
            if not title or dedup_key in seen:
                continue
            seen.add(dedup_key)
            selected.append(row)
        removed = max(len(rows) - len(selected), 0)
        return selected, removed

    def _strip_html_markup(self, text: str) -> str:
        raw = str(text or "")
        cleaned = re.sub(r"<[^>]+>", "", raw)
        cleaned = cleaned.replace("\u3000", " ").replace("\r", " ").replace("\n", " ")
        return re.sub(r"\s+", " ", cleaned).strip()

    def _parse_news_row(self, row) -> Optional[Dict]:
        """
        解析新闻行数据

        Args:
            row: 新闻数据行

        Returns:
            解析后的新闻字典
        """
        try:
            title = str(row.get("新闻标题", row.get("标题", "")) or "").strip()
            content = str(row.get("新闻内容", row.get("内容", "")) or "").strip()
            if not title:
                return None

            publish_time_raw = row.get("发布时间", row.get("时间", ""))
            publish_dt = self._parse_publish_time(publish_time_raw)
            if publish_dt is None:
                self._last_fetch_meta["bad_timestamp_count"] += 1
                return None

            normalized_url = self._normalize_news_url(row.get("新闻链接", row.get("链接", "")))
            news_id_seed = f"{title}|{publish_dt.strftime('%Y-%m-%d %H:%M:%S')}|{normalized_url}"
            news_id = hashlib.md5(news_id_seed.encode("utf-8")).hexdigest()[:12]

            combined_text = f"{title} {content}".strip()
            categories = self._classify_news(combined_text)
            sentiment = self._analyze_sentiment(combined_text, title=title)
            importance = self._calculate_importance(title, content)
            affected_sectors = self._extract_affected_sectors(combined_text)

            return {
                "news_id": news_id,
                "title": title,
                "content": content[:1000] if content else "",
                "source": str(row.get("来源", "未知")).strip() or "未知",
                "publish_time": publish_dt.strftime("%Y-%m-%d %H:%M:%S"),
                "url": normalized_url,
                "categories": categories,
                "sentiment": sentiment,
                "importance": importance,
                "affected_sectors": affected_sectors,
            }
        except Exception as exc:
            logger.warning("解析新闻失败: %s", exc)
            return None

    def _classify_news(self, text: str) -> List[str]:
        """
        分类新闻

        Args:
            text: 新闻文本

        Returns:
            分类列表
        """
        categories = []
        for category, keywords in self.category_map.items():
            for keyword in keywords:
                if keyword in text:
                    categories.append(category)
                    break
        return categories if categories else ["其他"]

    def _analyze_sentiment(self, text: str, title: str = "") -> float:
        """
        混合情感分析（关键词规则分 + 轻量语义模式分）。
        """
        raw_text = str(text or "")
        title_text = str(title or "")
        merged = f"{title_text} {raw_text}".strip()
        if not merged:
            return 0.0

        keyword_score = self._keyword_sentiment_score(merged, title_text=title_text)
        semantic_score = self._semantic_sentiment_score(merged, title_text=title_text)
        hybrid_score = 0.6 * keyword_score + 0.4 * semantic_score
        return round(float(np.clip(hybrid_score, -1.0, 1.0)), 3)

    def _keyword_sentiment_score(self, text: str, title_text: str = "") -> float:
        score = 0.0
        normalized_text = str(text or "")

        for word, weight in self.positive_words.items():
            for match in re.finditer(re.escape(word), normalized_text):
                modifier = self._sentiment_context_modifier(normalized_text, match.start())
                title_boost = 1.15 if title_text and word in title_text else 1.0
                score += weight * modifier * title_boost

        for word, weight in self.negative_words.items():
            for match in re.finditer(re.escape(word), normalized_text):
                modifier = self._sentiment_context_modifier(normalized_text, match.start())
                title_boost = 1.15 if title_text and word in title_text else 1.0
                score -= weight * modifier * title_boost

        if score == 0:
            return 0.0
        return float(np.tanh(score / 4.0))

    def _semantic_sentiment_score(self, text: str, title_text: str = "") -> float:
        merged = str(text or "")
        pos_hits = 0
        neg_hits = 0

        for pattern in self.semantic_positive_patterns:
            count = merged.count(pattern)
            if count:
                pos_hits += count
                if title_text and pattern in title_text:
                    pos_hits += 1

        for pattern in self.semantic_negative_patterns:
            count = merged.count(pattern)
            if count:
                neg_hits += count
                if title_text and pattern in title_text:
                    neg_hits += 1

        # 不确定表达降低强度，避免过度自信
        uncertain_penalty = sum(merged.count(word) for word in self.uncertain_words)
        total_hits = pos_hits + neg_hits
        if total_hits == 0:
            return 0.0

        raw = (pos_hits - neg_hits) / max(total_hits, 1)
        scale = max(0.5, 1.0 - 0.08 * uncertain_penalty)
        return float(np.clip(raw * scale, -1.0, 1.0))

    def _sentiment_context_modifier(self, text: str, hit_start: int) -> float:
        start = max(0, hit_start - 8)
        context = text[start:hit_start]

        negated = any(word in context for word in self.negation_words)
        intensified = any(word in context for word in self.intensify_words)

        modifier = -1.0 if negated else 1.0
        if intensified:
            modifier *= 1.2
        return modifier

    def _calculate_importance(self, title: str, content: str) -> float:
        """
        计算新闻重要性评分

        Args:
            title: 标题
            content: 内容

        Returns:
            重要性评分 (0-1)
        """
        score = 0.5

        important_title_words = [
            "央行",
            "证监会",
            "发改委",
            "国务院",
            "政策",
            "降准",
            "降息",
            "IPO",
            "重组",
            "并购",
            "财报",
        ]
        for word in important_title_words:
            if word in title:
                score += 0.1

        if len(content) > 200:
            score += 0.1

        if re.search(r"\d+[.%亿万元]", content):
            score += 0.05

        return min(1.0, round(score, 2))

    def _extract_affected_sectors(self, text: str) -> Dict[str, float]:
        """
        提取受影响的领域
        """
        affected = {}
        for sector, keywords in self.sector_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    impact = 0.5
                    if keyword in text[:100]:
                        impact = 0.8
                    affected[sector] = impact
                    break
        return affected

    def _filter_by_stock(self, news_list: List[Dict], stock_code: str) -> List[Dict]:
        """
        按股票代码筛选新闻。若无命中，返回空列表。
        """
        aliases = self._get_stock_names(stock_code)
        if not aliases:
            return []

        filtered: List[Dict] = []
        alias_norm_pairs = [
            (alias, self._normalize_text(alias))
            for alias in aliases
            if alias and len(str(alias).strip()) >= 2
        ]
        code_text = self._normalize_stock_code(stock_code)

        for news in news_list:
            title = str(news.get("title", "") or "")
            content = str(news.get("content", "") or "")
            title_norm = self._normalize_text(title)
            merged_norm = self._normalize_text(f"{title} {content}")

            matched = False
            relevance = 0.75

            if code_text and code_text in merged_norm:
                matched = True
                relevance = 0.85 if code_text in title_norm else 0.8
            else:
                for _, alias_norm in alias_norm_pairs:
                    if not alias_norm:
                        continue
                    if alias_norm in merged_norm:
                        matched = True
                        relevance = 0.9 if alias_norm in title_norm else 0.8
                        break

            if matched:
                enriched = dict(news)
                enriched["relevance"] = round(relevance, 3)
                filtered.append(enriched)

        return filtered

    def _get_stock_names(self, stock_code: str) -> List[str]:
        """获取股票相关名称（动态别名 + 静态兜底）。"""
        normalized_code = self._normalize_stock_code(stock_code)
        if not normalized_code:
            return []

        self._ensure_stock_alias_map()
        aliases: List[str] = []

        dynamic_aliases = self._stock_alias_map.get(normalized_code, [])
        if dynamic_aliases:
            aliases.extend(dynamic_aliases)

        static_aliases = self._static_stock_name_map.get(normalized_code, [])
        if static_aliases:
            aliases.extend(static_aliases)

        aliases.append(normalized_code)
        deduped = sorted({str(alias).strip() for alias in aliases if str(alias).strip()}, key=len, reverse=True)
        return deduped

    def _ensure_stock_alias_map(self, force: bool = False) -> None:
        now = datetime.now()
        if (
            not force
            and self._stock_alias_map
            and self._stock_alias_cache_time is not None
            and (now - self._stock_alias_cache_time) <= timedelta(hours=self._stock_alias_ttl_hours)
        ):
            return

        alias_map: Dict[str, Set[str]] = {}
        stock_list_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "cache", "stock_list.csv")
        if os.path.exists(stock_list_path):
            try:
                stock_df = pd.read_csv(stock_list_path, dtype={"code": str})
                if "code" in stock_df.columns:
                    for _, row in stock_df.iterrows():
                        code = self._normalize_stock_code(row.get("code"))
                        name = str(row.get("name", "") or "").strip()
                        if not code or not name:
                            continue

                        raw_aliases = {
                            name,
                            name.replace(" ", ""),
                            self._strip_company_suffix(name),
                            self._strip_company_suffix(name.replace(" ", "")),
                        }
                        raw_aliases = {alias for alias in raw_aliases if alias and len(alias) >= 2}
                        if not raw_aliases:
                            continue

                        alias_map.setdefault(code, set()).update(raw_aliases)
            except Exception as exc:
                logger.warning("加载股票别名缓存失败: %s", exc)

        for code, aliases in self._static_stock_name_map.items():
            norm_code = self._normalize_stock_code(code)
            if not norm_code:
                continue
            alias_map.setdefault(norm_code, set()).update(str(alias).strip() for alias in aliases if str(alias).strip())

        self._stock_alias_map = {
            code: sorted({alias for alias in aliases if len(alias) >= 2}, key=len, reverse=True)
            for code, aliases in alias_map.items()
        }
        self._stock_alias_cache_time = now

    def _strip_company_suffix(self, name: str) -> str:
        cleaned = str(name or "").strip()
        suffixes = [
            "股份有限公司",
            "集团股份有限公司",
            "有限公司",
            "股份",
            "集团",
            "控股",
            "A股",
            "A",
            "B",
            "H",
        ]
        for suffix in suffixes:
            if cleaned.endswith(suffix) and len(cleaned) > len(suffix):
                cleaned = cleaned[: -len(suffix)]
        return cleaned.strip()

    def _cache_file_path(self, stock_code: Optional[str], source: str) -> str:
        scope = self._normalize_stock_code(stock_code) or "all"
        source_key = self._normalize_source(source)
        return os.path.join(self.cache_dir, f"news_{scope}_{source_key}.json")

    def _legacy_cache_file_path(self, stock_code: Optional[str]) -> str:
        scope = self._normalize_stock_code(stock_code) or "all"
        return os.path.join(self.cache_dir, f"news_{scope}.json")

    def _load_from_cache(
        self,
        stock_code: Optional[str] = None,
        source: str = "all",
        max_age_hours: Optional[int] = 72,
    ) -> Optional[List[Dict]]:
        """从缓存加载新闻。"""
        normalized_source = self._normalize_source(source)
        candidates = [self._cache_file_path(stock_code, normalized_source)]
        # 兼容历史缓存命名
        candidates.append(self._legacy_cache_file_path(stock_code))

        for cache_file in candidates:
            if not os.path.exists(cache_file):
                continue

            try:
                file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
                if datetime.now() - file_time > timedelta(hours=self.cache_ttl_hours):
                    continue

                with open(cache_file, "r", encoding="utf-8") as file_obj:
                    payload = json.load(file_obj)
                if not isinstance(payload, list):
                    continue

                filtered, _ = self._apply_freshness_filter(payload, max_age_hours)
                if filtered:
                    return filtered
            except Exception as exc:
                logger.warning("读取缓存失败: %s", exc)

        return None

    def _save_to_cache(self, news_list: List[Dict], stock_code: Optional[str] = None, source: str = "all"):
        """保存新闻到缓存。"""
        normalized_source = self._normalize_source(source)
        cache_file = self._cache_file_path(stock_code, normalized_source)
        try:
            with open(cache_file, "w", encoding="utf-8") as file_obj:
                json.dump(news_list, file_obj, ensure_ascii=False, indent=2)

            # 保留历史命名，避免旧链路读不到缓存
            if normalized_source == "all":
                legacy_file = self._legacy_cache_file_path(stock_code)
                with open(legacy_file, "w", encoding="utf-8") as file_obj:
                    json.dump(news_list, file_obj, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.warning("保存缓存失败: %s", exc)

    def _deduplicate_news(self, news_list: List[Dict]) -> Tuple[List[Dict], int]:
        if not news_list:
            return [], 0

        selected: List[Dict] = []
        key_to_index: Dict[str, int] = {}

        for item in news_list:
            if not isinstance(item, dict):
                continue
            dedup_keys = self._build_dedup_keys(item)
            if not dedup_keys:
                continue

            existed_index = None
            for key in dedup_keys:
                if key in key_to_index:
                    existed_index = key_to_index[key]
                    break

            if existed_index is None:
                selected.append(dict(item))
                new_index = len(selected) - 1
                for key in dedup_keys:
                    key_to_index[key] = new_index
                continue

            current = selected[existed_index]
            if self._news_quality_score(item) > self._news_quality_score(current):
                selected[existed_index] = dict(item)
            for key in dedup_keys:
                key_to_index[key] = existed_index

        deduped = sorted(
            selected,
            key=lambda item: self._publish_timestamp_or_min(item.get("publish_time")),
            reverse=True,
        )
        removed = max(len(news_list) - len(deduped), 0)
        return deduped, removed

    def _build_dedup_keys(self, news: Dict) -> List[str]:
        keys = []
        news_id = str(news.get("news_id", "") or "").strip()
        if news_id:
            keys.append(f"id::{news_id}")

        title_norm = self._normalize_title(news.get("title", ""))
        if not title_norm:
            return keys

        publish_dt = self._parse_publish_time(news.get("publish_time"))
        if publish_dt is not None:
            minute_bucket = 30 * (publish_dt.minute // 30)
            bucket_ts = publish_dt.replace(minute=minute_bucket, second=0, microsecond=0)
            keys.append(f"title_bucket::{title_norm}::{bucket_ts.strftime('%Y%m%d%H%M')}")
            keys.append(f"title_day::{title_norm}::{bucket_ts.strftime('%Y%m%d')}")
        else:
            keys.append(f"title::{title_norm}")

        normalized_url = self._normalize_news_url(news.get("url", ""))
        if normalized_url:
            keys.append(f"url::{normalized_url}")

        return keys

    def _news_quality_score(self, news: Dict) -> float:
        content_len = len(str(news.get("content", "") or ""))
        importance = float(news.get("importance", 0.0) or 0.0)
        has_url = 1.0 if self._normalize_news_url(news.get("url", "")) else 0.0
        return content_len + importance * 120.0 + has_url * 15.0

    def _apply_freshness_filter(self, news_list: List[Dict], max_age_hours: Optional[int]) -> Tuple[List[Dict], int]:
        normalized_age = self._normalize_max_age_hours(max_age_hours)
        if normalized_age is None:
            valid = [item for item in news_list if self._parse_publish_time(item.get("publish_time")) is not None]
            removed = max(len(news_list) - len(valid), 0)
            return valid, removed

        cutoff = datetime.now() - timedelta(hours=normalized_age)
        filtered = []
        removed = 0
        for item in news_list:
            publish_dt = self._parse_publish_time(item.get("publish_time"))
            if publish_dt is None or publish_dt < cutoff:
                removed += 1
                continue
            filtered.append(item)
        return filtered, removed

    def _parse_publish_time(self, publish_time) -> Optional[datetime]:
        if publish_time is None:
            return None
        try:
            ts = pd.to_datetime(publish_time, errors="coerce")
        except Exception:
            return None
        if ts is None or pd.isna(ts):
            return None
        try:
            if getattr(ts, "tzinfo", None) is not None:
                ts = ts.tz_convert(None)
        except Exception:
            try:
                ts = ts.tz_localize(None)
            except Exception:
                return None
        return ts.to_pydatetime()

    def _publish_timestamp_or_min(self, publish_time) -> datetime:
        parsed = self._parse_publish_time(publish_time)
        return parsed if parsed is not None else datetime(1970, 1, 1)

    def _normalize_source(self, source: str) -> str:
        registry = get_news_source_registry()
        return registry.resolve_source_id(source)

    def _normalize_stock_code(self, stock_code: Optional[str]) -> Optional[str]:
        if stock_code is None:
            return None
        code = re.sub(r"\D", "", str(stock_code).strip())
        if not code:
            return None
        return code.zfill(6)[-6:]

    def _normalize_max_age_hours(self, max_age_hours: Optional[int]) -> Optional[int]:
        if max_age_hours is None:
            return None
        try:
            value = int(float(max_age_hours))
        except Exception:
            return 72
        if value <= 0:
            return None
        return min(value, 24 * 90)

    def _normalize_news_url(self, url: str) -> str:
        text = str(url or "").strip()
        if not text:
            return ""
        if text.startswith("//"):
            text = f"https:{text}"
        if not (text.startswith("http://") or text.startswith("https://")):
            return ""
        try:
            parsed = urlparse(text)
        except Exception:
            return ""
        if not parsed.netloc:
            return ""
        path = parsed.path or ""
        return f"{parsed.scheme.lower()}://{parsed.netloc.lower()}{path}".rstrip("/")

    def _normalize_title(self, title: str) -> str:
        return self._normalize_text(title)

    def _normalize_text(self, text: str) -> str:
        raw = str(text or "").lower()
        return re.sub(r"[^\w\u4e00-\u9fff]+", "", raw)

    def get_news_statistics(self, news_list: List[Dict]) -> Dict:
        """
        获取新闻统计信息

        Args:
            news_list: 新闻列表

        Returns:
            统计信息字典
        """
        if not news_list:
            return {"total": 0, "quality_meta": dict(self._last_fetch_meta)}

        sentiments = [float(n.get("sentiment", 0) or 0) for n in news_list]
        positive = sum(1 for value in sentiments if value > 0.2)
        negative = sum(1 for value in sentiments if value < -0.2)
        neutral = len(sentiments) - positive - negative

        sectors = {}
        for news in news_list:
            for sector, _ in (news.get("affected_sectors", {}) or {}).items():
                sectors[sector] = sectors.get(sector, 0) + 1

        sources = {}
        source_name_map = {}
        for news in news_list:
            source_id = str(news.get("source", "unknown") or "unknown").strip() or "unknown"
            source_name = str(news.get("source_name", source_id) or source_id).strip() or source_id
            sources[source_id] = sources.get(source_id, 0) + 1
            source_name_map[source_id] = source_name

        return {
            "total": len(news_list),
            "sentiment_distribution": {
                "positive": positive,
                "negative": negative,
                "neutral": neutral,
            },
            "avg_sentiment": round(np.mean(sentiments), 3) if sentiments else 0,
            "avg_importance": round(np.mean([float(n.get("importance", 0) or 0) for n in news_list]), 2),
            "sector_distribution": dict(sorted(sectors.items(), key=lambda item: item[1], reverse=True)[:10]),
            "source_distribution": dict(sorted(sources.items(), key=lambda item: item[1], reverse=True)),
            "source_name_map": source_name_map,
            "quality_meta": dict(self._last_fetch_meta),
        }


# 全局实例
news_crawler = NewsCrawler()


def get_news_crawler() -> NewsCrawler:
    """获取新闻爬虫实例"""
    return news_crawler
