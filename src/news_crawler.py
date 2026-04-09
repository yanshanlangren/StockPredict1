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
from typing import Optional, List, Dict, Tuple, Set
from datetime import datetime, timedelta
from urllib.parse import urlparse
import re

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
    logger.info("✓ akshare库已加载，新闻爬取功能可用")
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.error("✗ akshare库未安装，新闻爬取功能不可用")


class NewsCrawler:
    """财经新闻爬取引擎"""

    SOURCE_SYMBOL_MAP = {
        "eastmoney": "财经新闻",
        "sina": "新浪财经",
        "tencent": "腾讯财经",
    }

    SOURCE_LABEL_MAP = {
        "eastmoney": "东方财富",
        "sina": "新浪财经",
        "tencent": "腾讯财经",
        "all": "全部来源",
        "unknown": "未知",
    }

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
            source: 新闻源 ('eastmoney', 'sina', 'tencent', 'all')
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
                source=normalized_source,
                max_age_hours=max_age_hours,
            )
            if cached:
                logger.info(
                    "从缓存加载 %s 条新闻 (stock=%s, source=%s)",
                    len(cached),
                    normalized_stock_code or "all",
                    normalized_source,
                )
                return cached[:limit_value]

        try:
            fetched_news: List[Dict] = []
            source_keys = (
                ["eastmoney", "sina", "tencent"]
                if normalized_source == "all"
                else [normalized_source]
            )

            for source_key in source_keys:
                symbol = self.SOURCE_SYMBOL_MAP.get(source_key)
                if not symbol:
                    continue

                try:
                    df = ak.stock_news_em(symbol=symbol)
                    if df is None or df.empty:
                        continue

                    for _, row in df.iterrows():
                        news_item = self._parse_news_row(row)
                        if not news_item:
                            continue
                        news_item["source"] = self.SOURCE_LABEL_MAP.get(source_key, "未知")
                        fetched_news.append(news_item)
                except Exception as exc:
                    logger.warning("获取 %s 新闻失败: %s", source_key, exc)

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
                    source=normalized_source,
                )

            filtered_news, age_filtered = self._apply_freshness_filter(sorted_news, max_age_hours)
            self._last_fetch_meta["age_filtered_count"] = age_filtered

            logger.info(
                "成功获取 %s 条新闻 (stock=%s, source=%s, dedup_removed=%s, age_filtered=%s, bad_ts=%s)",
                len(filtered_news),
                normalized_stock_code or "all",
                normalized_source,
                dedup_removed,
                age_filtered,
                self._last_fetch_meta["bad_timestamp_count"],
            )
            return filtered_news[:limit_value]

        except Exception as exc:
            logger.error("获取新闻失败: %s", exc)
            return []

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
        normalized = str(source or "").strip().lower()
        if normalized in {"eastmoney", "sina", "tencent", "all"}:
            return normalized
        if normalized in {"东方财富", "财经新闻"}:
            return "eastmoney"
        if normalized in {"新浪", "新浪财经"}:
            return "sina"
        if normalized in {"腾讯", "腾讯财经"}:
            return "tencent"
        return "eastmoney"

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
        for news in news_list:
            source = str(news.get("source", "未知") or "未知")
            sources[source] = sources.get(source, 0) + 1

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
            "quality_meta": dict(self._last_fetch_meta),
        }


# 全局实例
news_crawler = NewsCrawler()


def get_news_crawler() -> NewsCrawler:
    """获取新闻爬虫实例"""
    return news_crawler
