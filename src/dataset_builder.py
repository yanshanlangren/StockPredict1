"""
Build daily market and news datasets for offline training.
"""
import argparse
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_DIR
from src.company_info_engine import get_company_info_engine
from src.data_source_manager import DataSource, DataSourceManager
from src.feature_store import FeatureStore
from src.news_crawler import get_news_crawler
from src.news_impact_analyzer import get_news_impact_analyzer
from src.relevance_graph import get_relevance_graph
from src.schedule_utils import (
    align_news_to_trade_date,
    build_trade_close_timestamp,
    normalize_trade_dates,
    parse_timestamp,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SECTOR_COLUMN_MAP = {
    "银行": "bank",
    "证券": "broker",
    "保险": "insurance",
    "地产": "real_estate",
    "汽车": "auto",
    "科技": "technology",
    "医药": "healthcare",
    "消费": "consumer",
    "能源": "energy",
    "军工": "defense",
    "基建": "infrastructure",
    "传媒": "media",
    "教育": "education",
    "农业": "agriculture",
    "交通": "transport",
    "白酒": "baijiu",
    "家电": "home_appliance",
}

STATIC_SECTOR_COLUMN_MAP = {
    "金融": "financial",
    "消费": "consumer",
    "地产": "real_estate",
    "交通": "transport",
    "制造": "manufacturing",
    "医疗": "healthcare",
    "科技": "technology",
    "能源": "energy",
}

EVENT_COLUMN_MAP = {
    "央行降准": "rrr_cut",
    "央行降息": "rate_cut",
    "IPO": "ipo",
    "财报发布": "earnings_report",
    "政策利好": "policy_positive",
    "政策利空": "policy_negative",
    "并购重组": "mna",
    "业绩预增": "earnings_positive",
    "业绩预减": "earnings_negative",
    "高管增持": "insider_buy",
    "高管减持": "insider_sell",
}

MARKET_FEATURE_COLUMNS = [
    "ret_1d",
    "ret_5d",
    "ret_20d",
    "vol_5d",
    "vol_20d",
    "rsi",
    "macd",
    "bb_position",
    "ma5_gt_ma10",
    "ma10_gt_ma20",
    "volume_ratio",
]


class DatasetBuilder:
    """Build normalized market and news feature datasets."""

    def __init__(
        self,
        processed_dir: Optional[str] = None,
        market_close_hour: int = 15,
        market_close_minute: int = 0,
        enrich_company_info: bool = False,
    ):
        self.data_manager = DataSourceManager(preferred_source=DataSource.TENCENT)
        self.news_crawler = get_news_crawler()
        self.news_analyzer = get_news_impact_analyzer()
        self.relevance_graph = get_relevance_graph()
        self.company_info_engine = get_company_info_engine()
        self.feature_store = FeatureStore(processed_dir) if processed_dir else FeatureStore()
        self.market_close_hour = market_close_hour
        self.market_close_minute = market_close_minute
        self.enrich_company_info = enrich_company_info
        self._company_info_cache: Dict[str, Dict] = {}

    def build(
        self,
        stock_limit: int = 50,
        days: int = 240,
        future_horizon: int = 5,
        label_threshold: float = 0.01,
        force_refresh: bool = False,
        refresh_news: bool = False,
        progress_callback: Optional[Callable[[int, str, Optional[Dict]], None]] = None,
    ) -> Dict:
        """Build and persist all offline datasets."""
        self._emit_progress(progress_callback, 5, "正在读取股票列表...")
        stock_list = self._get_stock_list(limit=stock_limit, force_refresh=force_refresh)
        if stock_list.empty:
            raise ValueError("No stock list available for dataset building")

        requested_codes = stock_list["code"].astype(str).str.zfill(6).tolist()
        self._emit_progress(progress_callback, 10, "正在构建相关性摘要...")
        relevance_feature_map = self._build_relevance_feature_map(requested_codes)

        market_frames: List[pd.DataFrame] = []
        news_raw_frames: List[pd.DataFrame] = []
        news_feature_frames: List[pd.DataFrame] = []
        model_frames: List[pd.DataFrame] = []
        processed_stocks: List[str] = []

        total_stocks = max(len(stock_list), 1)
        for index, stock in enumerate(stock_list.itertuples(index=False), start=1):
            stock_code = str(getattr(stock, "code", "")).zfill(6)
            stock_name = str(getattr(stock, "name", stock_code))

            market_source = self._load_market_history(
                stock_code=stock_code,
                days=days,
                force_refresh=force_refresh,
            )
            if market_source.empty or len(market_source) < 60 + future_horizon:
                logger.info("Skip %s: insufficient market history", stock_code)
                progress = 10 + int(index / total_stocks * 70)
                self._emit_progress(
                    progress_callback,
                    progress,
                    f"跳过 {stock_code}，行情数据不足",
                    {"processed": index, "total": total_stocks, "stock_code": stock_code},
                )
                continue

            market_daily = self._build_market_features(
                stock_code=stock_code,
                stock_name=stock_name,
                market_df=market_source,
                future_horizon=future_horizon,
                label_threshold=label_threshold,
            )
            trade_dates = normalize_trade_dates(market_daily["trade_date"].tolist())
            if not trade_dates:
                logger.info("Skip %s: empty trade calendar", stock_code)
                continue

            news_list = self._load_news_items(stock_code=stock_code, refresh_news=refresh_news)
            news_raw = self._build_raw_news_frame(stock_code=stock_code, news_list=news_list)
            news_daily = self._build_news_daily_features(
                stock_code=stock_code,
                trade_dates=trade_dates,
                news_list=news_list,
            )
            static_daily = self._build_static_feature_frame(
                stock_code=stock_code,
                trade_dates=trade_dates,
                relevance_features=relevance_feature_map.get(stock_code, {}),
            )

            model_dataset = market_daily.merge(
                news_daily,
                on=["stock_code", "trade_date"],
                how="left",
            ).merge(
                static_daily,
                on=["stock_code", "trade_date"],
                how="left",
            )

            model_dataset["stock_name"] = stock_name
            model_dataset = self._finalize_model_dataset(model_dataset)

            market_frames.append(market_daily)
            news_feature_frames.append(news_daily)
            model_frames.append(model_dataset)
            if not news_raw.empty:
                news_raw_frames.append(news_raw)
            processed_stocks.append(stock_code)
            progress = 10 + int(index / total_stocks * 70)
            self._emit_progress(
                progress_callback,
                progress,
                f"已处理 {stock_code} ({index}/{total_stocks})",
                {"processed": index, "total": total_stocks, "stock_code": stock_code},
            )

            logger.info(
                "Built datasets for %s: market=%s, news=%s, samples=%s",
                stock_code,
                len(market_daily),
                len(news_list),
                len(model_dataset),
            )

        market_daily_df = pd.concat(market_frames, ignore_index=True) if market_frames else pd.DataFrame()
        news_raw_df = pd.concat(news_raw_frames, ignore_index=True) if news_raw_frames else pd.DataFrame()
        news_daily_df = (
            pd.concat(news_feature_frames, ignore_index=True) if news_feature_frames else pd.DataFrame()
        )
        model_dataset_df = pd.concat(model_frames, ignore_index=True) if model_frames else pd.DataFrame()

        self._emit_progress(progress_callback, 88, "正在保存离线数据集...")
        paths = {
            "market_daily": self.feature_store.save_dataframe(market_daily_df, "market_daily"),
            "news_raw": self.feature_store.save_dataframe(news_raw_df, "news_raw"),
            "news_daily_features": self.feature_store.save_dataframe(
                news_daily_df, "news_daily_features"
            ),
            "model_dataset": self.feature_store.save_dataframe(model_dataset_df, "model_dataset"),
        }

        self._emit_progress(progress_callback, 96, "正在写入元数据...")
        metadata = {
            "version": 1,
            "created_at": pd.Timestamp.now().isoformat(),
            "config": {
                "stock_limit": stock_limit,
                "days": days,
                "future_horizon": future_horizon,
                "label_threshold": label_threshold,
                "force_refresh": force_refresh,
                "refresh_news": refresh_news,
                "market_close_hour": self.market_close_hour,
                "market_close_minute": self.market_close_minute,
            },
            "processed_stocks": processed_stocks,
            "row_counts": {
                "market_daily": int(len(market_daily_df)),
                "news_raw": int(len(news_raw_df)),
                "news_daily_features": int(len(news_daily_df)),
                "model_dataset": int(len(model_dataset_df)),
            },
            "paths": paths,
        }
        metadata_path = os.path.join(self.feature_store.base_dir, "dataset_metadata.json")
        metadata["paths"]["metadata"] = metadata_path
        self.feature_store.save_json(metadata, "dataset_metadata")
        self._emit_progress(
            progress_callback,
            100,
            "数据集构建完成",
            {"processed_stocks": len(processed_stocks), "samples": int(len(model_dataset_df))},
        )

        logger.info("Dataset build complete: %s stocks, %s samples", len(processed_stocks), len(model_dataset_df))
        return metadata

    def _get_stock_list(self, limit: int, force_refresh: bool) -> pd.DataFrame:
        """Load stock list from the data manager or local cache fallback."""
        try:
            stock_list = self.data_manager.get_stock_list(limit=limit, force_refresh=force_refresh)
        except TypeError:
            stock_list = self.data_manager.get_stock_list(limit=limit)

        if stock_list is not None and not stock_list.empty:
            stock_list = stock_list.copy()
            stock_list["code"] = stock_list["code"].astype(str).str.zfill(6)
            if "name" not in stock_list.columns:
                stock_list["name"] = stock_list["code"]
            return stock_list

        logger.warning("Falling back to local market cache for stock list")
        cache_dir = Path(DATA_DIR) / "cache"
        cache_codes = sorted(
            file_path.stem
            for file_path in cache_dir.glob("*.csv")
            if file_path.stem.isdigit() and len(file_path.stem) == 6
        )
        if limit:
            cache_codes = cache_codes[:limit]

        return pd.DataFrame({"code": cache_codes, "name": cache_codes})

    def _load_market_history(self, stock_code: str, days: int, force_refresh: bool) -> pd.DataFrame:
        """Load and normalize daily market history."""
        market_df = self.data_manager.get_stock_kline(
            stock_code,
            days=days,
            force_refresh=force_refresh,
        )
        if market_df is None or market_df.empty:
            return pd.DataFrame()

        market_df = market_df.copy()
        if "date" not in market_df.columns:
            market_df = market_df.reset_index()
            if "date" not in market_df.columns:
                market_df = market_df.rename(columns={market_df.columns[0]: "date"})

        market_df["date"] = pd.to_datetime(market_df["date"], errors="coerce").dt.normalize()
        market_df = market_df.dropna(subset=["date"]).sort_values("date").drop_duplicates("date")

        for column in ["open", "high", "low", "close", "volume", "change_pct"]:
            if column in market_df.columns:
                market_df[column] = pd.to_numeric(market_df[column], errors="coerce")

        required_columns = ["date", "open", "high", "low", "close", "volume"]
        missing_columns = [column for column in required_columns if column not in market_df.columns]
        if missing_columns:
            logger.warning("Skip %s: missing columns %s", stock_code, missing_columns)
            return pd.DataFrame()

        return market_df.tail(days).reset_index(drop=True)

    def _build_market_features(
        self,
        stock_code: str,
        stock_name: str,
        market_df: pd.DataFrame,
        future_horizon: int,
        label_threshold: float,
    ) -> pd.DataFrame:
        """Compute daily technical features and forward labels."""
        frame = market_df.copy()
        close = frame["close"].replace(0, np.nan)
        volume = frame["volume"].replace(0, np.nan)
        returns = close.pct_change()

        frame["stock_code"] = stock_code
        frame["stock_name"] = stock_name
        frame["trade_date"] = frame["date"]
        frame["ret_1d"] = returns
        frame["ret_5d"] = close.pct_change(5)
        frame["ret_20d"] = close.pct_change(20)
        frame["vol_5d"] = returns.rolling(5).std()
        frame["vol_20d"] = returns.rolling(20).std()

        delta = close.diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        frame["rsi"] = 100 - (100 / (1 + rs))
        frame["rsi"] = frame["rsi"] / 100.0

        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        frame["macd"] = (ema12 - ema26) / close

        ma20 = close.rolling(20).mean()
        std20 = close.rolling(20).std()
        upper = ma20 + (2 * std20)
        lower = ma20 - (2 * std20)
        frame["bb_position"] = (close - lower) / (upper - lower + 1e-10)

        ma5 = close.rolling(5).mean()
        ma10 = close.rolling(10).mean()
        frame["ma5_gt_ma10"] = (ma5 > ma10).astype(float)
        frame["ma10_gt_ma20"] = (ma10 > ma20).astype(float)
        frame["volume_ratio"] = volume / volume.rolling(20).mean()

        future_return = close.shift(-future_horizon) / close - 1.0
        frame["future_ret_5d"] = future_return
        frame["label_up_5d"] = np.where(
            future_return.notna(),
            (future_return > label_threshold).astype(float),
            np.nan,
        )

        return frame

    def _load_news_items(self, stock_code: str, refresh_news: bool) -> List[Dict]:
        """Load cached news items and optionally fetch current items."""
        cache_file = Path(self.news_crawler.cache_dir) / f"news_{stock_code}.json"
        news_list: List[Dict] = []

        if cache_file.exists():
            try:
                with open(cache_file, "r", encoding="utf-8") as file_obj:
                    payload = json.load(file_obj)
                    if isinstance(payload, list):
                        news_list.extend(payload)
            except Exception as exc:
                logger.warning("Failed to read cached news for %s: %s", stock_code, exc)

        if not news_list and refresh_news and self.news_crawler.is_available():
            try:
                news_list = self.news_crawler.get_news(
                    stock_code=stock_code,
                    limit=200,
                    use_cache=False,
                )
            except Exception as exc:
                logger.warning("Failed to refresh news for %s: %s", stock_code, exc)

        deduped = {}
        for news in news_list:
            news_id = str(news.get("news_id", "")).strip()
            if not news_id:
                publish_time = str(news.get("publish_time", ""))
                title = str(news.get("title", ""))
                news_id = f"{publish_time}:{title}"
            deduped[news_id] = news

        return list(deduped.values())

    def _build_raw_news_frame(self, stock_code: str, news_list: List[Dict]) -> pd.DataFrame:
        """Normalize raw news rows for persistence."""
        rows = []

        for news in news_list:
            rows.append(
                {
                    "stock_code": stock_code,
                    "news_id": str(news.get("news_id", "")),
                    "publish_time": self._safe_timestamp_string(news.get("publish_time")),
                    "source": str(news.get("source", "")),
                    "title": str(news.get("title", "")),
                    "content": str(news.get("content", "")),
                    "url": str(news.get("url", "")),
                    "categories": json.dumps(news.get("categories", []), ensure_ascii=False),
                    "sentiment": self._safe_float(news.get("sentiment", 0.0)),
                    "importance": self._safe_float(news.get("importance", 0.0)),
                    "affected_sectors": json.dumps(
                        news.get("affected_sectors", {}), ensure_ascii=False
                    ),
                }
            )

        return pd.DataFrame(rows)

    def _build_news_daily_features(
        self,
        stock_code: str,
        trade_dates: Iterable,
        news_list: List[Dict],
    ) -> pd.DataFrame:
        """Aggregate raw news into daily stock features."""
        calendar = normalize_trade_dates(trade_dates)
        grouped_news: Dict[pd.Timestamp, List[Dict]] = defaultdict(list)
        calendar_set = set(calendar)

        for news in news_list:
            trade_date = align_news_to_trade_date(
                news.get("publish_time"),
                calendar,
                market_close_hour=self.market_close_hour,
                market_close_minute=self.market_close_minute,
            )
            if trade_date is None or trade_date not in calendar_set:
                continue
            grouped_news[trade_date].append(news)

        rows = []
        for trade_date in calendar:
            rows.append(self._build_daily_news_feature_row(stock_code, trade_date, grouped_news[trade_date]))

        return pd.DataFrame(rows)

    def _build_daily_news_feature_row(
        self,
        stock_code: str,
        trade_date: pd.Timestamp,
        news_list: List[Dict],
    ) -> Dict:
        """Build one stock-day aggregate from a list of news items."""
        row = {
            "stock_code": stock_code,
            "trade_date": trade_date,
            "news_count": 0,
            "avg_sentiment": 0.0,
            "weighted_sentiment": 0.0,
            "avg_importance": 0.0,
            "max_importance": 0.0,
            "positive_ratio": 0.0,
            "negative_ratio": 0.0,
            "source_count": 0.0,
            "news_impact_total": 0.0,
            "news_impact_abs_total": 0.0,
            "news_sentiment_abs_sum": 0.0,
        }

        for sector_slug in SECTOR_COLUMN_MAP.values():
            row[f"sector_{sector_slug}"] = 0.0
        for event_slug in EVENT_COLUMN_MAP.values():
            row[f"event_{event_slug}_count"] = 0.0

        if not news_list:
            return row

        close_ts = build_trade_close_timestamp(
            trade_date,
            market_close_hour=self.market_close_hour,
            market_close_minute=self.market_close_minute,
        )

        sentiments = np.array([self._safe_float(news.get("sentiment", 0.0)) for news in news_list])
        importances = np.array([self._safe_float(news.get("importance", 0.0)) for news in news_list])
        weights = np.array([self._news_decay_weight(news, close_ts) for news in news_list])

        positive_mask = sentiments > 0.2
        negative_mask = sentiments < -0.2

        row["news_count"] = int(len(news_list))
        row["avg_sentiment"] = self._safe_float(np.mean(sentiments))
        row["weighted_sentiment"] = self._weighted_average(sentiments, weights)
        row["avg_importance"] = self._safe_float(np.mean(importances))
        row["max_importance"] = self._safe_float(np.max(importances))
        row["positive_ratio"] = self._safe_float(positive_mask.mean())
        row["negative_ratio"] = self._safe_float(negative_mask.mean())
        row["source_count"] = float(
            len({str(news.get("source", "")).strip() for news in news_list if str(news.get("source", "")).strip()})
        )
        row["news_sentiment_abs_sum"] = self._safe_float(np.abs(sentiments).sum())

        sector_scores = defaultdict(float)
        for index, news in enumerate(news_list):
            sentiment = sentiments[index]
            importance = importances[index]
            decay = weights[index]

            affected_sectors = news.get("affected_sectors", {}) or {}
            if isinstance(affected_sectors, dict):
                for sector_name, sector_weight in affected_sectors.items():
                    sector_scores[str(sector_name)] += (
                        self._safe_float(sector_weight) * sentiment * importance * decay
                    )

            news_text = f"{news.get('title', '')} {news.get('content', '')}"
            for event_name, event_slug in EVENT_COLUMN_MAP.items():
                if event_name in news_text:
                    row[f"event_{event_slug}_count"] += 1.0

        for sector_name, sector_slug in SECTOR_COLUMN_MAP.items():
            row[f"sector_{sector_slug}"] = self._safe_float(sector_scores.get(sector_name, 0.0))

        row["news_impact_total"] = self._safe_float(sum(sector_scores.values()))
        row["news_impact_abs_total"] = self._safe_float(sum(abs(value) for value in sector_scores.values()))
        return row

    def _build_static_feature_frame(
        self,
        stock_code: str,
        trade_dates: Iterable,
        relevance_features: Dict[str, float],
    ) -> pd.DataFrame:
        """Create per-day static features for one stock."""
        calendar = normalize_trade_dates(trade_dates)
        company_info = self._get_company_info(stock_code)
        concepts = company_info.get("concepts", [])
        if not isinstance(concepts, list):
            concepts = []

        listing_date = parse_timestamp(company_info.get("listing_date"))
        listing_date = listing_date.normalize() if listing_date is not None else None
        sector_text = " ".join(
            filter(
                None,
                [
                    str(company_info.get("sector", "")).strip(),
                    str(company_info.get("industry", "")).strip(),
                ],
            )
        )

        rows = []
        for trade_date in calendar:
            row = {
                "stock_code": stock_code,
                "trade_date": trade_date,
                "static_concept_count": float(len(concepts)),
                "static_listing_days": float((trade_date - listing_date).days)
                if listing_date is not None
                else np.nan,
                "relevance_mean": self._safe_float(relevance_features.get("relevance_mean", 0.0)),
                "relevance_max": self._safe_float(relevance_features.get("relevance_max", 0.0)),
                "relevance_std": self._safe_float(relevance_features.get("relevance_std", 0.0)),
                "relevance_strong_ratio": self._safe_float(
                    relevance_features.get("relevance_strong_ratio", 0.0)
                ),
            }

            for sector_name, sector_slug in STATIC_SECTOR_COLUMN_MAP.items():
                row[f"stock_sector_{sector_slug}"] = 1.0 if sector_name and sector_name in sector_text else 0.0

            rows.append(row)

        return pd.DataFrame(rows)

    def _build_relevance_feature_map(self, stock_codes: List[str]) -> Dict[str, Dict[str, float]]:
        """Summarize graph correlation features for each stock."""
        if not stock_codes:
            return {}

        matrix_data = self.relevance_graph.get_relevance_matrix(stock_codes=stock_codes)
        matrix = np.array(matrix_data["matrix"], dtype=float)
        feature_map: Dict[str, Dict[str, float]] = {}

        for index, stock_code in enumerate(matrix_data["stock_codes"]):
            row = matrix[index]
            if len(row) > 1:
                peers = np.delete(row, index)
            else:
                peers = np.array([], dtype=float)

            if peers.size == 0:
                feature_map[stock_code] = {
                    "relevance_mean": 0.0,
                    "relevance_max": 0.0,
                    "relevance_std": 0.0,
                    "relevance_strong_ratio": 0.0,
                }
                continue

            feature_map[stock_code] = {
                "relevance_mean": self._safe_float(np.mean(peers)),
                "relevance_max": self._safe_float(np.max(peers)),
                "relevance_std": self._safe_float(np.std(peers)),
                "relevance_strong_ratio": self._safe_float(np.mean(peers > 0.5)),
            }

        return feature_map

    def _get_company_info(self, stock_code: str) -> Dict:
        """Return local company metadata without requiring remote calls by default."""
        if stock_code in self._company_info_cache:
            return self._company_info_cache[stock_code]

        info = {}
        predefined = getattr(self.company_info_engine, "_predefined_companies", {})
        if stock_code in predefined:
            info = dict(predefined[stock_code])
        elif self.enrich_company_info:
            try:
                info = self.company_info_engine.get_company_info(stock_code, use_cache=True)
            except Exception as exc:
                logger.warning("Failed to enrich company info for %s: %s", stock_code, exc)
                info = {}

        self._company_info_cache[stock_code] = info
        return info

    def _finalize_model_dataset(self, model_dataset: pd.DataFrame) -> pd.DataFrame:
        """Fill sparse features and drop rows that are not trainable."""
        feature_columns = list(MARKET_FEATURE_COLUMNS)
        feature_columns.extend(
            column
            for column in model_dataset.columns
            if column.startswith("sector_")
            or column.startswith("event_")
            or column.startswith("static_")
            or column.startswith("relevance_")
            or column in {
                "news_count",
                "avg_sentiment",
                "weighted_sentiment",
                "avg_importance",
                "max_importance",
                "positive_ratio",
                "negative_ratio",
                "source_count",
                "news_impact_total",
                "news_impact_abs_total",
                "news_sentiment_abs_sum",
            }
        )

        sparse_columns = [
            column
            for column in feature_columns
            if column not in MARKET_FEATURE_COLUMNS
        ]
        if sparse_columns:
            model_dataset[sparse_columns] = model_dataset[sparse_columns].fillna(0.0)

        model_dataset = model_dataset.dropna(
            subset=MARKET_FEATURE_COLUMNS + ["future_ret_5d", "label_up_5d"]
        ).copy()
        model_dataset["label_up_5d"] = model_dataset["label_up_5d"].astype(int)
        model_dataset = model_dataset.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
        return model_dataset

    def _news_decay_weight(self, news: Dict, close_ts: Optional[pd.Timestamp]) -> float:
        """Compute decay relative to the sample close time."""
        if close_ts is None:
            return 1.0

        publish_ts = parse_timestamp(news.get("publish_time"))
        if publish_ts is None:
            return 0.5

        hours_to_close = max((close_ts - publish_ts).total_seconds() / 3600.0, 0.0)
        half_life = max(float(getattr(self.news_analyzer, "decay_half_life", 24.0)), 1.0)
        decay = math.exp(-hours_to_close * math.log(2) / half_life)
        return max(0.01, min(1.0, decay))

    @staticmethod
    def _weighted_average(values: np.ndarray, weights: np.ndarray) -> float:
        """Compute a stable weighted average."""
        if values.size == 0:
            return 0.0
        if weights.size == 0 or np.allclose(weights.sum(), 0.0):
            return DatasetBuilder._safe_float(np.mean(values))
        return DatasetBuilder._safe_float(np.average(values, weights=weights))

    @staticmethod
    def _safe_float(value, default: float = 0.0) -> float:
        """Convert to float while guarding against NaN and inf."""
        try:
            result = float(value)
        except Exception:
            return default

        if np.isnan(result) or np.isinf(result):
            return default
        return result

    @staticmethod
    def _safe_timestamp_string(value) -> str:
        """Normalize timestamps before persistence."""
        timestamp = parse_timestamp(value)
        if timestamp is None:
            return ""
        return timestamp.strftime("%Y-%m-%d %H:%M:%S")

    @staticmethod
    def _emit_progress(
        callback: Optional[Callable[[int, str, Optional[Dict]], None]],
        progress: int,
        message: str,
        extra: Optional[Dict] = None,
    ) -> None:
        """Safely emit progress updates for UI or background jobs."""
        if callback is None:
            return

        callback(int(progress), message, extra or {})


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Build offline market/news datasets")
    parser.add_argument("--stocks", type=int, default=50, help="Number of stocks to process")
    parser.add_argument("--days", type=int, default=240, help="Lookback market days per stock")
    parser.add_argument("--horizon", type=int, default=5, help="Forward label horizon")
    parser.add_argument(
        "--label-threshold",
        type=float,
        default=0.01,
        help="Forward return threshold for positive labels",
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Bypass local market cache when loading price history",
    )
    parser.add_argument(
        "--refresh-news",
        action="store_true",
        help="Fetch live news if no local stock news cache exists",
    )
    parser.add_argument(
        "--enrich-company-info",
        action="store_true",
        help="Allow remote company info enrichment when local metadata is missing",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    builder = DatasetBuilder(enrich_company_info=args.enrich_company_info)
    metadata = builder.build(
        stock_limit=args.stocks,
        days=args.days,
        future_horizon=args.horizon,
        label_threshold=args.label_threshold,
        force_refresh=args.force_refresh,
        refresh_news=args.refresh_news,
    )

    logger.info("Output files:")
    for name, path in metadata["paths"].items():
        logger.info("  %s -> %s", name, path)


if __name__ == "__main__":
    main()
