#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练多模态预测模型（Phase 4 可实施版）。

实现原则：
1. 优先依赖标准化日频数据集（model_dataset/news_raw/market_daily）
2. 按 stock_code + trade_date 口径构样本
3. 不再使用“当前新闻覆盖历史窗口”或合成新闻作为正式训练数据
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# 添加项目根目录与 src 目录到路径
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from config import PROCESSED_DATA_DIR
from src.multimodal_model import get_multimodal_predictor
from src.schedule_utils import align_news_to_trade_date, normalize_trade_dates

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


TECH_COLUMNS = [
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

NEWS_BASE_COLUMNS = [
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
]


def _resolve_dataset_file(path: Optional[str], name: str) -> str:
    """解析数据集文件路径（优先指定路径，其次 parquet，再次 csv）。"""
    if path:
        if not os.path.exists(path):
            raise FileNotFoundError(f"指定文件不存在: {path}")
        return path

    parquet_path = os.path.join(PROCESSED_DATA_DIR, f"{name}.parquet")
    csv_path = os.path.join(PROCESSED_DATA_DIR, f"{name}.csv")

    if os.path.exists(parquet_path):
        return parquet_path
    if os.path.exists(csv_path):
        return csv_path

    raise FileNotFoundError(
        f"未找到 {name} 数据集。请先构建数据集（/api/dataset/build 或 src/dataset_builder.py）"
    )


def _load_dataframe(path: str) -> pd.DataFrame:
    """按扩展名读取 DataFrame。"""
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _normalize_code(value) -> str:
    return str(value).strip().zfill(6)


def _detect_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str], List[str], List[str]]:
    """自动识别多模态输入字段。"""
    numeric_cols = set(df.select_dtypes(include=[np.number]).columns.tolist())

    tech_cols = [col for col in TECH_COLUMNS if col in numeric_cols]

    news_cols: List[str] = []
    for col in NEWS_BASE_COLUMNS:
        if col in numeric_cols:
            news_cols.append(col)

    event_cols = sorted(col for col in numeric_cols if col.startswith("event_"))
    news_cols.extend(event_cols)

    sector_cols = sorted(col for col in numeric_cols if col.startswith("sector_"))

    relevance_cols = sorted(
        col
        for col in numeric_cols
        if col.startswith("relevance_")
        or col.startswith("static_")
        or col.startswith("stock_sector_")
    )

    return news_cols, sector_cols, tech_cols, relevance_cols


def _normalize_trade_date_column(df: pd.DataFrame, column: str = "trade_date") -> pd.DataFrame:
    frame = df.copy()
    if column not in frame.columns:
        raise ValueError(f"缺少字段: {column}")

    frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame = frame.dropna(subset=[column])
    frame[column] = frame[column].dt.normalize()
    return frame


def _restrict_dataset_scope(df: pd.DataFrame, n_stocks: int, days: int) -> pd.DataFrame:
    """兼容旧参数：限制股票数和日期跨度。"""
    frame = df.copy()

    frame["stock_code"] = frame["stock_code"].map(_normalize_code)

    if n_stocks > 0:
        stock_counts = (
            frame.groupby("stock_code", sort=False)
            .size()
            .sort_values(ascending=False)
        )
        selected_codes = stock_counts.head(n_stocks).index.tolist()
        frame = frame[frame["stock_code"].isin(selected_codes)]

    if days > 0:
        all_dates = sorted(frame["trade_date"].dropna().unique())
        if len(all_dates) > days:
            keep_dates = set(all_dates[-days:])
            frame = frame[frame["trade_date"].isin(keep_dates)]

    frame = frame.sort_values(["trade_date", "stock_code"]).reset_index(drop=True)
    return frame


def _build_trade_calendar_map(market_df: pd.DataFrame) -> Dict[str, List[pd.Timestamp]]:
    """构建每只股票的交易日历。"""
    frame = _normalize_trade_date_column(market_df, "trade_date")
    if "stock_code" not in frame.columns:
        raise ValueError("market_daily 缺少 stock_code 字段")

    frame["stock_code"] = frame["stock_code"].map(_normalize_code)

    calendar_map: Dict[str, List[pd.Timestamp]] = {}
    for stock_code, group in frame.groupby("stock_code", sort=False):
        calendar_map[stock_code] = normalize_trade_dates(group["trade_date"].tolist())
    return calendar_map


def _safe_json_load_list(value) -> List:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = json.loads(text)
    except Exception:
        return []
    return parsed if isinstance(parsed, list) else []


def _build_news_lookup(news_raw_df: pd.DataFrame, calendar_map: Dict[str, List[pd.Timestamp]]) -> Dict[Tuple[str, str], List[Dict]]:
    """
    将 news_raw 对齐到 trade_date，返回 (stock_code, YYYY-MM-DD) -> news_list。
    """
    required = {"stock_code", "publish_time", "title", "content"}
    missing = [col for col in required if col not in news_raw_df.columns]
    if missing:
        raise ValueError(f"news_raw 缺少字段: {', '.join(missing)}")

    frame = news_raw_df.copy()
    frame["stock_code"] = frame["stock_code"].map(_normalize_code)

    lookup: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)

    for row in frame.itertuples(index=False):
        stock_code = _normalize_code(getattr(row, "stock_code", ""))
        trade_calendar = calendar_map.get(stock_code)
        if not trade_calendar:
            continue

        publish_time = getattr(row, "publish_time", None)
        trade_date = align_news_to_trade_date(publish_time, trade_calendar, market_close_hour=15, market_close_minute=0)
        if trade_date is None:
            continue

        key = (stock_code, trade_date.strftime("%Y-%m-%d"))
        lookup[key].append(
            {
                "title": str(getattr(row, "title", "")),
                "content": str(getattr(row, "content", "")),
                "publish_time": str(getattr(row, "publish_time", "")),
                "sentiment": float(getattr(row, "sentiment", 0.0) or 0.0),
                "importance": float(getattr(row, "importance", 0.0) or 0.0),
                "categories": _safe_json_load_list(getattr(row, "categories", [])),
            }
        )

    return lookup


def _extract_feature_array(df: pd.DataFrame, cols: Iterable[str]) -> np.ndarray:
    selected = list(cols)
    if not selected:
        return np.zeros((len(df), 0), dtype=float)

    frame = df[selected].copy()
    for col in selected:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return frame.to_numpy(dtype=float)


def prepare_training_data_from_dataset(
    model_dataset_path: str,
    market_daily_path: str,
    news_raw_path: str,
    n_stocks: int = 50,
    days: int = 200,
) -> Dict:
    """
    基于标准数据集构建多模态训练数据。
    """
    logger.info("加载标准数据集...")
    model_df = _normalize_trade_date_column(_load_dataframe(model_dataset_path), "trade_date")
    market_df = _normalize_trade_date_column(_load_dataframe(market_daily_path), "trade_date")
    news_raw_df = _load_dataframe(news_raw_path)

    required_columns = {"stock_code", "trade_date", "label_up_5d"}
    missing = [col for col in required_columns if col not in model_df.columns]
    if missing:
        raise ValueError(f"model_dataset 缺少必要字段: {', '.join(missing)}")

    model_df["stock_code"] = model_df["stock_code"].map(_normalize_code)
    model_df = _restrict_dataset_scope(model_df, n_stocks=n_stocks, days=days)

    # 保证标签二值
    model_df["label_up_5d"] = pd.to_numeric(model_df["label_up_5d"], errors="coerce")
    model_df = model_df.dropna(subset=["label_up_5d"])
    model_df["label_up_5d"] = model_df["label_up_5d"].astype(int)

    if model_df.empty:
        raise ValueError("过滤后 model_dataset 为空，无法训练")

    logger.info("解析特征字段...")
    news_cols, sector_cols, tech_cols, relevance_cols = _detect_feature_columns(model_df)

    if not tech_cols:
        raise ValueError("model_dataset 缺少技术面特征，无法训练")

    logger.info(
        "特征分组: news=%s, sector=%s, tech=%s, relevance=%s",
        len(news_cols),
        len(sector_cols),
        len(tech_cols),
        len(relevance_cols),
    )

    # 构建文本向量依赖的 news_lookup
    logger.info("构建新闻文本对齐映射...")
    calendar_map = _build_trade_calendar_map(market_df)
    news_lookup = _build_news_lookup(news_raw_df, calendar_map)

    predictor = get_multimodal_predictor()

    # 先把每个 unique key 的文本向量编码出来，避免重复计算
    unique_keys = (
        model_df[["stock_code", "trade_date"]]
        .drop_duplicates()
        .assign(trade_date=lambda x: x["trade_date"].dt.strftime("%Y-%m-%d"))
        .itertuples(index=False)
    )

    text_feature_cache: Dict[Tuple[str, str], np.ndarray] = {}
    for item in unique_keys:
        key = (item.stock_code, item.trade_date)
        text_feature_cache[key] = predictor.encode_news_text_features(news_lookup.get(key, [])).reshape(-1)

    sample_keys = model_df.assign(
        trade_date_key=model_df["trade_date"].dt.strftime("%Y-%m-%d")
    )[["stock_code", "trade_date_key"]].itertuples(index=False)

    text_features = np.vstack(
        [text_feature_cache[(item.stock_code, item.trade_date_key)] for item in sample_keys]
    )

    training_data = {
        "news_features": _extract_feature_array(model_df, news_cols),
        "sector_features": _extract_feature_array(model_df, sector_cols),
        "tech_features": _extract_feature_array(model_df, tech_cols),
        "relevance_features": _extract_feature_array(model_df, relevance_cols),
        "text_features": text_features,
        "labels": model_df["label_up_5d"].to_numpy(dtype=int),
    }

    positive = int(training_data["labels"].sum())
    negative = int(len(training_data["labels"]) - positive)
    logger.info("训练样本统计: 总样本=%s, 正样本=%s, 负样本=%s", len(training_data["labels"]), positive, negative)

    return training_data


def train_model(training_data: dict, epochs: int = 50, model_tier: str = "heavy") -> dict:
    """训练模型"""
    predictor = get_multimodal_predictor()

    logger.info("开始训练多模态模型（model_tier=%s）...", model_tier)
    result = predictor.train(
        training_data,
        epochs=epochs,
        batch_size=32,
        model_tier=model_tier,
    )

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练多模态预测模型")
    parser.add_argument("--stocks", type=int, default=50, help="训练使用的股票数量（按样本覆盖优先排序）")
    parser.add_argument("--days", type=int, default=200, help="训练使用的最近交易日数量")
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数")
    parser.add_argument(
        "--model-tier",
        type=str,
        default="heavy",
        choices=["heavy"],
        help="模型复杂度固定为 heavy",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="model_dataset 路径（可选，默认 data/processed/model_dataset）",
    )
    parser.add_argument(
        "--market-path",
        type=str,
        default=None,
        help="market_daily 路径（可选，默认 data/processed/market_daily）",
    )
    parser.add_argument(
        "--news-raw-path",
        type=str,
        default=None,
        help="news_raw 路径（可选，默认 data/processed/news_raw）",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logger.info("=" * 50)
    logger.info("多模态预测模型训练")
    logger.info("=" * 50)

    model_dataset_path = _resolve_dataset_file(args.dataset_path, "model_dataset")
    market_daily_path = _resolve_dataset_file(args.market_path, "market_daily")
    news_raw_path = _resolve_dataset_file(args.news_raw_path, "news_raw")

    logger.info(
        "参数: stocks=%s, days=%s, epochs=%s, model_tier=%s",
        args.stocks,
        args.days,
        args.epochs,
        "heavy",
    )
    logger.info("输入数据: model_dataset=%s", model_dataset_path)
    logger.info("输入数据: market_daily=%s", market_daily_path)
    logger.info("输入数据: news_raw=%s", news_raw_path)

    training_data = prepare_training_data_from_dataset(
        model_dataset_path=model_dataset_path,
        market_daily_path=market_daily_path,
        news_raw_path=news_raw_path,
        n_stocks=max(1, int(args.stocks)),
        days=max(30, int(args.days)),
    )

    result = train_model(training_data, epochs=args.epochs, model_tier="heavy")

    if result.get("success"):
        logger.info("=" * 50)
        logger.info("✓ 训练完成！")
        logger.info("模型保存至: %s", result.get("model_path"))
        if result.get("message"):
            logger.info("训练说明: %s", result.get("message"))
        logger.info("最终指标: %s", result.get("metadata", {}).get("final_metrics"))
        logger.info("=" * 50)
    else:
        logger.error("训练失败: %s", result.get("message"))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
