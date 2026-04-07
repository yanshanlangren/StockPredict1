"""
Cross-section backtest engine for stock_code + trade_date datasets.
"""
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import DATA_DIR, RESULT_DIR

logger = logging.getLogger(__name__)

NON_FEATURE_COLUMNS = {
    "date",
    "trade_date",
    "stock_code",
    "stock_name",
    "future_ret_5d",
    "label_up_5d",
}

TECHNICAL_FEATURE_COLUMNS = [
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


@dataclass
class DateSplit:
    """Container for train/valid/test splits."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


class CrossSectionBacktestEngine:
    """Run simple cross-section top-N backtests."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        label_col: str = "label_up_5d",
        date_col: str = "trade_date",
        return_col: str = "future_ret_5d",
        random_state: int = 42,
    ):
        self.dataset_path = dataset_path or self._resolve_dataset_path()
        self.label_col = label_col
        self.date_col = date_col
        self.return_col = return_col
        self.random_state = random_state

    def run(
        self,
        model_type: str = "logistic",
        feature_set: str = "all_features",
        top_n: int = 20,
        hold_days: int = 5,
        train_ratio: float = 0.70,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        commission_rate: float = 0.0003,
        stamp_tax_rate: float = 0.001,
        slippage_rate: float = 0.0002,
    ) -> Dict:
        """Train one baseline model and backtest on test split."""
        dataset = self._load_dataset(self.dataset_path)
        split = self._split_by_trade_date(dataset, train_ratio, valid_ratio, test_ratio)
        feature_cols = self._select_feature_columns(dataset, feature_set=feature_set)

        train_df = pd.concat([split.train, split.valid], axis=0, ignore_index=True)
        if train_df.empty or split.test.empty:
            raise ValueError("Train/Test split is empty, unable to run backtest")

        model = self._build_model(model_type=model_type)
        model.fit(train_df[feature_cols], train_df[self.label_col].astype(int))

        scored_test = split.test.copy()
        scored_test["pred_prob"] = model.predict_proba(scored_test[feature_cols])[:, 1]
        scored_test["trade_day"] = scored_test[self.date_col].dt.normalize()

        holdings, equity_curve, sector_distribution = self._simulate_topn_strategy(
            scored_test=scored_test,
            top_n=top_n,
            commission_rate=commission_rate,
            stamp_tax_rate=stamp_tax_rate,
            slippage_rate=slippage_rate,
        )

        summary = self._summarize_backtest(
            holdings=holdings,
            equity_curve=equity_curve,
            hold_days=hold_days,
        )

        report = {
            "created_at": datetime.now().isoformat(),
            "dataset_path": self.dataset_path,
            "config": {
                "model_type": model_type,
                "feature_set": feature_set,
                "top_n": top_n,
                "hold_days": hold_days,
                "train_ratio": train_ratio,
                "valid_ratio": valid_ratio,
                "test_ratio": test_ratio,
                "commission_rate": commission_rate,
                "stamp_tax_rate": stamp_tax_rate,
                "slippage_rate": slippage_rate,
                "feature_count": len(feature_cols),
            },
            "split": {
                "train_samples": int(len(split.train)),
                "valid_samples": int(len(split.valid)),
                "test_samples": int(len(split.test)),
                "train_start": self._safe_iso(split.train[self.date_col].min()),
                "train_end": self._safe_iso(split.train[self.date_col].max()),
                "valid_start": self._safe_iso(split.valid[self.date_col].min()),
                "valid_end": self._safe_iso(split.valid[self.date_col].max()),
                "test_start": self._safe_iso(split.test[self.date_col].min()),
                "test_end": self._safe_iso(split.test[self.date_col].max()),
            },
            "summary": summary,
            "sector_distribution": sector_distribution,
            "equity_curve": equity_curve,
            "daily_holdings": holdings,
        }

        report_path, latest_path = self._save_report(report)
        report["report_path"] = report_path
        report["latest_path"] = latest_path
        return report

    def _resolve_dataset_path(self) -> str:
        processed_dir = os.path.join(DATA_DIR, "processed")
        parquet_path = os.path.join(processed_dir, "model_dataset.parquet")
        csv_path = os.path.join(processed_dir, "model_dataset.csv")

        if os.path.exists(parquet_path):
            return parquet_path
        if os.path.exists(csv_path):
            return csv_path
        raise FileNotFoundError("model_dataset not found. Build dataset first via src/dataset_builder.py")

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

        if dataset_path.endswith(".parquet"):
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path)

        if self.date_col not in df.columns:
            raise ValueError(f"Missing date column: {self.date_col}")
        if self.label_col not in df.columns:
            raise ValueError(f"Missing label column: {self.label_col}")
        if self.return_col not in df.columns:
            raise ValueError(f"Missing return column: {self.return_col}")

        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        df = df.dropna(subset=[self.date_col, self.label_col, self.return_col])
        df = df.sort_values([self.date_col, "stock_code"]).reset_index(drop=True)
        df[self.label_col] = df[self.label_col].astype(int)
        return df

    def _split_by_trade_date(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        valid_ratio: float,
        test_ratio: float,
    ) -> DateSplit:
        dates = sorted(df[self.date_col].dt.normalize().unique())
        total_dates = len(dates)
        if total_dates < 60:
            raise ValueError(f"Insufficient trade dates for backtest split: {total_dates}")

        train_ratio = float(max(0.5, min(train_ratio, 0.9)))
        valid_ratio = float(max(0.05, min(valid_ratio, 0.3)))
        test_ratio = float(max(0.05, min(test_ratio, 0.3)))

        if train_ratio + valid_ratio + test_ratio > 1.0:
            scale = train_ratio + valid_ratio + test_ratio
            train_ratio /= scale
            valid_ratio /= scale
            test_ratio /= scale

        train_size = int(total_dates * train_ratio)
        valid_size = int(total_dates * valid_ratio)
        test_size = total_dates - train_size - valid_size

        if train_size < 30 or valid_size < 5 or test_size < 10:
            raise ValueError(
                f"Split too small (train={train_size}, valid={valid_size}, test={test_size})"
            )

        train_dates = set(dates[:train_size])
        valid_dates = set(dates[train_size : train_size + valid_size])
        test_dates = set(dates[train_size + valid_size :])

        date_series = df[self.date_col].dt.normalize()
        train_df = df[date_series.isin(train_dates)].copy()
        valid_df = df[date_series.isin(valid_dates)].copy()
        test_df = df[date_series.isin(test_dates)].copy()
        return DateSplit(train=train_df, valid=valid_df, test=test_df)

    def _select_feature_columns(self, df: pd.DataFrame, feature_set: str) -> List[str]:
        feature_set = str(feature_set).strip().lower()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        if feature_set == "technical_only":
            columns = [col for col in TECHNICAL_FEATURE_COLUMNS if col in df.columns]
            if not columns:
                raise ValueError("No technical feature columns found in dataset")
            return columns

        if feature_set != "all_features":
            raise ValueError("feature_set must be technical_only or all_features")

        columns = [
            col
            for col in numeric_cols
            if col not in NON_FEATURE_COLUMNS
            and col != self.label_col
            and col != self.return_col
        ]
        if not columns:
            raise ValueError("No numeric features available for backtest model")
        return columns

    def _build_model(self, model_type: str) -> Pipeline:
        model_type = str(model_type).strip().lower()

        if model_type == "logistic":
            estimator = LogisticRegression(
                max_iter=2000,
                random_state=self.random_state,
                class_weight="balanced",
                solver="lbfgs",
            )
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("model", estimator),
                ]
            )

        if model_type == "random_forest":
            estimator = RandomForestClassifier(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=5,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight="balanced_subsample",
            )
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", estimator),
                ]
            )

        raise ValueError(f"Unsupported model_type: {model_type}")

    def _simulate_topn_strategy(
        self,
        scored_test: pd.DataFrame,
        top_n: int,
        commission_rate: float,
        stamp_tax_rate: float,
        slippage_rate: float,
    ) -> Tuple[List[Dict], List[Dict], Dict[str, float]]:
        top_n = int(max(1, top_n))
        holdings: List[Dict] = []
        equity_curve: List[Dict] = []
        sector_accumulator: Dict[str, float] = {}
        prev_codes: Sequence[str] = []
        nav = 1.0
        peak_nav = 1.0

        sector_columns = [col for col in scored_test.columns if col.startswith("stock_sector_")]

        for trade_day, group in scored_test.groupby("trade_day", sort=True):
            picked = group.sort_values("pred_prob", ascending=False).head(top_n).copy()
            if picked.empty:
                continue

            current_codes = picked["stock_code"].astype(str).tolist()
            raw_return = float(picked[self.return_col].mean())
            avg_probability = float(picked["pred_prob"].mean())

            turnover = self._calc_turnover(prev_codes=prev_codes, current_codes=current_codes)
            if not prev_codes:
                trade_cost = float(commission_rate + slippage_rate)
            else:
                buy_cost = commission_rate + slippage_rate
                sell_cost = commission_rate + slippage_rate + stamp_tax_rate
                trade_cost = float(turnover * (buy_cost + sell_cost))

            net_return = raw_return - trade_cost
            nav *= (1.0 + net_return)
            peak_nav = max(peak_nav, nav)
            drawdown = (nav / peak_nav) - 1.0

            holdings.append(
                {
                    "trade_date": self._safe_iso(trade_day),
                    "selected_count": int(len(current_codes)),
                    "selected_codes": current_codes,
                    "avg_probability": avg_probability,
                    "raw_future_return_5d": raw_return,
                    "turnover": turnover,
                    "trade_cost": trade_cost,
                    "net_return": net_return,
                    "avg_label": float(picked[self.label_col].mean()),
                }
            )

            equity_curve.append(
                {
                    "trade_date": self._safe_iso(trade_day),
                    "nav": float(nav),
                    "drawdown": float(drawdown),
                }
            )

            for sector_col in sector_columns:
                value = float(picked[sector_col].mean())
                sector_accumulator[sector_col] = sector_accumulator.get(sector_col, 0.0) + value

            prev_codes = current_codes

        periods = max(len(holdings), 1)
        sector_distribution = {
            key.replace("stock_sector_", ""): float(val / periods)
            for key, val in sorted(sector_accumulator.items(), key=lambda item: item[0])
        }
        return holdings, equity_curve, sector_distribution

    def _summarize_backtest(
        self,
        holdings: List[Dict],
        equity_curve: List[Dict],
        hold_days: int,
    ) -> Dict:
        if not holdings:
            return {
                "periods": 0,
                "cumulative_return": 0.0,
                "annualized_return": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "avg_period_return": 0.0,
                "avg_trade_cost": 0.0,
                "avg_turnover": 0.0,
                "sharpe_like": 0.0,
            }

        returns = np.array([item["net_return"] for item in holdings], dtype=float)
        costs = np.array([item["trade_cost"] for item in holdings], dtype=float)
        turnovers = np.array([item["turnover"] for item in holdings], dtype=float)
        cumulative_return = float((1.0 + returns).prod() - 1.0)
        win_rate = float(np.mean(returns > 0))

        periods = len(returns)
        periods_per_year = 252.0 / max(float(hold_days), 1.0)
        if periods > 0 and (1.0 + cumulative_return) > 0:
            annualized_return = float((1.0 + cumulative_return) ** (periods_per_year / periods) - 1.0)
        else:
            annualized_return = 0.0

        if len(returns) > 1 and np.std(returns) > 1e-12:
            sharpe_like = float(np.mean(returns) / np.std(returns) * np.sqrt(periods_per_year))
        else:
            sharpe_like = 0.0

        max_drawdown = 0.0
        if equity_curve:
            max_drawdown = float(min(item.get("drawdown", 0.0) for item in equity_curve))

        return {
            "periods": int(periods),
            "cumulative_return": cumulative_return,
            "annualized_return": annualized_return,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "avg_period_return": float(np.mean(returns)),
            "avg_trade_cost": float(np.mean(costs)),
            "avg_turnover": float(np.mean(turnovers)),
            "sharpe_like": sharpe_like,
        }

    def _save_report(self, report: Dict) -> Tuple[str, str]:
        os.makedirs(RESULT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(RESULT_DIR, f"cross_section_backtest_{timestamp}.json")
        latest_path = os.path.join(RESULT_DIR, "cross_section_backtest_latest.json")

        with open(report_path, "w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)
        with open(latest_path, "w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)

        logger.info("Saved backtest report: %s", report_path)
        return report_path, latest_path

    @staticmethod
    def _calc_turnover(prev_codes: Sequence[str], current_codes: Sequence[str]) -> float:
        if not current_codes:
            return 0.0
        if not prev_codes:
            return 1.0

        prev_set = set(prev_codes)
        curr_set = set(current_codes)
        entered = len(curr_set - prev_set)
        exited = len(prev_set - curr_set)
        one_side = (entered + exited) / 2.0
        return float(one_side / max(len(current_codes), 1))

    @staticmethod
    def _safe_iso(value) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        return str(value)
