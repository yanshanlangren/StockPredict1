"""
Offline evaluator with holdout + rolling time-split comparison.
"""
import json
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
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


class OfflineEvaluator:
    """Evaluate baseline models with strict temporal splits."""

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
        top_k: int = 20,
        train_ratio: float = 0.70,
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        train_days: int = 180,
        valid_days: int = 30,
        test_days: int = 30,
        step_days: int = 20,
        rolling_windows: int = 3,
    ) -> Dict:
        """Run holdout + rolling comparison between technical and full features."""
        dataset = self._load_dataset(self.dataset_path)
        feature_sets = self._build_feature_sets(dataset)

        holdout_split = self._split_by_ratio(dataset, train_ratio, valid_ratio, test_ratio)
        holdout_results = {}
        for feature_set_name, feature_cols in feature_sets.items():
            holdout_results[feature_set_name] = self._evaluate_one_split(
                model_type=model_type,
                top_k=top_k,
                train_df=holdout_split["train"],
                valid_df=holdout_split["valid"],
                test_df=holdout_split["test"],
                feature_cols=feature_cols,
            )

        rolling_splits = self._build_rolling_splits(
            dataset=dataset,
            train_days=train_days,
            valid_days=valid_days,
            test_days=test_days,
            step_days=step_days,
            rolling_windows=rolling_windows,
        )
        rolling_fallback_from_holdout = False
        if not rolling_splits:
            rolling_splits = [holdout_split]
            rolling_fallback_from_holdout = True

        rolling_results = []
        for idx, split in enumerate(rolling_splits, start=1):
            row = {
                "window_id": idx,
                "train_start": self._safe_iso(split["train"][self.date_col].min()),
                "train_end": self._safe_iso(split["train"][self.date_col].max()),
                "valid_start": self._safe_iso(split["valid"][self.date_col].min()),
                "valid_end": self._safe_iso(split["valid"][self.date_col].max()),
                "test_start": self._safe_iso(split["test"][self.date_col].min()),
                "test_end": self._safe_iso(split["test"][self.date_col].max()),
            }

            for feature_set_name, feature_cols in feature_sets.items():
                row[feature_set_name] = self._evaluate_one_split(
                    model_type=model_type,
                    top_k=top_k,
                    train_df=split["train"],
                    valid_df=split["valid"],
                    test_df=split["test"],
                    feature_cols=feature_cols,
                )

            row["uplift_full_vs_tech"] = self._calc_metric_uplift(
                base=row["technical_only"]["test_metrics"],
                target=row["all_features"]["test_metrics"],
            )
            rolling_results.append(row)

        rolling_summary = self._summarize_rolling(rolling_results)
        holdout_uplift = self._calc_metric_uplift(
            base=holdout_results["technical_only"]["test_metrics"],
            target=holdout_results["all_features"]["test_metrics"],
        )

        report = {
            "created_at": datetime.now().isoformat(),
            "dataset_path": self.dataset_path,
            "config": {
                "model_type": model_type,
                "top_k": int(top_k),
                "train_ratio": float(train_ratio),
                "valid_ratio": float(valid_ratio),
                "test_ratio": float(test_ratio),
                "train_days": int(train_days),
                "valid_days": int(valid_days),
                "test_days": int(test_days),
                "step_days": int(step_days),
                "rolling_windows": int(rolling_windows),
                "rolling_fallback_from_holdout": rolling_fallback_from_holdout,
            },
            "feature_sets": {
                key: {"feature_count": len(cols), "features": cols[:80]}
                for key, cols in feature_sets.items()
            },
            "holdout_results": holdout_results,
            "holdout_uplift_full_vs_tech": holdout_uplift,
            "rolling_results": rolling_results,
            "rolling_summary": rolling_summary,
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

        df = df.copy()
        df[self.date_col] = pd.to_datetime(df[self.date_col], errors="coerce")
        drop_cols = [self.date_col, self.label_col]
        if self.return_col in df.columns:
            drop_cols.append(self.return_col)
        df = df.dropna(subset=drop_cols)
        df[self.label_col] = df[self.label_col].astype(int)
        df = df.sort_values([self.date_col, "stock_code"]).reset_index(drop=True)

        labels = set(df[self.label_col].unique().tolist())
        if not labels.issubset({0, 1}):
            raise ValueError(f"Label column must be binary 0/1, got {sorted(labels)}")
        return df

    def _build_feature_sets(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        technical = [col for col in TECHNICAL_FEATURE_COLUMNS if col in df.columns]
        if not technical:
            raise ValueError("No technical feature columns found in dataset")

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        all_features = [
            col
            for col in numeric_cols
            if col not in NON_FEATURE_COLUMNS
            and col != self.label_col
            and col != self.return_col
        ]
        if not all_features:
            raise ValueError("No numeric features available")

        return {
            "technical_only": technical,
            "all_features": all_features,
        }

    def _split_by_ratio(
        self,
        df: pd.DataFrame,
        train_ratio: float,
        valid_ratio: float,
        test_ratio: float,
    ) -> Dict[str, pd.DataFrame]:
        dates = sorted(df[self.date_col].dt.normalize().unique())
        total_dates = len(dates)
        if total_dates < 60:
            raise ValueError(f"Insufficient trade dates for split: {total_dates}")

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
        return {
            "train": df[date_series.isin(train_dates)].copy(),
            "valid": df[date_series.isin(valid_dates)].copy(),
            "test": df[date_series.isin(test_dates)].copy(),
        }

    def _build_rolling_splits(
        self,
        dataset: pd.DataFrame,
        train_days: int,
        valid_days: int,
        test_days: int,
        step_days: int,
        rolling_windows: int,
    ) -> List[Dict[str, pd.DataFrame]]:
        dates = sorted(dataset[self.date_col].dt.normalize().unique())
        total = len(dates)

        train_days = int(max(60, train_days))
        valid_days = int(max(10, valid_days))
        test_days = int(max(10, test_days))
        step_days = int(max(5, step_days))

        if total < train_days + valid_days + test_days:
            return []

        candidates: List[Tuple[int, int, int]] = []
        start = train_days
        end = total - valid_days - test_days
        for train_end in range(start, end + 1, step_days):
            valid_end = train_end + valid_days
            test_end = valid_end + test_days
            if test_end <= total:
                candidates.append((train_end, valid_end, test_end))

        if not candidates:
            return []

        if rolling_windows > 0 and len(candidates) > rolling_windows:
            candidates = candidates[-rolling_windows:]

        splits: List[Dict[str, pd.DataFrame]] = []
        date_series = dataset[self.date_col].dt.normalize()
        for train_end, valid_end, test_end in candidates:
            train_dates = set(dates[:train_end])
            valid_dates = set(dates[train_end:valid_end])
            test_dates = set(dates[valid_end:test_end])

            split = {
                "train": dataset[date_series.isin(train_dates)].copy(),
                "valid": dataset[date_series.isin(valid_dates)].copy(),
                "test": dataset[date_series.isin(test_dates)].copy(),
            }
            if split["train"].empty or split["valid"].empty or split["test"].empty:
                continue
            splits.append(split)
        return splits

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

        if model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
            except Exception as exc:
                raise ValueError(
                    "lightgbm is not installed. Use logistic/random_forest or install lightgbm."
                ) from exc
            estimator = LGBMClassifier(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=31,
                random_state=self.random_state,
                subsample=0.9,
                colsample_bytree=0.9,
                min_child_samples=30,
            )
            return Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("model", estimator),
                ]
            )

        raise ValueError(f"Unsupported model_type: {model_type}")

    def _evaluate_one_split(
        self,
        model_type: str,
        top_k: int,
        train_df: pd.DataFrame,
        valid_df: pd.DataFrame,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict:
        model = self._build_model(model_type)
        model.fit(train_df[feature_cols], train_df[self.label_col].astype(int))

        valid_metrics = self._evaluate_metrics(model, valid_df, feature_cols, top_k=top_k)
        test_metrics = self._evaluate_metrics(model, test_df, feature_cols, top_k=top_k)

        return {
            "split": {
                "train_samples": int(len(train_df)),
                "valid_samples": int(len(valid_df)),
                "test_samples": int(len(test_df)),
            },
            "valid_metrics": valid_metrics,
            "test_metrics": test_metrics,
        }

    def _evaluate_metrics(
        self,
        model: Pipeline,
        df: pd.DataFrame,
        feature_cols: List[str],
        top_k: int,
    ) -> Dict:
        if df.empty:
            return {}

        X = df[feature_cols]
        y = df[self.label_col].astype(int).values
        probs = model.predict_proba(X)[:, 1]
        preds = (probs >= 0.5).astype(int)

        metrics = {
            "samples": int(len(df)),
            "positive_ratio": float(np.mean(y)),
            "accuracy": float(accuracy_score(y, preds)),
            "precision": float(precision_score(y, preds, zero_division=0)),
            "recall": float(recall_score(y, preds, zero_division=0)),
            "f1": float(f1_score(y, preds, zero_division=0)),
            "avg_precision": float(average_precision_score(y, probs)),
        }

        if len(np.unique(y)) > 1:
            metrics["auc"] = float(roc_auc_score(y, probs))
            metrics["logloss"] = float(log_loss(y, probs))
        else:
            metrics["auc"] = 0.5
            metrics["logloss"] = 0.0

        topk = self._calc_topk_metrics(df, probs=probs, top_k=top_k)
        metrics.update(topk)
        return metrics

    def _calc_topk_metrics(self, df: pd.DataFrame, probs: np.ndarray, top_k: int) -> Dict:
        eval_df = df.copy()
        eval_df["pred_prob"] = probs
        eval_df["trade_day"] = eval_df[self.date_col].dt.normalize()

        top_returns = []
        top_hits = []
        top_counts = []
        for _, group in eval_df.groupby("trade_day", sort=True):
            ranked = group.sort_values("pred_prob", ascending=False).head(top_k)
            if ranked.empty:
                continue
            top_counts.append(int(len(ranked)))
            top_hits.append(float(ranked[self.label_col].mean()))
            if self.return_col in ranked.columns:
                top_returns.append(float(ranked[self.return_col].mean()))

        return {
            "topk_days": int(len(top_counts)),
            "topk_avg_count": float(np.mean(top_counts)) if top_counts else 0.0,
            "topk_hit_rate": float(np.mean(top_hits)) if top_hits else 0.0,
            "topk_avg_future_return": float(np.mean(top_returns)) if top_returns else 0.0,
        }

    def _calc_metric_uplift(self, base: Dict, target: Dict) -> Dict:
        keys = ["auc", "f1", "precision", "recall", "topk_hit_rate", "topk_avg_future_return"]
        uplift = {}
        for key in keys:
            base_value = base.get(key)
            target_value = target.get(key)
            if base_value is None or target_value is None:
                continue
            uplift[key] = float(target_value - base_value)
        return uplift

    def _summarize_rolling(self, rolling_results: List[Dict]) -> Dict:
        if not rolling_results:
            return {
                "windows": 0,
                "avg_uplift_auc": 0.0,
                "avg_uplift_f1": 0.0,
                "avg_uplift_topk_return": 0.0,
                "positive_auc_windows": 0,
                "positive_topk_return_windows": 0,
            }

        auc_uplifts = []
        f1_uplifts = []
        return_uplifts = []
        for item in rolling_results:
            uplift = item.get("uplift_full_vs_tech", {})
            if "auc" in uplift:
                auc_uplifts.append(float(uplift["auc"]))
            if "f1" in uplift:
                f1_uplifts.append(float(uplift["f1"]))
            if "topk_avg_future_return" in uplift:
                return_uplifts.append(float(uplift["topk_avg_future_return"]))

        return {
            "windows": int(len(rolling_results)),
            "avg_uplift_auc": float(np.mean(auc_uplifts)) if auc_uplifts else 0.0,
            "avg_uplift_f1": float(np.mean(f1_uplifts)) if f1_uplifts else 0.0,
            "avg_uplift_topk_return": float(np.mean(return_uplifts)) if return_uplifts else 0.0,
            "positive_auc_windows": int(sum(value > 0 for value in auc_uplifts)),
            "positive_topk_return_windows": int(sum(value > 0 for value in return_uplifts)),
        }

    def _save_report(self, report: Dict) -> Tuple[str, str]:
        os.makedirs(RESULT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(RESULT_DIR, f"offline_evaluation_report_{timestamp}.json")
        latest_path = os.path.join(RESULT_DIR, "offline_evaluation_report_latest.json")

        with open(report_path, "w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)
        with open(latest_path, "w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)

        logger.info("Saved evaluation report: %s", report_path)
        return report_path, latest_path

    @staticmethod
    def _safe_iso(value) -> Optional[str]:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, (pd.Timestamp, datetime)):
            return value.isoformat()
        return str(value)
