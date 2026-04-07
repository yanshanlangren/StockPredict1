"""
Structured baseline trainer for stock_code + trade_date datasets.
"""
import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import joblib
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

from config import DATA_DIR, MODEL_DIR, RESULT_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


NON_FEATURE_COLUMNS = {
    "date",
    "trade_date",
    "stock_code",
    "stock_name",
    "future_ret_5d",
    "label_up_5d",
}


@dataclass
class DatasetSplit:
    """Container for train/valid/test splits."""

    train: pd.DataFrame
    valid: pd.DataFrame
    test: pd.DataFrame


class BaselineModelTrainer:
    """Train and evaluate baseline models on structured daily datasets."""

    def __init__(
        self,
        dataset_path: Optional[str] = None,
        label_col: str = "label_up_5d",
        date_col: str = "trade_date",
        return_col: str = "future_ret_5d",
        valid_ratio: float = 0.15,
        test_ratio: float = 0.15,
        random_state: int = 42,
    ):
        self.dataset_path = dataset_path or self._resolve_dataset_path()
        self.label_col = label_col
        self.date_col = date_col
        self.return_col = return_col
        self.valid_ratio = valid_ratio
        self.test_ratio = test_ratio
        self.random_state = random_state

    def run(self, model_type: str = "logistic", top_k: int = 20) -> Dict:
        """Run baseline training and return a report."""
        dataset = self._load_dataset(self.dataset_path)
        feature_cols = self._select_feature_columns(dataset)
        split = self._split_by_trade_date(dataset)

        logger.info(
            "Dataset split: train=%s, valid=%s, test=%s, features=%s",
            len(split.train),
            len(split.valid),
            len(split.test),
            len(feature_cols),
        )

        train_model = self._build_model(model_type=model_type)

        X_train = split.train[feature_cols]
        y_train = split.train[self.label_col].astype(int)
        train_model.fit(X_train, y_train)

        valid_metrics = self._evaluate_split(train_model, split.valid, feature_cols, top_k=top_k)
        test_metrics = self._evaluate_split(train_model, split.test, feature_cols, top_k=top_k)

        combined_train = pd.concat([split.train, split.valid], axis=0, ignore_index=True)
        final_model = self._build_model(model_type=model_type)
        final_model.fit(combined_train[feature_cols], combined_train[self.label_col].astype(int))

        model_info = self._save_model(final_model, feature_cols, model_type)
        top_features = self._extract_top_features(final_model, feature_cols, top_n=20)
        report = self._save_report(
            model_type=model_type,
            feature_cols=feature_cols,
            split=split,
            valid_metrics=valid_metrics,
            test_metrics=test_metrics,
            model_info=model_info,
            top_features=top_features,
            top_k=top_k,
        )

        logger.info(
            "Baseline training finished. Test AUC=%.4f, Test F1=%.4f",
            report["metrics"]["test"].get("auc", 0.0),
            report["metrics"]["test"].get("f1", 0.0),
        )
        return report

    def _resolve_dataset_path(self) -> str:
        """Resolve default dataset path from data/processed."""
        processed_dir = os.path.join(DATA_DIR, "processed")
        parquet_path = os.path.join(processed_dir, "model_dataset.parquet")
        csv_path = os.path.join(processed_dir, "model_dataset.csv")

        if os.path.exists(parquet_path):
            return parquet_path
        if os.path.exists(csv_path):
            return csv_path

        raise FileNotFoundError(
            "model_dataset not found. Build dataset first via src/dataset_builder.py"
        )

    def _load_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load dataset and enforce basic schema."""
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
        df = df.dropna(subset=[self.date_col, self.label_col]).sort_values([self.date_col, "stock_code"])
        df[self.label_col] = df[self.label_col].astype(int)

        valid_labels = set(df[self.label_col].unique().tolist())
        if not valid_labels.issubset({0, 1}):
            raise ValueError(f"Label column must be binary 0/1, got {sorted(valid_labels)}")

        return df.reset_index(drop=True)

    def _select_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Select numeric feature columns."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feature_cols = [
            col
            for col in numeric_cols
            if col not in NON_FEATURE_COLUMNS
            and col != self.label_col
            and col != self.return_col
        ]

        if not feature_cols:
            raise ValueError("No numeric features available for baseline training")

        return feature_cols

    def _split_by_trade_date(self, df: pd.DataFrame) -> DatasetSplit:
        """Split by sorted trade dates to prevent temporal leakage."""
        dates = sorted(df[self.date_col].dt.normalize().unique())
        total_dates = len(dates)

        if total_dates < 30:
            raise ValueError(f"Insufficient trade dates for split: {total_dates}")

        test_size = max(1, int(total_dates * self.test_ratio))
        valid_size = max(1, int(total_dates * self.valid_ratio))
        train_size = total_dates - test_size - valid_size

        if train_size < 10:
            raise ValueError(
                "Train split too small. Increase dataset size or reduce valid/test ratio."
            )

        train_dates = set(dates[:train_size])
        valid_dates = set(dates[train_size : train_size + valid_size])
        test_dates = set(dates[train_size + valid_size :])

        date_series = df[self.date_col].dt.normalize()
        train_df = df[date_series.isin(train_dates)].copy()
        valid_df = df[date_series.isin(valid_dates)].copy()
        test_df = df[date_series.isin(test_dates)].copy()

        return DatasetSplit(train=train_df, valid=valid_df, test=test_df)

    def _build_model(self, model_type: str) -> Pipeline:
        """Build baseline model pipeline."""
        model_type = model_type.lower().strip()

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

    def _evaluate_split(
        self, model: Pipeline, df: pd.DataFrame, feature_cols: List[str], top_k: int
    ) -> Dict:
        """Evaluate one split with classification and top-k return metrics."""
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
        """Compute day-level top-k metrics."""
        eval_df = df.copy()
        eval_df["pred_prob"] = probs
        eval_df["trade_day"] = eval_df[self.date_col].dt.normalize()

        top_returns: List[float] = []
        top_hits: List[float] = []
        top_counts: List[int] = []

        grouped = eval_df.groupby("trade_day", sort=True)
        for _, group in grouped:
            ranked = group.sort_values("pred_prob", ascending=False).head(top_k)
            if ranked.empty:
                continue

            top_counts.append(int(len(ranked)))
            top_hits.append(float(ranked[self.label_col].mean()))
            if self.return_col in ranked.columns:
                top_returns.append(float(ranked[self.return_col].mean()))

        result = {
            "topk_days": int(len(top_counts)),
            "topk_avg_count": float(np.mean(top_counts)) if top_counts else 0.0,
            "topk_hit_rate": float(np.mean(top_hits)) if top_hits else 0.0,
            "topk_avg_future_return": float(np.mean(top_returns)) if top_returns else 0.0,
        }
        return result

    def _extract_top_features(
        self, model: Pipeline, feature_cols: List[str], top_n: int = 20
    ) -> List[Dict]:
        """Extract top feature signals."""
        estimator = model.named_steps.get("model")
        if estimator is None:
            return []

        if hasattr(estimator, "coef_"):
            values = np.abs(estimator.coef_[0])
        elif hasattr(estimator, "feature_importances_"):
            values = estimator.feature_importances_
        else:
            return []

        order = np.argsort(values)[::-1][:top_n]
        return [
            {"feature": feature_cols[idx], "score": float(values[idx])}
            for idx in order
        ]

    def _save_model(self, model: Pipeline, feature_cols: List[str], model_type: str) -> Dict:
        """Persist final model and metadata."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        model_path = os.path.join(MODEL_DIR, "structured_baseline.joblib")
        meta_path = os.path.join(MODEL_DIR, "structured_baseline_metadata.json")

        payload = {
            "model": model,
            "feature_columns": feature_cols,
            "label_column": self.label_col,
            "date_column": self.date_col,
            "return_column": self.return_col,
            "model_type": model_type,
            "saved_at": datetime.now().isoformat(),
        }
        joblib.dump(payload, model_path)

        metadata = {
            "model_path": model_path,
            "model_type": model_type,
            "feature_count": len(feature_cols),
            "saved_at": payload["saved_at"],
        }
        with open(meta_path, "w", encoding="utf-8") as file_obj:
            json.dump(metadata, file_obj, ensure_ascii=False, indent=2)

        return {"model_path": model_path, "metadata_path": meta_path}

    def _save_report(
        self,
        model_type: str,
        feature_cols: List[str],
        split: DatasetSplit,
        valid_metrics: Dict,
        test_metrics: Dict,
        model_info: Dict,
        top_features: List[Dict],
        top_k: int,
    ) -> Dict:
        """Persist evaluation report."""
        os.makedirs(RESULT_DIR, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(RESULT_DIR, f"structured_baseline_report_{timestamp}.json")
        latest_path = os.path.join(RESULT_DIR, "structured_baseline_report_latest.json")

        report = {
            "created_at": datetime.now().isoformat(),
            "dataset_path": self.dataset_path,
            "model_type": model_type,
            "feature_count": len(feature_cols),
            "top_k": top_k,
            "split": {
                "train_samples": int(len(split.train)),
                "valid_samples": int(len(split.valid)),
                "test_samples": int(len(split.test)),
                "train_start": split.train[self.date_col].min().isoformat(),
                "train_end": split.train[self.date_col].max().isoformat(),
                "valid_start": split.valid[self.date_col].min().isoformat(),
                "valid_end": split.valid[self.date_col].max().isoformat(),
                "test_start": split.test[self.date_col].min().isoformat(),
                "test_end": split.test[self.date_col].max().isoformat(),
            },
            "metrics": {"valid": valid_metrics, "test": test_metrics},
            "top_features": top_features,
            "model_files": model_info,
        }

        with open(report_path, "w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)
        with open(latest_path, "w", encoding="utf-8") as file_obj:
            json.dump(report, file_obj, ensure_ascii=False, indent=2)

        report["report_path"] = report_path
        report["latest_path"] = latest_path
        return report


def parse_args() -> argparse.Namespace:
    """Parse CLI args."""
    parser = argparse.ArgumentParser(description="Train structured baseline model")
    parser.add_argument("--dataset-path", type=str, default=None, help="Path to model_dataset csv/parquet")
    parser.add_argument(
        "--model-type",
        type=str,
        default="logistic",
        choices=["logistic", "random_forest"],
        help="Baseline model type",
    )
    parser.add_argument("--top-k", type=int, default=20, help="Top-k for return metrics")
    parser.add_argument("--valid-ratio", type=float, default=0.15, help="Validation date ratio")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Test date ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    trainer = BaselineModelTrainer(
        dataset_path=args.dataset_path,
        valid_ratio=args.valid_ratio,
        test_ratio=args.test_ratio,
        random_state=args.random_state,
    )
    report = trainer.run(model_type=args.model_type, top_k=args.top_k)

    logger.info("Saved baseline model: %s", report["model_files"]["model_path"])
    logger.info("Saved baseline report: %s", report["report_path"])
    logger.info("Test metrics: %s", json.dumps(report["metrics"]["test"], ensure_ascii=False))


if __name__ == "__main__":
    main()
