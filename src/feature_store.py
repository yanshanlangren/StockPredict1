"""
Feature dataset persistence helpers.
"""
import importlib.util
import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd

from config import PROCESSED_DATA_DIR

logger = logging.getLogger(__name__)


class FeatureStore:
    """Persist feature datasets and metadata."""

    def __init__(self, base_dir: str = PROCESSED_DATA_DIR):
        self.base_dir = base_dir
        os.makedirs(self.base_dir, exist_ok=True)

    def parquet_supported(self) -> bool:
        """Return whether parquet output is supported."""
        return bool(
            importlib.util.find_spec("pyarrow") or importlib.util.find_spec("fastparquet")
        )

    def save_dataframe(
        self,
        df: pd.DataFrame,
        dataset_name: str,
        index: bool = False,
        prefer_parquet: bool = True,
    ) -> str:
        """Save a DataFrame and fall back to csv if parquet is unavailable."""
        use_parquet = prefer_parquet and self.parquet_supported()

        if use_parquet:
            output_path = os.path.join(self.base_dir, f"{dataset_name}.parquet")
            df.to_parquet(output_path, index=index)
            logger.info("Saved dataset: %s", output_path)
            return output_path

        output_path = os.path.join(self.base_dir, f"{dataset_name}.csv")
        df.to_csv(output_path, index=index, encoding="utf-8-sig")
        logger.info("Saved dataset: %s", output_path)
        return output_path

    def load_dataframe(self, dataset_name: str) -> Optional[pd.DataFrame]:
        """Load a dataset if it exists."""
        parquet_path = os.path.join(self.base_dir, f"{dataset_name}.parquet")
        csv_path = os.path.join(self.base_dir, f"{dataset_name}.csv")

        if os.path.exists(parquet_path):
            return pd.read_parquet(parquet_path)

        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)

        return None

    def save_json(self, payload: Dict[str, Any], dataset_name: str) -> str:
        """Save metadata as JSON."""
        output_path = os.path.join(self.base_dir, f"{dataset_name}.json")

        with open(output_path, "w", encoding="utf-8") as file_obj:
            json.dump(payload, file_obj, ensure_ascii=False, indent=2)

        logger.info("Saved metadata: %s", output_path)
        return output_path
