#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dataset information assembly helper."""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from src.services.dataset_precheck_service import build_multimodal_precheck
from src.services.dataset_storage_service import (
    load_dataset_metadata,
    load_dataset_preview,
    load_latest_baseline_report,
)


def build_dataset_info() -> dict[str, Any]:
    """Assemble dataset info including outputs and sample preview."""
    baseline_report = load_latest_baseline_report()
    metadata = load_dataset_metadata()
    multimodal_precheck = build_multimodal_precheck(metadata)

    if not metadata:
        return {
            "available": False,
            "metadata": None,
            "outputs": [],
            "preview": {"columns": [], "rows": []},
            "baseline": baseline_report,
            "multimodal_precheck": multimodal_precheck,
        }

    outputs = []
    for name, path in metadata.get("paths", {}).items():
        exists = bool(path and os.path.exists(path))
        outputs.append(
            {
                "name": name,
                "path": path,
                "exists": exists,
                "size_bytes": os.path.getsize(path) if exists else 0,
                "modified_at": datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if exists else None,
            }
        )

    preview = load_dataset_preview(metadata.get("paths", {}).get("model_dataset"))

    return {
        "available": True,
        "metadata": metadata,
        "outputs": outputs,
        "preview": preview,
        "baseline": baseline_report,
        "multimodal_precheck": multimodal_precheck,
    }
