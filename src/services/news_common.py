#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common news service helpers."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

import src.web_runtime as runtime


def parse_optional_age_hours(raw_value) -> Optional[int]:
    """Parse optional max age hours; accepts empty/0 as None."""
    if raw_value in ("", "0", 0, 0.0, "none", "null"):
        return None
    if raw_value is None:
        return None
    return runtime.safe_int_param(raw_value, default=72, min_value=1, max_value=24 * 90)


def to_plain_value(value):
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value
