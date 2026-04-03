"""
Trading-day and news timestamp alignment helpers.
"""
from bisect import bisect_left, bisect_right
from typing import Iterable, List, Optional

import pandas as pd


def parse_timestamp(value) -> Optional[pd.Timestamp]:
    """Parse input into a timezone-naive timestamp."""
    if value is None:
        return None

    try:
        ts = pd.Timestamp(value)
    except Exception:
        return None

    if pd.isna(ts):
        return None

    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)

    return ts


def normalize_trade_dates(values: Iterable) -> List[pd.Timestamp]:
    """Normalize, deduplicate, and sort a trade-date list."""
    normalized = []

    for value in values:
        ts = parse_timestamp(value)
        if ts is None:
            continue
        normalized.append(ts.normalize())

    if not normalized:
        return []

    return sorted(set(normalized))


def build_trade_close_timestamp(
    trade_date,
    market_close_hour: int = 15,
    market_close_minute: int = 0,
) -> Optional[pd.Timestamp]:
    """Build the market close timestamp for one trade date."""
    normalized = parse_timestamp(trade_date)
    if normalized is None:
        return None

    normalized = normalized.normalize()
    return normalized + pd.Timedelta(hours=market_close_hour, minutes=market_close_minute)


def get_next_trade_date(
    trade_dates: Iterable,
    current_date,
    include_current: bool = False,
) -> Optional[pd.Timestamp]:
    """Return the next available trade date."""
    calendar = normalize_trade_dates(trade_dates)
    if not calendar:
        return None

    current = parse_timestamp(current_date)
    if current is None:
        return None

    current = current.normalize()
    index = bisect_left(calendar, current) if include_current else bisect_right(calendar, current)

    if index >= len(calendar):
        return None

    return calendar[index]


def get_previous_trade_date(
    trade_dates: Iterable,
    current_date,
    include_current: bool = False,
) -> Optional[pd.Timestamp]:
    """Return the previous available trade date."""
    calendar = normalize_trade_dates(trade_dates)
    if not calendar:
        return None

    current = parse_timestamp(current_date)
    if current is None:
        return None

    current = current.normalize()
    index = bisect_right(calendar, current) - 1 if include_current else bisect_left(calendar, current) - 1

    if index < 0:
        return None

    return calendar[index]


def align_news_to_trade_date(
    publish_time,
    trade_dates: Iterable,
    market_close_hour: int = 15,
    market_close_minute: int = 0,
) -> Optional[pd.Timestamp]:
    """
    Map a news publish timestamp onto the effective trade date.

    Rules:
    - news published before close on a trade date belongs to that date
    - news published after close belongs to the next trade date
    - news published on a non-trading date belongs to the next trade date
    """
    calendar = normalize_trade_dates(trade_dates)
    if not calendar:
        return None

    publish_ts = parse_timestamp(publish_time)
    if publish_ts is None:
        return None

    publish_date = publish_ts.normalize()
    close_ts = build_trade_close_timestamp(
        publish_date,
        market_close_hour=market_close_hour,
        market_close_minute=market_close_minute,
    )
    trade_date_set = set(calendar)

    if publish_date in trade_date_set and close_ts is not None and publish_ts <= close_ts:
        return publish_date

    if publish_date in trade_date_set:
        return get_next_trade_date(calendar, publish_date, include_current=False)

    return get_next_trade_date(calendar, publish_date, include_current=True)
