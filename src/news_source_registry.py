#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""News source registry backed by JSON config file."""

from __future__ import annotations

import copy
import json
import os
import re
import shutil
import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from config import BASE_DIR, DATA_DIR


SUPPORTED_NEWS_ADAPTERS = ("json_api", "html_selector", "rss")
LEGACY_NEWS_ADAPTERS = ("eastmoney", "sina", "tencent")

ADAPTER_CONFIG_TEMPLATES: dict[str, dict[str, Any]] = {
    "json_api": {
        "url": "https://example.com/api/news",
        "params": {},
        "headers": {},
        "list_path": "data.items",
        "field_map": {
            "title": "title",
            "content": "summary",
            "publish_time": "publish_time",
            "url": "url",
            "source": "source",
        },
        "query_param": "q",
        "page_param": "page",
        "page_start": 1,
        "max_pages": 1,
        "timeout": 20,
    },
    "html_selector": {
        "url": "https://example.com/news",
        "headers": {},
        "item_selector": ".news-item",
        "field_map": {
            "title": {"selector": "h3", "attr": "text"},
            "content": {"selector": ".summary", "attr": "text"},
            "publish_time": {"selector": ".time", "attr": "text"},
            "url": {"selector": "a", "attr": "href"},
            "source": {"selector": ".source", "attr": "text"},
        },
        "link_base": "https://example.com",
        "query_param": "q",
        "page_param": "page",
        "page_start": 1,
        "max_pages": 1,
        "timeout": 20,
    },
    "rss": {
        "url": "https://example.com/feed.xml",
        "headers": {},
        "item_tag": "item",
        "timeout": 20,
    },
}


@dataclass
class NewsSourceConfig:
    source_id: str
    name: str
    adapter: str
    keyword: str
    enabled: bool = True
    description: str = ""
    sort_order: int = 100
    is_builtin: bool = False
    adapter_config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NewsSourceRegistry:
    """Manage news source configurations persisted in JSON."""

    _LEGACY_ALIAS_MAP = {
        "东方财富": "eastmoney",
        "财经新闻": "eastmoney",
        "新浪": "sina",
        "新浪财经": "sina",
        "腾讯": "tencent",
        "腾讯财经": "tencent",
    }
    _LEGACY_CONFIG_FILE = os.path.join(DATA_DIR, "config", "news_sources.json")

    def __init__(self, config_file: str | None = None):
        config_dir = os.path.join(BASE_DIR, "configs")
        os.makedirs(config_dir, exist_ok=True)

        self._config_file = config_file or os.path.join(config_dir, "news_sources.json")
        os.makedirs(os.path.dirname(self._config_file), exist_ok=True)
        self._lock = threading.Lock()

    @property
    def config_file(self) -> str:
        return self._config_file

    def get_adapter_templates(self) -> dict[str, dict[str, Any]]:
        return copy.deepcopy(ADAPTER_CONFIG_TEMPLATES)

    def _default_sources(self) -> list[NewsSourceConfig]:
        return [
            NewsSourceConfig(
                source_id="eastmoney",
                name="东方财富",
                adapter="json_api",
                keyword="财经新闻",
                enabled=True,
                description="默认东方财富新闻源",
                sort_order=10,
                is_builtin=True,
                adapter_config={"preset": "eastmoney_search"},
            ),
            NewsSourceConfig(
                source_id="sina",
                name="新浪财经",
                adapter="json_api",
                keyword="新浪财经",
                enabled=True,
                description="默认新浪财经新闻源",
                sort_order=20,
                is_builtin=True,
                adapter_config={"preset": "eastmoney_search"},
            ),
            NewsSourceConfig(
                source_id="tencent",
                name="腾讯财经",
                adapter="json_api",
                keyword="腾讯财经",
                enabled=True,
                description="默认腾讯财经新闻源",
                sort_order=30,
                is_builtin=True,
                adapter_config={"preset": "tencent_hot", "fallback_search": True},
            ),
        ]

    def _normalize_source_id(self, source_id: str) -> str:
        text = str(source_id or "").strip().lower()
        text = re.sub(r"[^a-z0-9_-]+", "_", text)
        text = re.sub(r"_+", "_", text).strip("_")
        if not text:
            return ""
        return text[:40]

    def _parse_adapter_config(self, raw_config: Any) -> dict[str, Any]:
        if raw_config is None:
            return {}
        if isinstance(raw_config, dict):
            return copy.deepcopy(raw_config)
        if isinstance(raw_config, str):
            text = raw_config.strip()
            if not text:
                return {}
            try:
                parsed = json.loads(text)
            except Exception as exc:
                raise ValueError(f"adapter_config JSON 解析失败: {exc}") from exc
            if not isinstance(parsed, dict):
                raise ValueError("adapter_config 必须是 JSON 对象")
            return parsed
        raise ValueError("adapter_config 必须是 JSON 对象")

    def _normalize_adapter_and_config(
        self,
        adapter_raw: Any,
        adapter_config: dict[str, Any],
    ) -> tuple[str, dict[str, Any]]:
        adapter = str(adapter_raw or "").strip().lower()
        if adapter in SUPPORTED_NEWS_ADAPTERS:
            return adapter, adapter_config

        if adapter in LEGACY_NEWS_ADAPTERS:
            if adapter == "tencent":
                base = {"preset": "tencent_hot", "fallback_search": True}
            else:
                base = {"preset": "eastmoney_search"}
            merged = dict(base)
            merged.update(adapter_config)
            return "json_api", merged

        return adapter, adapter_config

    def _build_source_from_dict(
        self,
        payload: dict[str, Any],
        *,
        default_id: str | None = None,
    ) -> NewsSourceConfig:
        source_id = self._normalize_source_id(payload.get("source_id") or default_id or "")
        if not source_id:
            raise ValueError("source_id 不能为空")

        name = str(payload.get("name") or "").strip()
        if not name:
            raise ValueError("name 不能为空")

        adapter_config = self._parse_adapter_config(payload.get("adapter_config"))
        adapter, adapter_config = self._normalize_adapter_and_config(
            payload.get("adapter"),
            adapter_config,
        )
        if adapter not in SUPPORTED_NEWS_ADAPTERS:
            raise ValueError(f"adapter 仅支持: {', '.join(SUPPORTED_NEWS_ADAPTERS)}")

        keyword = str(payload.get("keyword") or "").strip() or name
        enabled = bool(payload.get("enabled", True))

        try:
            sort_order = int(payload.get("sort_order", 100))
        except Exception:
            sort_order = 100
        sort_order = max(-10000, min(sort_order, 10000))

        description = str(payload.get("description") or "").strip()
        is_builtin = bool(payload.get("is_builtin", False))

        return NewsSourceConfig(
            source_id=source_id,
            name=name,
            adapter=adapter,
            keyword=keyword,
            enabled=enabled,
            description=description,
            sort_order=sort_order,
            is_builtin=is_builtin,
            adapter_config=adapter_config,
        )

    def _ensure_config_file(self) -> None:
        self._migrate_legacy_config_if_needed()
        if os.path.exists(self._config_file):
            return
        defaults = [item.to_dict() for item in self._default_sources()]
        with open(self._config_file, "w", encoding="utf-8") as file_obj:
            json.dump(defaults, file_obj, ensure_ascii=False, indent=2)

    def _migrate_legacy_config_if_needed(self) -> None:
        legacy_file = self._LEGACY_CONFIG_FILE
        if self._config_file == legacy_file:
            return
        if os.path.exists(self._config_file):
            return
        if not os.path.exists(legacy_file):
            return
        try:
            shutil.copyfile(legacy_file, self._config_file)
        except Exception:
            pass

    def _load_sources_locked(self) -> list[NewsSourceConfig]:
        self._ensure_config_file()

        try:
            with open(self._config_file, "r", encoding="utf-8") as file_obj:
                payload = json.load(file_obj)
        except Exception:
            payload = []

        if not isinstance(payload, list):
            payload = []

        sources: list[NewsSourceConfig] = []
        seen: set[str] = set()
        for item in payload:
            if not isinstance(item, dict):
                continue
            try:
                source = self._build_source_from_dict(item)
            except Exception:
                continue
            if source.source_id in seen:
                continue
            seen.add(source.source_id)
            sources.append(source)

        if not sources:
            sources = self._default_sources()
            self._save_sources_locked(sources)

        sources.sort(key=lambda x: (x.sort_order, x.source_id))
        return sources

    def _save_sources_locked(self, sources: list[NewsSourceConfig]) -> None:
        serializable = [item.to_dict() for item in sources]
        with open(self._config_file, "w", encoding="utf-8") as file_obj:
            json.dump(serializable, file_obj, ensure_ascii=False, indent=2)

    def list_sources(self, include_disabled: bool = True) -> list[dict[str, Any]]:
        with self._lock:
            sources = self._load_sources_locked()
            result = []
            for source in sources:
                if (not include_disabled) and (not source.enabled):
                    continue
                result.append(source.to_dict())
            return result

    def get_source(self, source_id: str) -> dict[str, Any] | None:
        target = self._normalize_source_id(source_id)
        with self._lock:
            for source in self._load_sources_locked():
                if source.source_id == target:
                    return source.to_dict()
        return None

    def list_enabled_source_ids(self) -> list[str]:
        with self._lock:
            sources = self._load_sources_locked()
            enabled_ids = [item.source_id for item in sources if item.enabled]
            if enabled_ids:
                return enabled_ids
            fallback = self._default_source_id_from_sources(sources)
            return [fallback] if fallback else ["eastmoney"]

    def list_enabled_sources(self) -> list[dict[str, Any]]:
        return self.list_sources(include_disabled=False)

    def get_default_source_id(self) -> str:
        enabled_ids = self.list_enabled_source_ids()
        return enabled_ids[0] if enabled_ids else "eastmoney"

    def _default_source_id_from_sources(self, sources: list[NewsSourceConfig]) -> str:
        enabled_ids = [item.source_id for item in sources if item.enabled]
        if enabled_ids:
            return enabled_ids[0]
        if sources:
            return sources[0].source_id
        return "eastmoney"

    def resolve_source_id(self, raw_source: Any) -> str:
        text = str(raw_source or "").strip()
        if not text:
            return self.get_default_source_id()

        text_lower = text.lower()
        if text_lower in {"all", "全部", "全部来源"}:
            return "all"

        normalized = self._normalize_source_id(text)
        with self._lock:
            sources = self._load_sources_locked()
            source_map = {item.source_id: item for item in sources}
            default_id = self._default_source_id_from_sources(sources)
            if normalized in source_map:
                return normalized

            for item in sources:
                if text_lower == str(item.name).strip().lower():
                    return item.source_id

            alias_target = self._LEGACY_ALIAS_MAP.get(text)
            if alias_target and alias_target in source_map:
                return alias_target

            return default_id

    def create_source(self, payload: dict[str, Any]) -> dict[str, Any]:
        raw = copy.deepcopy(payload or {})
        raw_id = self._normalize_source_id(raw.get("source_id") or "")
        if not raw_id:
            generated = self._normalize_source_id(raw.get("name") or "")
            raw_id = generated or f"source_{len(self.list_sources(include_disabled=True)) + 1}"
            raw["source_id"] = raw_id

        with self._lock:
            sources = self._load_sources_locked()
            if any(item.source_id == raw_id for item in sources):
                raise ValueError(f"source_id 已存在: {raw_id}")

            source = self._build_source_from_dict(raw, default_id=raw_id)
            sources.append(source)
            sources.sort(key=lambda x: (x.sort_order, x.source_id))
            self._save_sources_locked(sources)
            return source.to_dict()

    def update_source(self, source_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        target_id = self._normalize_source_id(source_id)
        raw = copy.deepcopy(payload or {})
        raw["source_id"] = target_id

        with self._lock:
            sources = self._load_sources_locked()
            index = -1
            for idx, item in enumerate(sources):
                if item.source_id == target_id:
                    index = idx
                    break
            if index < 0:
                raise KeyError(f"新闻源不存在: {target_id}")

            base = sources[index].to_dict()
            base.update(raw)
            base["source_id"] = target_id
            source = self._build_source_from_dict(base, default_id=target_id)
            sources[index] = source
            sources.sort(key=lambda x: (x.sort_order, x.source_id))
            self._save_sources_locked(sources)
            return source.to_dict()

    def delete_source(self, source_id: str) -> None:
        target_id = self._normalize_source_id(source_id)
        with self._lock:
            sources = self._load_sources_locked()
            remaining = [item for item in sources if item.source_id != target_id]
            if len(remaining) == len(sources):
                raise KeyError(f"新闻源不存在: {target_id}")
            if not remaining:
                raise ValueError("至少保留一个新闻源")
            self._save_sources_locked(remaining)


_registry_singleton: NewsSourceRegistry | None = None
_registry_lock = threading.Lock()


def get_news_source_registry() -> NewsSourceRegistry:
    global _registry_singleton
    with _registry_lock:
        if _registry_singleton is None:
            _registry_singleton = NewsSourceRegistry()
        return _registry_singleton
