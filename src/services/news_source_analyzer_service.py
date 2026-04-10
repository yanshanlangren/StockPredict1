#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""News source auto-analysis APIs."""

from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse

from flask import jsonify

import src.web_runtime as runtime
from src.news_crawler import get_news_crawler

try:
    from bs4 import BeautifulSoup
except ImportError:
    BeautifulSoup = None


_ARTICLE_URL_PATTERNS = [
    re.compile(r"/\d{8}/\d+\.shtml(?:$|\?)", re.IGNORECASE),
    re.compile(r"/\d{4}/\d{2}/\d{2}/\d+\.shtml(?:$|\?)", re.IGNORECASE),
    re.compile(r"/\d{6,8}\.shtml(?:$|\?)", re.IGNORECASE),
    re.compile(r"\.shtml(?:$|\?)", re.IGNORECASE),
]

_DATE_PATTERN = re.compile(r"(20\d{2}[-/.年]\d{1,2}[-/.月]\d{1,2})")

_CANDIDATE_ITEM_SELECTORS = [
    "div.channel_list.cj_list div.ls_news_c.ls_news_r",
    "div.ls_news_c.ls_news_r",
    "div.ls_news",
    "ul.head_news li",
    "article",
    "div.news-item",
    "li",
]

_CANDIDATE_TITLE_SELECTORS = [
    "a.ls_news_tit",
    "h3 a",
    "h2 a",
    "a",
]

_CANDIDATE_DATE_SELECTORS = [
    "span.list_time",
    "time",
    ".time",
    ".date",
]


def _normalize_url(raw_url: str) -> str:
    text = str(raw_url or "").strip()
    if not text:
        return ""
    if not text.startswith(("http://", "https://")):
        text = f"https://{text.lstrip('/')}"
    return text


def _looks_like_article_url(url: str, host: str) -> bool:
    if not url:
        return False
    try:
        parsed = urlparse(url)
    except Exception:
        return False
    if not parsed.netloc:
        return False
    if host and host not in parsed.netloc:
        return False
    lower = url.lower()
    return any(pattern.search(lower) for pattern in _ARTICLE_URL_PATTERNS)


def _extract_title_anchor(node):
    for selector in _CANDIDATE_TITLE_SELECTORS:
        anchor = node.select_one(selector)
        if anchor is None:
            continue
        title = anchor.get_text(" ", strip=True)
        href = str(anchor.get("href", "") or "").strip()
        if len(title) >= 6 and href:
            return selector, anchor
    return None, None


def _pick_best_selector(soup, base_url: str, host: str):
    best = None
    for selector in _CANDIDATE_ITEM_SELECTORS:
        nodes = soup.select(selector)
        if not nodes:
            continue

        hit = 0
        date_hit = 0
        links = set()
        examples = []

        for node in nodes[:240]:
            _, anchor = _extract_title_anchor(node)
            if anchor is None:
                continue
            href = urljoin(base_url, str(anchor.get("href", "") or "").strip())
            if not _looks_like_article_url(href, host):
                continue
            title = anchor.get_text(" ", strip=True)
            hit += 1
            links.add(href)
            if _DATE_PATTERN.search(node.get_text(" ", strip=True)):
                date_hit += 1
            if len(examples) < 5:
                examples.append(title[:80])

        unique_hit = len(links)
        if unique_hit < 3:
            continue
        score = unique_hit + 0.35 * date_hit + 0.02 * min(hit, 100)
        candidate = {
            "selector": selector,
            "score": score,
            "unique_hit": unique_hit,
            "date_hit": date_hit,
            "examples": examples,
            "nodes": nodes,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def _pick_field_selectors(nodes):
    if not nodes:
        return "a", "span.list_time"

    title_best = "a"
    title_count = -1
    for selector in _CANDIDATE_TITLE_SELECTORS:
        count = 0
        for node in nodes[:100]:
            anchor = node.select_one(selector)
            if anchor is None:
                continue
            title = anchor.get_text(" ", strip=True)
            href = str(anchor.get("href", "") or "").strip()
            if len(title) >= 6 and href:
                count += 1
        if count > title_count:
            title_count = count
            title_best = selector

    date_best = "span.list_time"
    date_count = -1
    for selector in _CANDIDATE_DATE_SELECTORS:
        count = 0
        for node in nodes[:100]:
            text = ""
            found = node.select_one(selector)
            if found is not None:
                text = found.get_text(" ", strip=True)
            if text and _DATE_PATTERN.search(text):
                count += 1
        if count > date_count:
            date_count = count
            date_best = selector

    return title_best, date_best


def _build_suggested_source_id(host: str, source_name: str) -> str:
    if host:
        token = re.sub(r"[^a-z0-9]+", "_", host.lower()).strip("_")
        if token:
            return token[:40]
    name_token = re.sub(r"[^a-z0-9]+", "_", source_name.lower()).strip("_")
    return (name_token or "custom_news_source")[:40]


def analyze_news_source_api(params: dict):
    if BeautifulSoup is None:
        return jsonify({"success": False, "message": "缺少 BeautifulSoup 依赖，无法自动分析"}), 500

    raw_url = (params or {}).get("url")
    source_name = str((params or {}).get("source_name") or "自定义新闻源").strip() or "自定义新闻源"
    keyword = str((params or {}).get("keyword") or source_name).strip() or source_name

    url = _normalize_url(raw_url)
    if not url:
        return jsonify({"success": False, "message": "url 不能为空"}), 400

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    if not host:
        return jsonify({"success": False, "message": "url 无效"}), 400

    crawler = get_news_crawler()
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Referer": f"{parsed.scheme or 'https'}://{host}/",
    }

    try:
        html = crawler._http_get_text(url=url, params={}, headers=headers, timeout=20)
        if not html:
            return jsonify({"success": False, "message": "页面内容为空，自动分析失败"}), 400

        try:
            soup = BeautifulSoup(html, "lxml")
        except Exception:
            soup = BeautifulSoup(html, "html.parser")

        rss_link = ""
        rss_tag = soup.select_one('link[rel="alternate"][type*="rss"], link[type*="rss"]')
        if rss_tag is not None:
            rss_link = urljoin(url, str(rss_tag.get("href", "") or "").strip())

        best = _pick_best_selector(soup, base_url=url, host=host)
        if best is None:
            return jsonify(
                {
                    "success": False,
                    "message": "未识别到稳定的列表结构，建议手工填写 adapter_config。",
                    "data": {
                        "rss_candidate": rss_link or None,
                        "host": host,
                    },
                }
            ), 400

        title_selector, date_selector = _pick_field_selectors(best["nodes"])

        adapter_config = {
            "url": url,
            "headers": headers,
            "item_selector": best["selector"],
            "field_map": {
                "title": {"selector": title_selector, "attr": "text"},
                "content": {"selector": title_selector, "attr": "text"},
                "publish_time": {"selector": date_selector, "attr": "text"},
                "url": {"selector": title_selector, "attr": "href"},
                "source": {"value": source_name},
            },
            "link_base": f"{parsed.scheme or 'https'}://{host}/",
            "max_pages": 1,
            "timeout": 20,
        }

        preview_rows = crawler._fetch_html_selector_rows(
            source_name=source_name,
            keyword=keyword,
            max_items=8,
            adapter_config=adapter_config,
        )
        preview = []
        for row in preview_rows[:5]:
            preview.append(
                {
                    "title": str(row.get("新闻标题", "") or ""),
                    "publish_time": str(row.get("发布时间", "") or ""),
                    "url": str(row.get("新闻链接", "") or ""),
                    "source": str(row.get("来源", "") or source_name),
                }
            )

        return jsonify(
            {
                "success": True,
                "message": "分析完成，已生成可用配置模板",
                "data": {
                    "adapter": "html_selector",
                    "adapter_config": adapter_config,
                    "suggested_source_id": _build_suggested_source_id(host, source_name),
                    "suggested_name": source_name,
                    "keyword": keyword,
                    "analysis": {
                        "host": host,
                        "selector": best["selector"],
                        "unique_hit": best["unique_hit"],
                        "date_hit": best["date_hit"],
                        "examples": best["examples"],
                        "rss_candidate": rss_link or None,
                    },
                    "preview": preview,
                },
            }
        )
    except Exception as exc:
        runtime.logger.error("新闻源自动分析失败: %s", exc)
        return jsonify({"success": False, "message": f"自动分析失败: {str(exc)}"}), 500
