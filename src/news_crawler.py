#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新闻爬取引擎 - 财经新闻数据采集

功能：
1. 从多个财经网站爬取新闻
2. 新闻分类和情感分析
3. 新闻缓存管理
"""

import os
import json
import logging
import hashlib
import time
from typing import Optional, List, Dict
from datetime import datetime, timedelta
import re

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
    logger.info("✓ akshare库已加载，新闻爬取功能可用")
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.error("✗ akshare库未安装，新闻爬取功能不可用")


class NewsCrawler:
    """财经新闻爬取引擎"""

    def __init__(self, cache_dir: str = "data/cache/news"):
        """
        初始化新闻爬虫

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 新闻类别映射
        self.category_map = {
            '财经': ['财经', '经济', '金融', '投资', '理财'],
            '股票': ['股票', 'A股', '港股', '美股', '股市'],
            '基金': ['基金', 'ETF', '公募', '私募'],
            '债券': ['债券', '国债', '企业债'],
            '期货': ['期货', '大宗', '商品'],
            '外汇': ['外汇', '汇率', '人民币', '美元'],
            '银行': ['银行', '存款', '贷款', '利率'],
            '保险': ['保险', '寿险', '财险'],
            '宏观': ['宏观', 'GDP', 'CPI', 'PMI', '政策'],
            '公司': ['公司', '企业', '财报', '业绩']
        }
        
        # 领域关键词映射
        self.sector_keywords = {
            '银行': ['银行', '贷款', '存款', '利率', 'LPR'],
            '证券': ['证券', '券商', '投行', 'IPO', '上市'],
            '保险': ['保险', '寿险', '财险', '养老金'],
            '地产': ['房地产', '楼盘', '房价', '土地', '万科'],
            '汽车': ['汽车', '新能源车', '电动车', '比亚迪', '特斯拉'],
            '科技': ['科技', '人工智能', 'AI', '芯片', '半导体'],
            '医药': ['医药', '医疗', '生物', '疫苗', '创新药'],
            '消费': ['消费', '零售', '电商', '白酒', '食品'],
            '能源': ['能源', '石油', '天然气', '新能源', '光伏'],
            '军工': ['军工', '国防', '航空', '航天'],
            '基建': ['基建', '建筑', '工程', '水泥', '钢铁'],
            '传媒': ['传媒', '影视', '游戏', '广告'],
            '教育': ['教育', '培训', '学校'],
            '农业': ['农业', '粮食', '种子', '养殖']
        }

    def is_available(self) -> bool:
        """检查爬虫是否可用"""
        return AKSHARE_AVAILABLE

    def get_news(self, stock_code: Optional[str] = None, 
                 limit: int = 50,
                 use_cache: bool = True,
                 source: str = 'eastmoney') -> List[Dict]:
        """
        获取财经新闻

        Args:
            stock_code: 股票代码（可选，用于筛选相关新闻）
            limit: 返回数量限制
            use_cache: 是否使用缓存
            source: 新闻源 ('eastmoney', 'sina', 'tencent', 'all')

        Returns:
            新闻列表
        """
        if not AKSHARE_AVAILABLE:
            logger.error("akshare库不可用，无法获取新闻")
            return []

        # 尝试从缓存读取
        if use_cache:
            cached = self._load_from_cache(stock_code)
            if cached:
                logger.info(f"从缓存加载 {len(cached)} 条新闻")
                return cached[:limit]

        try:
            # 获取新闻数据
            news_list = []
            
            # 根据source参数获取不同来源的新闻
            if source in ['eastmoney', 'all']:
                try:
                    # 东方财富财经新闻
                    df = ak.stock_news_em(symbol="财经新闻")
                    if not df.empty:
                        for _, row in df.iterrows():
                            news_item = self._parse_news_row(row)
                            if news_item:
                                news_item['source'] = news_item.get('source', '东方财富')
                                news_list.append(news_item)
                except Exception as e:
                    logger.warning(f"获取东方财富新闻失败: {e}")
            
            if source in ['sina', 'all']:
                try:
                    # 新浪财经新闻
                    df = ak.stock_news_em(symbol="新浪财经")
                    if not df.empty:
                        for _, row in df.iterrows():
                            news_item = self._parse_news_row(row)
                            if news_item:
                                news_item['source'] = '新浪财经'
                                news_list.append(news_item)
                except Exception as e:
                    logger.warning(f"获取新浪财经新闻失败: {e}")
            
            if source in ['tencent', 'all']:
                try:
                    # 腾讯财经新闻
                    df = ak.stock_news_em(symbol="腾讯财经")
                    if not df.empty:
                        for _, row in df.iterrows():
                            news_item = self._parse_news_row(row)
                            if news_item:
                                news_item['source'] = '腾讯财经'
                                news_list.append(news_item)
                except Exception as e:
                    logger.warning(f"获取腾讯财经新闻失败: {e}")

            # 如果指定了股票代码，筛选相关新闻
            if stock_code and news_list:
                news_list = self._filter_by_stock(news_list, stock_code)

            # 按时间排序
            news_list.sort(key=lambda x: x.get('publish_time', ''), reverse=True)

            # 保存到缓存
            if news_list:
                self._save_to_cache(news_list, stock_code)

            logger.info(f"成功获取 {len(news_list)} 条新闻")
            return news_list[:limit]

        except Exception as e:
            logger.error(f"获取新闻失败: {e}")
            return []

    def _parse_news_row(self, row) -> Optional[Dict]:
        """
        解析新闻行数据

        Args:
            row: 新闻数据行

        Returns:
            解析后的新闻字典
        """
        try:
            # 获取标题和内容
            title = str(row.get('新闻标题', row.get('标题', '')))
            content = str(row.get('新闻内容', row.get('内容', '')))
            
            # 获取时间
            publish_time = row.get('发布时间', row.get('时间', ''))
            if isinstance(publish_time, str):
                try:
                    publish_time = datetime.strptime(publish_time, '%Y-%m-%d %H:%M:%S')
                except:
                    publish_time = datetime.now()
            elif not isinstance(publish_time, datetime):
                publish_time = datetime.now()

            # 生成新闻ID
            news_id = hashlib.md5(f"{title}{publish_time}".encode()).hexdigest()[:12]

            # 分类新闻
            categories = self._classify_news(title + ' ' + content)
            
            # 情感分析
            sentiment = self._analyze_sentiment(title + ' ' + content)
            
            # 重要性评分
            importance = self._calculate_importance(title, content)
            
            # 影响的领域
            affected_sectors = self._extract_affected_sectors(title + ' ' + content)

            return {
                'news_id': news_id,
                'title': title,
                'content': content[:500] if content else '',  # 限制内容长度
                'source': str(row.get('来源', '未知')),
                'publish_time': publish_time.strftime('%Y-%m-%d %H:%M:%S'),
                'url': str(row.get('新闻链接', row.get('链接', ''))),
                'categories': categories,
                'sentiment': sentiment,
                'importance': importance,
                'affected_sectors': affected_sectors
            }

        except Exception as e:
            logger.warning(f"解析新闻失败: {e}")
            return None

    def _classify_news(self, text: str) -> List[str]:
        """
        分类新闻

        Args:
            text: 新闻文本

        Returns:
            分类列表
        """
        categories = []
        for category, keywords in self.category_map.items():
            for keyword in keywords:
                if keyword in text:
                    categories.append(category)
                    break
        return categories if categories else ['其他']

    def _analyze_sentiment(self, text: str) -> float:
        """
        简单的情感分析

        Args:
            text: 新闻文本

        Returns:
            情感得分 (-1 到 1)
        """
        # 正面关键词
        positive_words = ['上涨', '增长', '盈利', '突破', '创新高', '利好', 
                         '增持', '回购', '分红', '业绩增长', '超预期',
                         '突破', '领涨', '涨停', '牛市', '强势']
        
        # 负面关键词
        negative_words = ['下跌', '亏损', '暴跌', '利空', '减持', '质押',
                         '违约', '退市', '熊市', '跌停', '下滑', '预警',
                         '风险', '诉讼', '处罚', '调查']

        positive_count = sum(1 for w in positive_words if w in text)
        negative_count = sum(1 for w in negative_words if w in text)

        total = positive_count + negative_count
        if total == 0:
            return 0.0
        
        sentiment = (positive_count - negative_count) / total
        return round(sentiment, 3)

    def _calculate_importance(self, title: str, content: str) -> float:
        """
        计算新闻重要性评分

        Args:
            title: 标题
            content: 内容

        Returns:
            重要性评分 (0-1)
        """
        score = 0.5  # 基础分

        # 标题中的关键词权重更高
        important_title_words = ['央行', '证监会', '发改委', '国务院', '政策',
                                '降准', '降息', 'IPO', '重组', '并购', '财报']
        for word in important_title_words:
            if word in title:
                score += 0.1

        # 内容长度贡献
        if len(content) > 200:
            score += 0.1

        # 包含数字（通常意味着具体数据）
        if re.search(r'\d+[.%亿万元]', content):
            score += 0.05

        return min(1.0, round(score, 2))

    def _extract_affected_sectors(self, text: str) -> Dict[str, float]:
        """
        提取受影响的领域

        Args:
            text: 新闻文本

        Returns:
            {领域: 影响程度} 字典
        """
        affected = {}
        for sector, keywords in self.sector_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    # 基础影响程度
                    impact = 0.5
                    # 标题中出现的影响更大
                    if keyword in text[:100]:  # 假设标题在前100字符
                        impact = 0.8
                    affected[sector] = impact
                    break
        return affected

    def _filter_by_stock(self, news_list: List[Dict], stock_code: str) -> List[Dict]:
        """
        按股票代码筛选新闻

        Args:
            news_list: 新闻列表
            stock_code: 股票代码

        Returns:
            筛选后的新闻列表
        """
        # 获取股票名称（简化处理）
        stock_names = self._get_stock_names(stock_code)
        
        filtered = []
        for news in news_list:
            text = news.get('title', '') + ' ' + news.get('content', '')
            for name in stock_names:
                if name in text:
                    news['relevance'] = 0.8
                    filtered.append(news)
                    break
        
        return filtered if filtered else news_list

    def _get_stock_names(self, stock_code: str) -> List[str]:
        """获取股票相关名称"""
        # 预定义的股票名称映射
        stock_name_map = {
            '600000': ['浦发银行', '浦发'],
            '600519': ['贵州茅台', '茅台'],
            '600036': ['招商银行', '招行'],
            '000001': ['平安银行', '平安'],
            '000002': ['万科A', '万科'],
            '600690': ['海尔智家', '海尔'],
            '000858': ['五粮液'],
            '600030': ['中信证券'],
            # 可以继续添加
        }
        return stock_name_map.get(stock_code, [stock_code])

    def _load_from_cache(self, stock_code: Optional[str] = None) -> Optional[List[Dict]]:
        """从缓存加载新闻"""
        cache_file = os.path.join(
            self.cache_dir, 
            f"news_{stock_code if stock_code else 'all'}.json"
        )
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            # 检查缓存是否过期（超过1小时）
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time > timedelta(hours=1):
                return None

            with open(cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"读取缓存失败: {e}")
            return None

    def _save_to_cache(self, news_list: List[Dict], stock_code: Optional[str] = None):
        """保存新闻到缓存"""
        cache_file = os.path.join(
            self.cache_dir, 
            f"news_{stock_code if stock_code else 'all'}.json"
        )
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(news_list, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def get_news_statistics(self, news_list: List[Dict]) -> Dict:
        """
        获取新闻统计信息

        Args:
            news_list: 新闻列表

        Returns:
            统计信息字典
        """
        if not news_list:
            return {'total': 0}

        # 情感分布
        sentiments = [n.get('sentiment', 0) for n in news_list]
        positive = sum(1 for s in sentiments if s > 0.2)
        negative = sum(1 for s in sentiments if s < -0.2)
        neutral = len(sentiments) - positive - negative

        # 领域分布
        sectors = {}
        for news in news_list:
            for sector, impact in news.get('affected_sectors', {}).items():
                if sector not in sectors:
                    sectors[sector] = 0
                sectors[sector] += 1
        
        # 来源分布
        sources = {}
        for news in news_list:
            source = news.get('source', '未知')
            sources[source] = sources.get(source, 0) + 1

        return {
            'total': len(news_list),
            'sentiment_distribution': {
                'positive': positive,
                'negative': negative,
                'neutral': neutral
            },
            'avg_sentiment': round(np.mean(sentiments), 3) if sentiments else 0,
            'avg_importance': round(np.mean([n.get('importance', 0) for n in news_list]), 2),
            'sector_distribution': dict(sorted(sectors.items(), key=lambda x: x[1], reverse=True)[:10]),
            'source_distribution': dict(sorted(sources.items(), key=lambda x: x[1], reverse=True))
        }


# 全局实例
news_crawler = NewsCrawler()


def get_news_crawler() -> NewsCrawler:
    """获取新闻爬虫实例"""
    return news_crawler
