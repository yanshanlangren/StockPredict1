#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公司相关性图谱 - 构建股票-领域-概念的相关性网络

功能：
1. 构建股票-领域相关性矩阵
2. 构建股票-概念相关性图谱
3. 计算股票间关联度
4. 提供图谱可视化数据
"""

import os
import json
import logging
from typing import Optional, Dict, List, Tuple, Set
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class RelevanceGraph:
    """公司相关性图谱"""

    def __init__(self, cache_dir: str = "data/cache/graph"):
        """
        初始化相关性图谱

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 领域定义
        self.sectors = [
            '银行', '证券', '保险', '地产', '汽车', 
            '科技', '医药', '消费', '能源', '军工',
            '基建', '传媒', '教育', '农业', '交通'
        ]
        
        # 概念定义
        self.concepts = [
            '金融科技', '消费升级', '新能源', '人工智能',
            '物联网', '5G', '芯片', '生物医药', '碳中和',
            '国企改革', '一带一路', '粤港澳大湾区', '雄安新区',
            '智能制造', '工业互联网', '数字货币', '国企混改'
        ]
        
        # 股票-领域相关性矩阵（预定义）
        self._stock_sector_matrix = {
            '600000': {'银行': 0.95, '金融': 0.90, '上海本地': 0.6},
            '600036': {'银行': 0.92, '金融': 0.88, '金融科技': 0.7, '零售银行': 0.85},
            '600519': {'白酒': 0.95, '消费': 0.90, '消费升级': 0.85, '奢侈品': 0.8},
            '000858': {'白酒': 0.92, '消费': 0.88, '消费升级': 0.82},
            '000001': {'银行': 0.90, '金融': 0.88, '金融科技': 0.75, '平安集团': 0.85},
            '000002': {'地产': 0.90, '物业管理': 0.85, '深圳本地': 0.7},
            '600690': {'家电': 0.88, '消费': 0.80, '智能家居': 0.75, '物联网': 0.65},
            '600030': {'证券': 0.95, '金融': 0.90, '券商': 0.92},
            '601318': {'保险': 0.92, '金融': 0.88, '综合金融': 0.85},
            '600009': {'机场': 0.90, '交通': 0.85, '免税店': 0.75, '上海本地': 0.7}
        }
        
        # 股票基本信息
        self._stock_info = {
            '600000': {'name': '浦发银行', 'industry': '银行'},
            '600036': {'name': '招商银行', 'industry': '银行'},
            '600519': {'name': '贵州茅台', 'industry': '白酒'},
            '000858': {'name': '五粮液', 'industry': '白酒'},
            '000001': {'name': '平安银行', 'industry': '银行'},
            '000002': {'name': '万科A', 'industry': '房地产'},
            '600690': {'name': '海尔智家', 'industry': '家电'},
            '600030': {'name': '中信证券', 'industry': '证券'},
            '601318': {'name': '中国平安', 'industry': '保险'},
            '600009': {'name': '上海机场', 'industry': '机场'}
        }
        
        # 领域-领域关联度
        self._sector_correlation = {
            ('银行', '证券'): 0.7,
            ('银行', '保险'): 0.6,
            ('证券', '保险'): 0.65,
            ('白酒', '消费'): 0.8,
            ('家电', '消费'): 0.75,
            ('地产', '银行'): 0.5,
            ('科技', '人工智能'): 0.85,
            ('科技', '芯片'): 0.8,
            ('汽车', '新能源'): 0.75,
            ('医药', '生物医药'): 0.85,
        }

    def get_stock_relevance_graph(self, stock_code: str, 
                                   depth: int = 2) -> Dict:
        """
        获取股票相关性图谱

        Args:
            stock_code: 股票代码
            depth: 图谱深度（1=直接关联，2=二度关联）

        Returns:
            图谱数据（节点和边）
        """
        nodes = []
        edges = []
        visited = set()

        # 添加中心节点
        stock_info = self._stock_info.get(stock_code, {'name': stock_code, 'industry': '未知'})
        nodes.append({
            'id': stock_code,
            'label': stock_info['name'],
            'type': 'stock',
            'industry': stock_info['industry'],
            'size': 30
        })
        visited.add(stock_code)

        # 获取股票的领域关联
        stock_sectors = self._stock_sector_matrix.get(stock_code, {})
        
        # 第一层：股票-领域关联
        for sector, weight in stock_sectors.items():
            sector_id = f"sector_{sector}"
            if sector_id not in visited:
                nodes.append({
                    'id': sector_id,
                    'label': sector,
                    'type': 'sector',
                    'size': 20
                })
                visited.add(sector_id)
            
            edges.append({
                'source': stock_code,
                'target': sector_id,
                'weight': round(weight, 2),
                'type': 'stock-sector'
            })

        # 第二层：领域-股票关联（同类股票）
        if depth >= 2:
            for sector in stock_sectors.keys():
                related_stocks = self._get_stocks_by_sector(sector)
                for related_stock, relevance in related_stocks.items():
                    if related_stock != stock_code and related_stock not in visited:
                        related_info = self._stock_info.get(related_stock, 
                                                           {'name': related_stock, 'industry': '未知'})
                        nodes.append({
                            'id': related_stock,
                            'label': related_info['name'],
                            'type': 'stock',
                            'industry': related_info['industry'],
                            'size': 25
                        })
                        visited.add(related_stock)
                        
                        sector_id = f"sector_{sector}"
                        edges.append({
                            'source': related_stock,
                            'target': sector_id,
                            'weight': round(relevance, 2),
                            'type': 'stock-sector'
                        })

        return {
            'center': stock_code,
            'nodes': nodes,
            'edges': edges,
            'depth': depth,
            'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def get_relevance_matrix(self, stock_codes: Optional[List[str]] = None) -> Dict:
        """
        获取股票相关性矩阵

        Args:
            stock_codes: 股票代码列表，None则使用全部

        Returns:
            相关性矩阵数据
        """
        if stock_codes is None:
            stock_codes = list(self._stock_info.keys())
        
        n = len(stock_codes)
        matrix = np.zeros((n, n))
        
        # 计算两两相关性
        for i, code1 in enumerate(stock_codes):
            for j, code2 in enumerate(stock_codes):
                if i == j:
                    matrix[i][j] = 1.0
                else:
                    matrix[i][j] = self._calculate_stock_correlation(code1, code2)
        
        return {
            'stock_codes': stock_codes,
            'stock_names': [self._stock_info.get(c, {}).get('name', c) for c in stock_codes],
            'matrix': matrix.tolist(),
            'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def _calculate_stock_correlation(self, code1: str, code2: str) -> float:
        """
        计算两只股票的相关性

        Args:
            code1: 股票代码1
            code2: 股票代码2

        Returns:
            相关性分数 (0-1)
        """
        sectors1 = self._stock_sector_matrix.get(code1, {})
        sectors2 = self._stock_sector_matrix.get(code2, {})
        
        if not sectors1 or not sectors2:
            return 0.0

        # 找共同领域
        common_sectors = set(sectors1.keys()) & set(sectors2.keys())
        if not common_sectors:
            return 0.0
        
        # 计算加权相关性
        correlation = 0
        for sector in common_sectors:
            w1 = sectors1[sector]
            w2 = sectors2[sector]
            correlation += (w1 + w2) / 2
        
        correlation /= len(common_sectors)
        return round(correlation, 3)

    def _get_stocks_by_sector(self, sector: str) -> Dict[str, float]:
        """
        获取某领域的所有股票

        Args:
            sector: 领域名称

        Returns:
            {股票代码: 相关性} 字典
        """
        result = {}
        for stock_code, sectors in self._stock_sector_matrix.items():
            if sector in sectors:
                result[stock_code] = sectors[sector]
        return result

    def get_sector_heatmap(self) -> Dict:
        """
        获取领域热度图数据

        Returns:
            热度图数据
        """
        sector_stats = {}
        
        for sector in self.sectors:
            stocks = self._get_stocks_by_sector(sector)
            sector_stats[sector] = {
                'stock_count': len(stocks),
                'avg_relevance': round(np.mean(list(stocks.values())) if stocks else 0, 3),
                'top_stocks': sorted(stocks.items(), key=lambda x: x[1], reverse=True)[:5]
            }
        
        return {
            'sectors': self.sectors,
            'stats': sector_stats,
            'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def find_related_stocks(self, stock_code: str, 
                            top_n: int = 10,
                            min_correlation: float = 0.3) -> List[Dict]:
        """
        查找相关股票

        Args:
            stock_code: 股票代码
            top_n: 返回数量
            min_correlation: 最小相关性阈值

        Returns:
            相关股票列表
        """
        if stock_code not in self._stock_sector_matrix:
            return []

        correlations = []
        for other_code in self._stock_info.keys():
            if other_code != stock_code:
                corr = self._calculate_stock_correlation(stock_code, other_code)
                if corr >= min_correlation:
                    correlations.append({
                        'code': other_code,
                        'name': self._stock_info.get(other_code, {}).get('name', other_code),
                        'correlation': corr,
                        'industry': self._stock_info.get(other_code, {}).get('industry', '未知')
                    })

        # 按相关性排序
        correlations.sort(key=lambda x: x['correlation'], reverse=True)
        
        return correlations[:top_n]

    def get_industry_chain(self, stock_code: str) -> Dict:
        """
        获取产业链图谱

        Args:
            stock_code: 股票代码

        Returns:
            产业链图谱
        """
        stock_info = self._stock_info.get(stock_code, {})
        industry = stock_info.get('industry', '')
        
        # 预定义产业链关系
        industry_chains = {
            '银行': {
                'upstream': ['金融科技', 'IT服务', '数据服务'],
                'midstream': ['银行', '支付服务'],
                'downstream': ['企业客户', '个人客户', '政府机构']
            },
            '白酒': {
                'upstream': ['粮食', '包装材料', '酿造设备'],
                'midstream': ['白酒生产', '品牌运营'],
                'downstream': ['经销商', '零售终端', '消费者']
            },
            '家电': {
                'upstream': ['钢材', '塑料', '电子元器件', '芯片'],
                'midstream': ['家电制造', '品牌运营'],
                'downstream': ['电商平台', '家电卖场', '消费者']
            },
            '房地产': {
                'upstream': ['土地', '建材', '建筑设计'],
                'midstream': ['房地产开发', '建筑工程'],
                'downstream': ['物业管理', '房产中介', '购房者']
            },
            '证券': {
                'upstream': ['金融数据', 'IT系统', '人才'],
                'midstream': ['证券服务', '投资银行'],
                'downstream': ['企业客户', '个人投资者', '机构投资者']
            }
        }

        chain = industry_chains.get(industry, {
            'upstream': ['原材料'],
            'midstream': ['生产制造'],
            'downstream': ['终端客户']
        })

        return {
            'stock_code': stock_code,
            'stock_name': stock_info.get('name', ''),
            'industry': industry,
            'chain': chain,
            'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def get_concept_stocks(self, concept: str) -> List[Dict]:
        """
        获取某概念的相关股票

        Args:
            concept: 概念名称

        Returns:
            相关股票列表
        """
        stocks = []
        
        for stock_code, sectors in self._stock_sector_matrix.items():
            if concept in sectors:
                stock_info = self._stock_info.get(stock_code, {})
                stocks.append({
                    'code': stock_code,
                    'name': stock_info.get('name', stock_code),
                    'relevance': sectors[concept],
                    'industry': stock_info.get('industry', '未知')
                })
        
        stocks.sort(key=lambda x: x['relevance'], reverse=True)
        return stocks

    def analyze_news_propagation(self, news_sectors: Dict[str, float]) -> Dict:
        """
        分析新闻影响传播路径

        Args:
            news_sectors: 新闻影响的领域

        Returns:
            传播路径分析
        """
        # 直接受影响的股票
        directly_affected = {}
        for sector, impact in news_sectors.items():
            stocks = self._get_stocks_by_sector(sector)
            for stock, relevance in stocks.items():
                if stock not in directly_affected:
                    directly_affected[stock] = 0
                directly_affected[stock] += relevance * impact

        # 间接受影响的股票（通过领域关联）
        indirectly_affected = {}
        for stock in directly_affected.keys():
            related = self.find_related_stocks(stock, top_n=5, min_correlation=0.3)
            for r in related:
                if r['code'] not in directly_affected:
                    if r['code'] not in indirectly_affected:
                        indirectly_affected[r['code']] = 0
                    indirectly_affected[r['code']] += r['correlation'] * 0.5

        # 排序
        direct_list = sorted(directly_affected.items(), key=lambda x: abs(x[1]), reverse=True)
        indirect_list = sorted(indirectly_affected.items(), key=lambda x: abs(x[1]), reverse=True)

        return {
            'direct_impact': [{'code': c, 'impact': round(i, 3)} for c, i in direct_list[:10]],
            'indirect_impact': [{'code': c, 'impact': round(i, 3)} for c, i in indirect_list[:10]],
            'propagation_depth': 2,
            'generate_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }


# 全局实例
relevance_graph = RelevanceGraph()


def get_relevance_graph() -> RelevanceGraph:
    """获取相关性图谱实例"""
    return relevance_graph
