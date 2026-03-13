#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
新闻影响分析引擎 - 分析新闻对股价的影响

功能：
1. 新闻情感分析
2. 领域影响量化
3. 新闻-股票关联分析
4. 影响时间衰减模型
"""

import os
import json
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timedelta
import math

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NewsImpactAnalyzer:
    """新闻影响分析引擎"""

    def __init__(self):
        """初始化分析引擎"""
        # 领域-股票相关性权重
        self.sector_stock_weights = {
            '银行': {
                '600000': 0.95, '600036': 0.90, '000001': 0.88, '601398': 0.95,
                'default': 0.3
            },
            '证券': {
                '600030': 0.95, '601211': 0.90, '600837': 0.85,
                'default': 0.2
            },
            '白酒': {
                '600519': 0.95, '000858': 0.90, '000568': 0.70,
                'default': 0.2
            },
            '地产': {
                '000002': 0.90, '600048': 0.88, '001979': 0.75,
                'default': 0.3
            },
            '家电': {
                '600690': 0.90, '000333': 0.88, '000651': 0.80,
                'default': 0.2
            },
            '保险': {
                '601318': 0.95, '601601': 0.85, '601628': 0.85,
                'default': 0.2
            }
        }
        
        # 事件类型影响系数
        self.event_impact_factors = {
            '央行降准': {'impact': 0.8, 'duration': 5, 'sectors': ['银行', '证券']},
            '央行降息': {'impact': 0.7, 'duration': 5, 'sectors': ['银行', '地产']},
            'IPO': {'impact': 0.3, 'duration': 2, 'sectors': ['证券']},
            '财报发布': {'impact': 0.5, 'duration': 3, 'sectors': []},
            '政策利好': {'impact': 0.6, 'duration': 4, 'sectors': []},
            '政策利空': {'impact': -0.6, 'duration': 4, 'sectors': []},
            '并购重组': {'impact': 0.7, 'duration': 7, 'sectors': []},
            '业绩预增': {'impact': 0.5, 'duration': 3, 'sectors': []},
            '业绩预减': {'impact': -0.5, 'duration': 3, 'sectors': []},
            '高管增持': {'impact': 0.3, 'duration': 2, 'sectors': []},
            '高管减持': {'impact': -0.3, 'duration': 2, 'sectors': []},
        }
        
        # 时间衰减参数
        self.decay_half_life = 24  # 半衰期（小时）

    def analyze_news_impact(self, news: Dict, stock_code: str) -> Dict:
        """
        分析新闻对特定股票的影响

        Args:
            news: 新闻数据
            stock_code: 股票代码

        Returns:
            影响分析结果
        """
        # 基础影响分数
        base_sentiment = news.get('sentiment', 0)
        importance = news.get('importance', 0.5)
        
        # 获取新闻影响的领域
        affected_sectors = news.get('affected_sectors', {})
        
        # 计算领域相关性
        sector_relevance = self._calculate_sector_relevance(affected_sectors, stock_code)
        
        # 计算事件类型影响
        event_impact = self._calculate_event_impact(news, stock_code)
        
        # 计算时间衰减
        time_decay = self._calculate_time_decay(news.get('publish_time', ''))
        
        # 综合影响分数
        # Impact = (情感分数 × 重要性 × 领域相关性 + 事件影响) × 时间衰减
        impact_score = (base_sentiment * importance * sector_relevance + event_impact) * time_decay
        
        # 影响方向
        direction = 'positive' if impact_score > 0.1 else ('negative' if impact_score < -0.1 else 'neutral')
        
        # 预期影响幅度（百分比）
        expected_change = abs(impact_score) * 5  # 最大5%的影响
        
        # 影响持续时间
        duration = self._estimate_impact_duration(news, impact_score)
        
        return {
            'stock_code': stock_code,
            'news_id': news.get('news_id'),
            'news_title': news.get('title', ''),
            'impact_score': round(impact_score, 4),
            'impact_direction': direction,
            'expected_change_pct': round(expected_change, 2),
            'confidence': round(min(abs(impact_score) * 2, 1.0), 2),
            'sector_relevance': round(sector_relevance, 3),
            'time_decay': round(time_decay, 3),
            'duration_days': duration,
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

    def analyze_batch_news_impact(self, news_list: List[Dict], stock_code: str) -> Dict:
        """
        批量分析新闻影响

        Args:
            news_list: 新闻列表
            stock_code: 股票代码

        Returns:
            综合影响分析结果
        """
        if not news_list:
            return {
                'stock_code': stock_code,
                'total_news': 0,
                'total_impact': 0,
                'impact_details': []
            }

        impact_details = []
        total_impact = 0
        positive_impacts = []
        negative_impacts = []

        for news in news_list:
            impact = self.analyze_news_impact(news, stock_code)
            impact_details.append(impact)
            
            impact_score = impact['impact_score']
            total_impact += impact_score
            
            if impact_score > 0.1:
                positive_impacts.append(impact)
            elif impact_score < -0.1:
                negative_impacts.append(impact)

        # 综合评估
        overall_sentiment = 'positive' if total_impact > 0.2 else (
            'negative' if total_impact < -0.2 else 'neutral'
        )

        # 按影响分数排序
        impact_details.sort(key=lambda x: abs(x['impact_score']), reverse=True)

        return {
            'stock_code': stock_code,
            'total_news': len(news_list),
            'total_impact': round(total_impact, 4),
            'overall_sentiment': overall_sentiment,
            'positive_count': len(positive_impacts),
            'negative_count': len(negative_impacts),
            'neutral_count': len(news_list) - len(positive_impacts) - len(negative_impacts),
            'avg_impact': round(total_impact / len(news_list), 4) if news_list else 0,
            'top_positive': positive_impacts[:3] if positive_impacts else [],
            'top_negative': negative_impacts[:3] if negative_impacts else [],
            'impact_details': impact_details[:20]  # 只返回前20条详情
        }

    def _calculate_sector_relevance(self, affected_sectors: Dict[str, float], 
                                   stock_code: str) -> float:
        """
        计算股票与新闻影响领域的相关性

        Args:
            affected_sectors: 新闻影响的领域
            stock_code: 股票代码

        Returns:
            相关性分数 (0-1)
        """
        if not affected_sectors:
            return 0.3  # 默认相关性

        total_relevance = 0
        for sector, impact in affected_sectors.items():
            # 查找该领域的股票权重
            sector_weights = self.sector_stock_weights.get(sector, {})
            stock_weight = sector_weights.get(stock_code, sector_weights.get('default', 0.2))
            total_relevance += stock_weight * impact

        # 归一化
        max_possible = len(affected_sectors)
        return min(total_relevance / max_possible if max_possible > 0 else 0, 1.0)

    def _calculate_event_impact(self, news: Dict, stock_code: str) -> float:
        """
        计算特定事件类型的影响

        Args:
            news: 新闻数据
            stock_code: 股票代码

        Returns:
            事件影响分数
        """
        title = news.get('title', '')
        content = news.get('content', '')
        text = title + ' ' + content
        
        event_impact = 0
        for event_type, event_info in self.event_impact_factors.items():
            if event_type in text:
                impact = event_info['impact']
                affected_sectors = event_info.get('sectors', [])
                
                # 检查股票是否在受影响的领域
                if affected_sectors:
                    sector_relevance = 0
                    for sector in affected_sectors:
                        sector_weights = self.sector_stock_weights.get(sector, {})
                        relevance = sector_weights.get(stock_code, sector_weights.get('default', 0.2))
                        sector_relevance = max(sector_relevance, relevance)
                    impact *= sector_relevance
                
                event_impact += impact

        return event_impact

    def _calculate_time_decay(self, publish_time: str) -> float:
        """
        计算时间衰减因子

        Args:
            publish_time: 发布时间

        Returns:
            时间衰减因子 (0-1)
        """
        if not publish_time:
            return 0.5

        try:
            pub_dt = datetime.strptime(publish_time, '%Y-%m-%d %H:%M:%S')
            hours_elapsed = (datetime.now() - pub_dt).total_seconds() / 3600
            
            # 指数衰减
            decay = math.exp(-hours_elapsed * math.log(2) / self.decay_half_life)
            return max(0.1, min(1.0, decay))
        except:
            return 0.5

    def _estimate_impact_duration(self, news: Dict, impact_score: float) -> int:
        """
        估计影响持续时间

        Args:
            news: 新闻数据
            impact_score: 影响分数

        Returns:
            预计影响持续天数
        """
        base_duration = 2  # 基础持续天数
        
        # 根据重要性调整
        importance = news.get('importance', 0.5)
        if importance > 0.8:
            base_duration += 3
        elif importance > 0.6:
            base_duration += 2
        
        # 根据影响分数调整
        if abs(impact_score) > 0.5:
            base_duration += 2
        
        # 检查是否有特定事件类型
        title = news.get('title', '')
        for event_type, event_info in self.event_impact_factors.items():
            if event_type in title:
                base_duration = max(base_duration, event_info.get('duration', 2))
                break

        return min(base_duration, 10)  # 最大10天

    def get_sector_impact_vector(self, news_list: List[Dict]) -> Dict[str, float]:
        """
        获取领域影响向量

        Args:
            news_list: 新闻列表

        Returns:
            {领域: 影响分数} 字典
        """
        sector_impacts = {}
        
        for news in news_list:
            affected_sectors = news.get('affected_sectors', {})
            sentiment = news.get('sentiment', 0)
            importance = news.get('importance', 0.5)
            time_decay = self._calculate_time_decay(news.get('publish_time', ''))
            
            for sector, base_impact in affected_sectors.items():
                if sector not in sector_impacts:
                    sector_impacts[sector] = 0
                
                impact = base_impact * sentiment * importance * time_decay
                sector_impacts[sector] += impact

        return dict(sorted(sector_impacts.items(), key=lambda x: abs(x[1]), reverse=True))

    def generate_impact_report(self, news_list: List[Dict], 
                               stock_code: str) -> Dict:
        """
        生成综合影响报告

        Args:
            news_list: 新闻列表
            stock_code: 股票代码

        Returns:
            影响报告
        """
        # 批量分析影响
        batch_analysis = self.analyze_batch_news_impact(news_list, stock_code)
        
        # 领域影响向量
        sector_impact = self.get_sector_impact_vector(news_list)
        
        # 影响趋势
        impact_trend = self._analyze_impact_trend(news_list, stock_code)
        
        # 风险提示
        risk_warnings = self._generate_risk_warnings(batch_analysis, sector_impact)

        return {
            'stock_code': stock_code,
            'report_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'summary': {
                'total_news': batch_analysis['total_news'],
                'overall_sentiment': batch_analysis['overall_sentiment'],
                'total_impact': batch_analysis['total_impact'],
                'confidence': batch_analysis.get('avg_impact', 0)
            },
            'sentiment_distribution': {
                'positive': batch_analysis['positive_count'],
                'negative': batch_analysis['negative_count'],
                'neutral': batch_analysis['neutral_count']
            },
            'sector_impact': sector_impact,
            'impact_trend': impact_trend,
            'top_impacts': batch_analysis['impact_details'][:5],
            'risk_warnings': risk_warnings,
            'recommendation': self._generate_recommendation(batch_analysis)
        }

    def _analyze_impact_trend(self, news_list: List[Dict], 
                             stock_code: str) -> Dict:
        """分析影响趋势"""
        if len(news_list) < 2:
            return {'trend': 'stable', 'description': '数据不足'}

        # 按时间排序
        sorted_news = sorted(news_list, 
                           key=lambda x: x.get('publish_time', ''), 
                           reverse=True)
        
        # 分成两半比较
        mid = len(sorted_news) // 2
        recent_news = sorted_news[:mid]
        older_news = sorted_news[mid:]
        
        recent_impact = sum(self.analyze_news_impact(n, stock_code)['impact_score'] 
                          for n in recent_news)
        older_impact = sum(self.analyze_news_impact(n, stock_code)['impact_score'] 
                         for n in older_news)

        if recent_impact > older_impact + 0.2:
            trend = 'improving'
            description = '近期影响趋于正面'
        elif recent_impact < older_impact - 0.2:
            trend = 'deteriorating'
            description = '近期影响趋于负面'
        else:
            trend = 'stable'
            description = '影响保持稳定'

        return {'trend': trend, 'description': description}

    def _generate_risk_warnings(self, batch_analysis: Dict, 
                                sector_impact: Dict) -> List[str]:
        """生成风险提示"""
        warnings = []
        
        if batch_analysis['negative_count'] > batch_analysis['positive_count'] * 2:
            warnings.append('负面新闻数量明显多于正面新闻，需关注潜在风险')
        
        if batch_analysis['total_impact'] < -0.5:
            warnings.append('综合影响分数较低，建议谨慎操作')
        
        negative_sectors = [s for s, i in sector_impact.items() if i < -0.3]
        if negative_sectors:
            warnings.append(f'以下领域存在较大负面影响: {", ".join(negative_sectors[:3])}')
        
        return warnings

    def _generate_recommendation(self, batch_analysis: Dict) -> str:
        """生成投资建议"""
        sentiment = batch_analysis['overall_sentiment']
        total_impact = batch_analysis['total_impact']
        
        if sentiment == 'positive' and total_impact > 0.3:
            return '新闻面整体偏正面，可适当关注'
        elif sentiment == 'negative' and total_impact < -0.3:
            return '新闻面整体偏负面，建议谨慎'
        else:
            return '新闻面中性，建议结合技术面综合判断'


# 全局实例
news_impact_analyzer = NewsImpactAnalyzer()


def get_news_impact_analyzer() -> NewsImpactAnalyzer:
    """获取新闻影响分析引擎实例"""
    return news_impact_analyzer
