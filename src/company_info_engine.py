#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公司信息引擎 - 公司基本信息、财务数据、公告研报

功能：
1. 获取公司基本信息
2. 获取财务数据
3. 获取公司公告和研报
4. 公司业务分析
"""

import os
import json
import logging
from typing import Optional, Dict, List
from datetime import datetime

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
    logger.info("✓ akshare库已加载，公司信息功能可用")
except ImportError:
    AKSHARE_AVAILABLE = False
    logger.error("✗ akshare库未安装，公司信息功能不可用")


class CompanyInfoEngine:
    """公司信息引擎"""

    def __init__(self, cache_dir: str = "data/cache/company"):
        """
        初始化公司信息引擎

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # 预定义的公司信息（作为备用）
        self._predefined_companies = {
            '600000': {
                'name': '浦发银行',
                'industry': '银行',
                'sector': '金融',
                'market': '上海证券交易所',
                'listing_date': '1999-11-10',
                'business': '公司银行业务、零售银行业务、资金业务等',
                'main_products': ['公司贷款', '个人贷款', '票据贴现', '债券投资'],
                'concepts': ['银行', '上海本地', '沪股通', '融资融券']
            },
            '600519': {
                'name': '贵州茅台',
                'industry': '白酒',
                'sector': '消费',
                'market': '上海证券交易所',
                'listing_date': '2001-08-27',
                'business': '茅台酒及系列酒的生产与销售',
                'main_products': ['贵州茅台酒', '茅台王子酒', '茅台迎宾酒'],
                'concepts': ['白酒', '消费升级', '奢侈品', '机构重仓']
            },
            '600036': {
                'name': '招商银行',
                'industry': '银行',
                'sector': '金融',
                'market': '上海证券交易所',
                'listing_date': '2002-04-09',
                'business': '银行业务、零售银行业务、信用卡业务等',
                'main_products': ['零售贷款', '对公贷款', '信用卡', '理财产品'],
                'concepts': ['银行', '零售银行', '金融科技', '沪股通']
            },
            '000001': {
                'name': '平安银行',
                'industry': '银行',
                'sector': '金融',
                'market': '深圳证券交易所',
                'listing_date': '1991-04-03',
                'business': '公司银行业务、零售银行业务、资金业务等',
                'main_products': ['企业贷款', '个人贷款', '信用卡', '投资银行'],
                'concepts': ['银行', '平安集团', '金融科技', '深股通']
            },
            '000002': {
                'name': '万科A',
                'industry': '房地产',
                'sector': '地产',
                'market': '深圳证券交易所',
                'listing_date': '1991-01-29',
                'business': '房地产开发与经营、物业管理等',
                'main_products': ['住宅开发', '商业地产', '物业管理', '物流地产'],
                'concepts': ['房地产', '物业管理', '深圳本地', '深股通']
            },
            '600690': {
                'name': '海尔智家',
                'industry': '家电',
                'sector': '消费',
                'market': '上海证券交易所',
                'listing_date': '1993-11-19',
                'business': '家用电器研发、生产和销售',
                'main_products': ['冰箱', '洗衣机', '空调', '热水器'],
                'concepts': ['家电', '智能家居', '物联网', 'A+H股']
            },
            '000858': {
                'name': '五粮液',
                'industry': '白酒',
                'sector': '消费',
                'market': '深圳证券交易所',
                'listing_date': '1998-04-21',
                'business': '五粮液酒及系列酒的生产与销售',
                'main_products': ['五粮液', '五粮春', '五粮醇'],
                'concepts': ['白酒', '消费升级', '深股通', '机构重仓']
            },
            '600030': {
                'name': '中信证券',
                'industry': '证券',
                'sector': '金融',
                'market': '上海证券交易所',
                'listing_date': '2003-01-06',
                'business': '证券经纪、投资银行、资产管理等',
                'main_products': ['经纪业务', '投行业务', '资管业务', '自营业务'],
                'concepts': ['证券', '券商', '金融', '沪股通']
            },
            '600009': {
                'name': '上海机场',
                'industry': '机场',
                'sector': '交通',
                'market': '上海证券交易所',
                'listing_date': '1998-02-18',
                'business': '航空地面服务、商业租赁等',
                'main_products': ['航空服务', '商业租赁', '地面服务'],
                'concepts': ['机场', '上海本地', '免税店', '消费']
            },
            '601318': {
                'name': '中国平安',
                'industry': '保险',
                'sector': '金融',
                'market': '上海证券交易所',
                'listing_date': '2007-03-01',
                'business': '保险、银行、投资等综合金融服务',
                'main_products': ['寿险', '财险', '银行业务', '投资管理'],
                'concepts': ['保险', '金融', 'A+H股', '机构重仓']
            }
        }
        
        # 行业分类映射
        self._industry_sectors = {
            '银行': '金融',
            '证券': '金融',
            '保险': '金融',
            '白酒': '消费',
            '家电': '消费',
            '房地产': '地产',
            '机场': '交通',
            '汽车': '制造',
            '医药': '医疗',
            '科技': '科技',
            '能源': '能源'
        }

    def is_available(self) -> bool:
        """检查引擎是否可用"""
        return AKSHARE_AVAILABLE

    def get_company_info(self, stock_code: str, use_cache: bool = True) -> Dict:
        """
        获取公司基本信息

        Args:
            stock_code: 股票代码
            use_cache: 是否使用缓存

        Returns:
            公司信息字典
        """
        # 尝试从缓存读取
        if use_cache:
            cached = self._load_cache(stock_code, 'info')
            if cached:
                return cached

        # 先使用预定义信息
        if stock_code in self._predefined_companies:
            company_info = self._predefined_companies[stock_code].copy()
        else:
            company_info = {
                'code': stock_code,
                'name': stock_code,
                'industry': '未知',
                'sector': '未知',
                'market': '未知',
                'listing_date': '未知',
                'business': '暂无信息',
                'main_products': [],
                'concepts': []
            }

        # 尝试从akshare获取更详细信息
        if AKSHARE_AVAILABLE:
            try:
                # 获取个股信息
                df = ak.stock_individual_info_em(symbol=stock_code)
                if not df.empty:
                    info_dict = dict(zip(df['item'], df['value']))
                    company_info.update({
                        'name': info_dict.get('股票简称', company_info.get('name')),
                        'industry': info_dict.get('行业', company_info.get('industry')),
                        'listing_date': info_dict.get('上市时间', company_info.get('listing_date')),
                        'market': info_dict.get('市场', company_info.get('market')),
                        'business': info_dict.get('主营业务', company_info.get('business')),
                    })
            except Exception as e:
                logger.warning(f"获取公司基本信息失败: {e}")

        # 确保行业对应板块
        if company_info.get('industry') and not company_info.get('sector'):
            industry = company_info.get('industry', '')
            for key, sector in self._industry_sectors.items():
                if key in industry:
                    company_info['sector'] = sector
                    break

        # 添加缓存时间戳
        company_info['cache_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # 保存缓存
        self._save_cache(stock_code, 'info', company_info)
        
        return company_info

    def get_financial_data(self, stock_code: str, use_cache: bool = True) -> Dict:
        """
        获取财务数据

        Args:
            stock_code: 股票代码
            use_cache: 是否使用缓存

        Returns:
            财务数据字典
        """
        # 尝试从缓存读取
        if use_cache:
            cached = self._load_cache(stock_code, 'financial')
            if cached:
                return cached

        financial_data = {
            'code': stock_code,
            'indicators': {},
            'recent_reports': []
        }

        if AKSHARE_AVAILABLE:
            try:
                # 获取财务指标
                df = ak.stock_financial_analysis_indicator(symbol=stock_code)
                if not df.empty:
                    # 取最近几期数据
                    recent = df.head(4)
                    indicators = {}
                    for _, row in recent.iterrows():
                        date = str(row.get('日期', ''))
                        indicators[date] = {
                            'roe': self._safe_float(row.get('净资产收益率(%)')),
                            'roa': self._safe_float(row.get('总资产净利率(%)')),
                            'gross_margin': self._safe_float(row.get('销售毛利率(%)')),
                            'net_margin': self._safe_float(row.get('销售净利率(%)')),
                            'debt_ratio': self._safe_float(row.get('资产负债率(%)')),
                            'current_ratio': self._safe_float(row.get('流动比率')),
                            'quick_ratio': self._safe_float(row.get('速动比率'))
                        }
                    financial_data['indicators'] = indicators
                    financial_data['recent_reports'] = list(indicators.keys())
            except Exception as e:
                logger.warning(f"获取财务数据失败: {e}")

        financial_data['cache_time'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        self._save_cache(stock_code, 'financial', financial_data)
        
        return financial_data

    def get_company_announcements(self, stock_code: str, limit: int = 20) -> List[Dict]:
        """
        获取公司公告

        Args:
            stock_code: 股票代码
            limit: 返回数量限制

        Returns:
            公告列表
        """
        announcements = []

        if AKSHARE_AVAILABLE:
            try:
                # 获取公告列表
                df = ak.stock_notice_report(symbol=stock_code)
                if not df.empty:
                    for _, row in df.head(limit).iterrows():
                        announcements.append({
                            'title': str(row.get('公告标题', '')),
                            'type': str(row.get('公告类型', '')),
                            'date': str(row.get('公告日期', '')),
                            'url': str(row.get('公告链接', ''))
                        })
            except Exception as e:
                logger.warning(f"获取公司公告失败: {e}")

        return announcements

    def get_research_reports(self, stock_code: str, limit: int = 10) -> List[Dict]:
        """
        获取研报信息

        Args:
            stock_code: 股票代码
            limit: 返回数量限制

        Returns:
            研报列表
        """
        reports = []

        if AKSHARE_AVAILABLE:
            try:
                # 获取研报
                df = ak.stock_research_report_em(symbol=stock_code)
                if not df.empty:
                    for _, row in df.head(limit).iterrows():
                        reports.append({
                            'title': str(row.get('标题', '')),
                            'institution': str(row.get('机构', '')),
                            'rating': str(row.get('评级', '')),
                            'date': str(row.get('日期', '')),
                            'author': str(row.get('研究员', ''))
                        })
            except Exception as e:
                logger.warning(f"获取研报失败: {e}")

        return reports

    def analyze_business_structure(self, stock_code: str) -> Dict:
        """
        分析公司业务结构

        Args:
            stock_code: 股票代码

        Returns:
            业务结构分析结果
        """
        company_info = self.get_company_info(stock_code)
        
        # 获取主营业务产品
        main_products = company_info.get('main_products', [])
        
        # 业务结构分析
        business_analysis = {
            'code': stock_code,
            'name': company_info.get('name', ''),
            'industry': company_info.get('industry', ''),
            'sector': company_info.get('sector', ''),
            'main_business': company_info.get('business', ''),
            'products': main_products,
            'concepts': company_info.get('concepts', []),
            'industry_position': self._analyze_industry_position(company_info),
            'business_risk': self._analyze_business_risk(company_info),
            'growth_potential': self._analyze_growth_potential(company_info)
        }

        return business_analysis

    def _analyze_industry_position(self, company_info: Dict) -> Dict:
        """分析行业地位"""
        industry = company_info.get('industry', '')
        concepts = company_info.get('concepts', [])
        
        position = '一般'
        if '龙头' in str(concepts) or '领涨' in str(concepts):
            position = '龙头'
        elif '头部' in str(concepts):
            position = '头部'
        elif industry in ['银行', '保险', '证券']:
            # 金融行业通常竞争激烈
            position = '主流'

        return {
            'position': position,
            'description': f"在{industry}行业中处于{position}地位"
        }

    def _analyze_business_risk(self, company_info: Dict) -> List[str]:
        """分析业务风险"""
        risks = []
        industry = company_info.get('industry', '')
        
        if industry == '房地产':
            risks.append('政策调控风险')
            risks.append('周期性风险')
        elif industry == '银行':
            risks.append('信用风险')
            risks.append('利率风险')
        elif industry == '白酒':
            risks.append('消费降级风险')
        elif industry == '家电':
            risks.append('原材料价格风险')
            risks.append('竞争加剧风险')
        
        if not risks:
            risks.append('市场波动风险')
        
        return risks

    def _analyze_growth_potential(self, company_info: Dict) -> Dict:
        """分析成长潜力"""
        concepts = company_info.get('concepts', [])
        
        potential = '中等'
        growth_drivers = []
        
        if '消费升级' in concepts:
            growth_drivers.append('消费升级趋势')
            potential = '较高'
        if '金融科技' in concepts:
            growth_drivers.append('数字化转型')
            potential = '较高'
        if '新能源' in concepts or '智能' in str(concepts):
            growth_drivers.append('技术创新')
            potential = '较高'

        return {
            'potential': potential,
            'drivers': growth_drivers if growth_drivers else ['行业自然增长']
        }

    def _safe_float(self, value) -> Optional[float]:
        """安全转换为浮点数"""
        try:
            if pd.isna(value):
                return None
            return float(value)
        except:
            return None

    def _load_cache(self, stock_code: str, data_type: str) -> Optional[Dict]:
        """加载缓存"""
        cache_file = os.path.join(self.cache_dir, f"{stock_code}_{data_type}.json")
        
        if not os.path.exists(cache_file):
            return None
        
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 检查缓存时间（超过24小时过期）
            cache_time = data.get('cache_time')
            if cache_time:
                cache_dt = datetime.strptime(cache_time, '%Y-%m-%d %H:%M:%S')
                if (datetime.now() - cache_dt).days >= 1:
                    return None
            
            return data
        except Exception as e:
            logger.warning(f"加载缓存失败: {e}")
            return None

    def _save_cache(self, stock_code: str, data_type: str, data: Dict):
        """保存缓存"""
        cache_file = os.path.join(self.cache_dir, f"{stock_code}_{data_type}.json")
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存缓存失败: {e}")

    def get_sector_companies(self, sector: str, limit: int = 20) -> List[Dict]:
        """
        获取同板块公司列表

        Args:
            sector: 板块名称
            limit: 返回数量限制

        Returns:
            同板块公司列表
        """
        companies = []
        
        for code, info in self._predefined_companies.items():
            if info.get('sector') == sector or info.get('industry') == sector:
                companies.append({
                    'code': code,
                    'name': info.get('name'),
                    'industry': info.get('industry'),
                    'sector': info.get('sector')
                })
        
        return companies[:limit]


# 全局实例
company_info_engine = CompanyInfoEngine()


def get_company_info_engine() -> CompanyInfoEngine:
    """获取公司信息引擎实例"""
    return company_info_engine
