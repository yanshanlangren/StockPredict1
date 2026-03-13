#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练多模态预测模型

使用所有股票数据训练全局多模态预测模型，融合：
- 新闻特征
- 领域影响
- 技术指标
- 相关性矩阵
"""

import os
import sys
import argparse
import logging
from datetime import datetime, timedelta
import json

import numpy as np
import pandas as pd

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 导入模块
from src.data_source_manager import DataSourceManager, DataSource
from src.news_crawler import get_news_crawler
from src.news_impact_analyzer import get_news_impact_analyzer
from src.relevance_graph import get_relevance_graph
from src.multimodal_model import get_multimodal_predictor

# 领域列表
SECTORS = [
    '银行', '证券', '保险', '地产', '汽车', 
    '科技', '医药', '消费', '能源', '军工',
    '基建', '传媒', '教育', '农业', '交通'
]


def generate_synthetic_news(stock_code: str, df: pd.DataFrame) -> list:
    """生成合成新闻数据（用于训练）"""
    news_list = []
    
    if df.empty:
        return news_list
    
    # 根据价格走势生成情感
    close = df['close']
    returns = close.pct_change().dropna()
    
    if len(returns) == 0:
        return news_list
    
    avg_return = returns.tail(5).mean()
    sentiment = 0.3 if avg_return > 0 else -0.3
    
    # 生成几条新闻
    for i in range(3):
        news_list.append({
            'title': f'{stock_code}相关新闻{i+1}',
            'content': '训练用合成新闻',
            'sentiment': sentiment + np.random.uniform(-0.2, 0.2),
            'importance': np.random.uniform(0.4, 0.8),
            'publish_time': (datetime.now() - timedelta(hours=i*8)).strftime('%Y-%m-%d %H:%M:%S'),
            'categories': []
        })
    
    return news_list


def get_stock_sector(stock_code: str) -> str:
    """获取股票所属行业"""
    sector_map = {
        '600000': '银行', '600036': '银行', '000001': '银行',
        '600519': '消费', '000858': '消费',
        '000002': '地产', '600690': '消费',
        '600030': '证券', '601318': '保险',
        '600009': '交通'
    }
    return sector_map.get(stock_code, '科技')


def prepare_training_data(data_manager: DataSourceManager, 
                          stock_list: pd.DataFrame,
                          n_stocks: int = 50,
                          days: int = 200) -> dict:
    """
    准备训练数据
    
    Args:
        data_manager: 数据源管理器
        stock_list: 股票列表
        n_stocks: 使用的股票数量
        days: 历史数据天数
        
    Returns:
        训练数据字典
    """
    logger.info(f"开始准备训练数据，目标股票数: {n_stocks}")
    
    # 初始化
    predictor = get_multimodal_predictor()
    news_crawler = get_news_crawler()
    news_analyzer = get_news_impact_analyzer()
    relevance_graph = get_relevance_graph()
    
    # 获取相关性矩阵
    try:
        matrix_data = relevance_graph.get_relevance_matrix()
        relevance_matrix = np.array(matrix_data['matrix'])
        stock_codes_list = matrix_data['stock_codes']
    except:
        relevance_matrix = None
        stock_codes_list = []
    
    # 数据收集
    all_news_features = []
    all_sector_features = []
    all_tech_features = []
    all_relevance_features = []
    all_labels = []
    
    stocks_processed = 0
    
    for idx, stock in stock_list.head(n_stocks).iterrows():
        stock_code = str(stock['code'])
        
        try:
            # 获取K线数据
            df = data_manager.get_stock_kline(stock_code, days=days)
            
            if df.empty or len(df) < 60:
                logger.debug(f"跳过 {stock_code}: 数据不足")
                continue
            
            # 获取新闻数据（如果可用，否则使用合成数据）
            try:
                news_list = news_crawler.get_news(stock_code=stock_code, limit=10)
                if not news_list:
                    news_list = generate_synthetic_news(stock_code, df)
            except:
                news_list = generate_synthetic_news(stock_code, df)
            
            # 获取领域影响
            try:
                sector_impact = news_analyzer.get_sector_impact_vector(news_list)
            except:
                sector_impact = {}
            
            # 确保行业有值
            if not sector_impact:
                sector = get_stock_sector(stock_code)
                sector_impact = {sector: 0.5}
            
            # 生成训练样本（滑动窗口）
            window_size = 20
            predict_days = 5
            
            for i in range(window_size, len(df) - predict_days):
                # 历史数据窗口
                hist_df = df.iloc[i-window_size:i+1].copy()
                
                # 未来收益作为标签
                future_return = (df['close'].iloc[i+predict_days] / df['close'].iloc[i] - 1)
                label = 1 if future_return > 0 else 0
                
                # 编码特征
                news_features = predictor.encode_news_features(news_list)
                sector_features = predictor.encode_sector_features(sector_impact)
                tech_features = predictor.encode_technical_features(hist_df)
                
                # 相关性特征
                if relevance_matrix is not None and stock_code in stock_codes_list:
                    stock_idx = stock_codes_list.index(stock_code)
                    rel_features = predictor.encode_relevance_features(relevance_matrix, stock_idx)
                else:
                    rel_features = np.zeros((1, predictor.relevance_dim))
                
                all_news_features.append(news_features.flatten())
                all_sector_features.append(sector_features.flatten())
                all_tech_features.append(tech_features.flatten())
                all_relevance_features.append(rel_features.flatten())
                all_labels.append(label)
            
            stocks_processed += 1
            logger.info(f"处理股票 {stock_code} 完成 ({stocks_processed}/{n_stocks})")
            
        except Exception as e:
            logger.warning(f"处理股票 {stock_code} 失败: {e}")
            continue
    
    logger.info(f"训练数据准备完成，处理 {stocks_processed} 只股票")
    
    # 检查数据量
    if len(all_labels) == 0:
        raise ValueError("未能生成任何训练样本")
    
    # 转换为数组
    training_data = {
        'news_features': np.array(all_news_features),
        'sector_features': np.array(all_sector_features),
        'tech_features': np.array(all_tech_features),
        'relevance_features': np.array(all_relevance_features),
        'labels': np.array(all_labels)
    }
    
    # 统计
    positive = sum(all_labels)
    negative = len(all_labels) - positive
    logger.info(f"训练样本统计: 正样本={positive}, 负样本={negative}")
    
    return training_data


def train_model(training_data: dict, epochs: int = 50) -> dict:
    """训练模型"""
    predictor = get_multimodal_predictor()
    
    logger.info("开始训练多模态模型...")
    result = predictor.train(training_data, epochs=epochs, batch_size=32)
    
    return result


def main():
    parser = argparse.ArgumentParser(description='训练多模态预测模型')
    parser.add_argument('--stocks', type=int, default=50, help='训练使用的股票数量')
    parser.add_argument('--days', type=int, default=200, help='历史数据天数')
    parser.add_argument('--epochs', type=int, default=50, help='训练轮数')
    
    args = parser.parse_args()
    
    logger.info("=" * 50)
    logger.info("多模态预测模型训练")
    logger.info("=" * 50)
    logger.info(f"参数: stocks={args.stocks}, days={args.days}, epochs={args.epochs}")
    
    # 初始化数据源
    logger.info("初始化数据源...")
    data_manager = DataSourceManager(preferred_source=DataSource.TENCENT)
    
    # 获取股票列表
    logger.info("获取股票列表...")
    stock_list = data_manager.get_stock_list()
    
    if stock_list.empty:
        logger.error("无法获取股票列表")
        return
    
    logger.info(f"获取到 {len(stock_list)} 只股票")
    
    # 准备训练数据
    logger.info("准备训练数据...")
    training_data = prepare_training_data(
        data_manager, 
        stock_list,
        n_stocks=args.stocks,
        days=args.days
    )
    
    # 训练模型
    logger.info("开始训练...")
    result = train_model(training_data, epochs=args.epochs)
    
    if result.get('success'):
        logger.info("=" * 50)
        logger.info("✓ 训练完成！")
        logger.info(f"模型保存至: {result.get('model_path')}")
        logger.info(f"最终指标: {result.get('metadata', {}).get('final_metrics')}")
        logger.info("=" * 50)
    else:
        logger.error(f"训练失败: {result.get('message')}")


if __name__ == '__main__':
    main()
