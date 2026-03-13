#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态预测模型 - 融合新闻、技术指标、相关性矩阵的预测模型

功能：
1. 新闻向量编码
2. 领域影响向量生成
3. 多模态特征融合
4. 股票预测
"""

import os
import json
import logging
from typing import Optional, Dict, List, Tuple
from datetime import datetime
import hashlib

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入TensorFlow
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, models
    TENSORFLOW_AVAILABLE = True
    logger.info("✓ TensorFlow已加载，多模态模型功能可用")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow未安装，使用简化版预测模型")


class MultiModalPredictor:
    """多模态股票预测模型"""

    def __init__(self, model_dir: str = "models"):
        """
        初始化多模态预测器

        Args:
            model_dir: 模型存储目录
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.model_name = "multimodal_stock_predictor"
        
        # 特征维度
        self.news_feature_dim = 64      # 新闻特征维度
        self.sector_feature_dim = 32    # 领域特征维度
        self.tech_feature_dim = 12      # 技术指标维度
        self.relevance_dim = 16         # 相关性特征维度
        
        # 领域列表
        self.sectors = [
            '银行', '证券', '保险', '地产', '汽车', 
            '科技', '医药', '消费', '能源', '军工',
            '基建', '传媒', '教育', '农业', '交通'
        ]
        
        # 尝试加载模型
        self._load_model()

    def is_available(self) -> bool:
        """检查模型是否可用"""
        return True  # 即使没有TensorFlow也可用简化版

    def _load_model(self):
        """加载模型"""
        if not TENSORFLOW_AVAILABLE:
            logger.info("使用简化版预测模型（无需TensorFlow）")
            return

        model_path = os.path.join(self.model_dir, f"{self.model_name}.keras")
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                logger.info(f"✓ 多模态模型加载成功: {model_path}")
            except Exception as e:
                logger.warning(f"加载模型失败: {e}")

    def _build_model(self):
        """构建模型架构"""
        if not TENSORFLOW_AVAILABLE:
            return

        # 新闻特征输入
        news_input = layers.Input(shape=(self.news_feature_dim,), name='news_features')
        news_dense = layers.Dense(32, activation='relu')(news_input)
        
        # 领域影响特征输入
        sector_input = layers.Input(shape=(self.sector_feature_dim,), name='sector_features')
        sector_dense = layers.Dense(16, activation='relu')(sector_input)
        
        # 技术指标输入
        tech_input = layers.Input(shape=(self.tech_feature_dim,), name='tech_features')
        tech_dense = layers.Dense(32, activation='relu')(tech_input)
        
        # 相关性特征输入
        relevance_input = layers.Input(shape=(self.relevance_dim,), name='relevance_features')
        relevance_dense = layers.Dense(8, activation='relu')(relevance_input)
        
        # 多模态融合
        concat = layers.Concatenate()([news_dense, sector_dense, tech_dense, relevance_dense])
        
        # 注意力机制
        attention = layers.MultiHeadAttention(num_heads=4, key_dim=32)(concat, concat)
        attention = layers.LayerNormalization()(attention + concat)
        
        # 全连接层
        x = layers.Dense(128, activation='relu')(attention)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        
        # 输出层
        output = layers.Dense(1, activation='sigmoid', name='prediction')(x)
        
        # 构建模型
        self.model = models.Model(
            inputs=[news_input, sector_input, tech_input, relevance_input],
            outputs=output
        )
        
        self.model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        logger.info("多模态模型构建完成")

    def encode_news_features(self, news_list: List[Dict]) -> np.ndarray:
        """
        编码新闻特征向量

        Args:
            news_list: 新闻列表

        Returns:
            新闻特征向量 (1, news_feature_dim)
        """
        features = np.zeros(self.news_feature_dim)
        
        if not news_list:
            return features.reshape(1, -1)

        # 统计特征
        sentiments = [n.get('sentiment', 0) for n in news_list]
        importances = [n.get('importance', 0.5) for n in news_list]
        
        # 基础统计
        features[0] = np.mean(sentiments)  # 平均情感
        features[1] = np.std(sentiments)   # 情感波动
        features[2] = np.mean(importances) # 平均重要性
        features[3] = len(news_list) / 50  # 新闻数量归一化
        
        # 情感分布
        features[4] = sum(1 for s in sentiments if s > 0.2) / len(sentiments)  # 正面比例
        features[5] = sum(1 for s in sentiments if s < -0.2) / len(sentiments) # 负面比例
        
        # 时间加权情感
        current_time = datetime.now()
        time_weights = []
        for news in news_list:
            pub_time = news.get('publish_time', '')
            if pub_time:
                try:
                    dt = datetime.strptime(pub_time, '%Y-%m-%d %H:%M:%S')
                    hours_ago = (current_time - dt).total_seconds() / 3600
                    weight = np.exp(-hours_ago / 24)  # 24小时半衰期
                except:
                    weight = 0.5
            else:
                weight = 0.5
            time_weights.append(weight)
        
        weighted_sentiment = np.average(sentiments, weights=time_weights) if time_weights else 0
        features[6] = weighted_sentiment
        
        # 类别分布（前20维）
        categories = {}
        for news in news_list:
            for cat in news.get('categories', []):
                categories[cat] = categories.get(cat, 0) + 1
        
        for i, cat in enumerate(list(categories.keys())[:20]):
            features[7 + i] = categories[cat] / len(news_list)
        
        return features.reshape(1, -1)

    def encode_sector_features(self, sector_impact: Dict[str, float]) -> np.ndarray:
        """
        编码领域影响特征向量

        Args:
            sector_impact: 领域影响字典

        Returns:
            领域特征向量 (1, sector_feature_dim)
        """
        features = np.zeros(self.sector_feature_dim)
        
        # 按预定义领域顺序编码
        for i, sector in enumerate(self.sectors[:self.sector_feature_dim]):
            features[i] = sector_impact.get(sector, 0)
        
        return features.reshape(1, -1)

    def encode_technical_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        编码技术指标特征

        Args:
            df: K线数据

        Returns:
            技术指标特征向量 (1, tech_feature_dim)
        """
        features = np.zeros(self.tech_feature_dim)
        
        if df.empty or len(df) < 20:
            return features.reshape(1, -1)

        try:
            close = df['close']
            
            # 收益率
            features[0] = (close.iloc[-1] / close.iloc[-5] - 1) * 100  # 5日收益
            features[1] = (close.iloc[-1] / close.iloc[-20] - 1) * 100  # 20日收益
            
            # 波动率
            returns = close.pct_change()
            features[2] = returns.tail(5).std() * 100   # 5日波动率
            features[3] = returns.tail(20).std() * 100  # 20日波动率
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features[4] = rsi.iloc[-1] / 100  # 归一化RSI
            
            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            features[5] = macd.iloc[-1] / close.iloc[-1] * 100  # 归一化MACD
            
            # 布林带位置
            ma20 = close.rolling(20).mean()
            std20 = close.rolling(20).std()
            upper = ma20 + 2 * std20
            lower = ma20 - 2 * std20
            bb_position = (close.iloc[-1] - lower.iloc[-1]) / (upper.iloc[-1] - lower.iloc[-1] + 1e-10)
            features[6] = bb_position
            
            # 成交量比率
            volume = df['volume']
            volume_ratio = volume.iloc[-1] / volume.tail(20).mean()
            features[7] = min(volume_ratio / 5, 1)  # 归一化
            
            # 价格位置（60日高低点）
            rolling_high = close.rolling(60).max()
            rolling_low = close.rolling(60).min()
            price_position = (close.iloc[-1] - rolling_low.iloc[-1]) / (rolling_high.iloc[-1] - rolling_low.iloc[-1] + 1e-10)
            features[8] = price_position
            
            # 均线趋势
            ma5 = close.rolling(5).mean()
            ma10 = close.rolling(10).mean()
            ma20 = close.rolling(20).mean()
            features[9] = 1 if ma5.iloc[-1] > ma10.iloc[-1] > ma20.iloc[-1] else -1
            features[10] = (ma5.iloc[-1] - ma20.iloc[-1]) / ma20.iloc[-1] * 100
            
            # 动量
            momentum = close.pct_change(10).iloc[-1] * 100
            features[11] = momentum
            
        except Exception as e:
            logger.warning(f"计算技术指标失败: {e}")

        return features.reshape(1, -1)

    def encode_relevance_features(self, relevance_matrix: np.ndarray,
                                  stock_idx: int) -> np.ndarray:
        """
        编码相关性特征

        Args:
            relevance_matrix: 相关性矩阵
            stock_idx: 目标股票索引

        Returns:
            相关性特征向量 (1, relevance_dim)
        """
        features = np.zeros(self.relevance_dim)
        
        if relevance_matrix is None or relevance_matrix.size == 0:
            return features.reshape(1, -1)

        try:
            # 获取该股票与其他股票的相关性
            correlations = relevance_matrix[stock_idx]
            
            # 统计特征
            features[0] = np.mean(correlations)  # 平均相关性
            features[1] = np.max(correlations)   # 最大相关性
            features[2] = np.std(correlations)   # 相关性波动
            
            # 高相关股票数量
            features[3] = np.sum(correlations > 0.5) / len(correlations)
            features[4] = np.sum(correlations > 0.7) / len(correlations)
            
            # 相关性分布
            sorted_corr = np.sort(correlations)[::-1]
            for i in range(min(10, len(sorted_corr))):
                features[5 + i] = sorted_corr[i]
                
        except Exception as e:
            logger.warning(f"编码相关性特征失败: {e}")

        return features.reshape(1, -1)

    def predict(self, news_features: np.ndarray,
                sector_features: np.ndarray,
                tech_features: np.ndarray,
                relevance_features: np.ndarray) -> Dict:
        """
        多模态预测

        Args:
            news_features: 新闻特征
            sector_features: 领域特征
            tech_features: 技术指标特征
            relevance_features: 相关性特征

        Returns:
            预测结果
        """
        if TENSORFLOW_AVAILABLE and self.model is not None:
            # 使用神经网络模型预测
            prob = self.model.predict([
                news_features,
                sector_features,
                tech_features,
                relevance_features
            ], verbose=0)[0][0]
        else:
            # 使用简化版预测（加权融合）
            prob = self._simple_predict(
                news_features,
                sector_features,
                tech_features,
                relevance_features
            )

        prediction = 1 if prob > 0.5 else 0
        confidence = prob if prediction == 1 else 1 - prob
        
        return {
            'success': True,
            'prediction': int(prediction),
            'prediction_text': '上涨' if prediction == 1 else '下跌',
            'probability': float(prob),
            'confidence': float(confidence),
            'model_type': 'multimodal',
            'features_used': {
                'news': bool(np.any(news_features)),
                'sector': bool(np.any(sector_features)),
                'tech': bool(np.any(tech_features)),
                'relevance': bool(np.any(relevance_features))
            }
        }

    def _simple_predict(self, news_features: np.ndarray,
                        sector_features: np.ndarray,
                        tech_features: np.ndarray,
                        relevance_features: np.ndarray) -> float:
        """
        简化版预测（加权融合）

        Args:
            news_features: 新闻特征
            sector_features: 领域特征
            tech_features: 技术指标特征
            relevance_features: 相关性特征

        Returns:
            预测概率
        """
        # 新闻情感权重
        news_score = news_features[0, 0] * 0.5 + news_features[0, 6] * 0.5
        
        # 领域影响权重
        sector_score = np.mean(sector_features)
        
        # 技术指标权重
        tech_score = 0.5  # 基础分
        if tech_features[0, 9] > 0:  # 均线多头排列
            tech_score += 0.2
        if 0.3 < tech_features[0, 4] < 0.7:  # RSI中性
            tech_score += 0.1
        if tech_features[0, 6] < 0.8:  # 未超买
            tech_score += 0.1
        if tech_features[0, 0] > 0:  # 近期上涨
            tech_score += 0.1
        
        # 相关性权重
        relevance_score = 0.5
        if relevance_features[0, 0] > 0.5:  # 高相关性
            relevance_score += 0.2
        
        # 加权融合
        # 权重：新闻30%，领域20%，技术指标40%，相关性10%
        combined = (
            news_score * 0.3 +
            sector_score * 0.2 +
            tech_score * 0.4 +
            relevance_score * 0.1
        )
        
        # 归一化到0-1
        prob = 1 / (1 + np.exp(-combined * 5))  # sigmoid
        
        return float(prob)

    def full_prediction(self, stock_code: str,
                        news_list: List[Dict],
                        kline_df: pd.DataFrame,
                        sector_impact: Dict[str, float],
                        relevance_matrix: Optional[np.ndarray] = None,
                        stock_idx: int = 0) -> Dict:
        """
        完整预测流程

        Args:
            stock_code: 股票代码
            news_list: 新闻列表
            kline_df: K线数据
            sector_impact: 领域影响
            relevance_matrix: 相关性矩阵
            stock_idx: 股票在矩阵中的索引

        Returns:
            完整预测结果
        """
        # 编码各模态特征
        news_features = self.encode_news_features(news_list)
        sector_features = self.encode_sector_features(sector_impact)
        tech_features = self.encode_technical_features(kline_df)
        
        if relevance_matrix is not None:
            relevance_features = self.encode_relevance_features(relevance_matrix, stock_idx)
        else:
            relevance_features = np.zeros((1, self.relevance_dim))

        # 执行预测
        prediction = self.predict(
            news_features,
            sector_features,
            tech_features,
            relevance_features
        )

        # 添加详细信息
        prediction.update({
            'stock_code': stock_code,
            'latest_price': float(kline_df['close'].iloc[-1]) if not kline_df.empty else 0,
            'news_count': len(news_list),
            'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'feature_summary': {
                'news_sentiment': float(news_features[0, 0]),
                'news_weighted_sentiment': float(news_features[0, 6]),
                'sector_impact': float(np.mean(list(sector_impact.values())) if sector_impact else 0),
                'tech_trend': 'up' if tech_features[0, 9] > 0 else 'down',
                'rsi': float(tech_features[0, 4])
            }
        })

        # 计算预期收益
        if prediction['prediction'] == 1:
            expected_return = prediction['confidence'] * 3  # 最高3%收益
            predicted_price = prediction['latest_price'] * (1 + expected_return / 100)
        else:
            expected_return = -prediction['confidence'] * 2  # 最高-2%亏损
            predicted_price = prediction['latest_price'] * (1 + expected_return / 100)

        prediction['expected_return'] = round(expected_return, 2)
        prediction['predicted_price'] = round(predicted_price, 2)

        return prediction


# 全局实例
multimodal_predictor = MultiModalPredictor()


def get_multimodal_predictor() -> MultiModalPredictor:
    """获取多模态预测器实例"""
    return multimodal_predictor
