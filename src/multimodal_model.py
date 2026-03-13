#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
多模态预测模型 - 融合新闻、技术指标、相关性矩阵的全局预测模型

功能：
1. 模型训练 - 使用所有股票数据训练全局多模态模型
2. 股票预测 - 基于训练好的模型进行单股预测
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
    from tensorflow.keras import layers, models, callbacks
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
    logger.info("✓ TensorFlow已加载，多模态模型功能可用")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow未安装，使用简化版预测模型")

# 获取项目根目录
def _get_project_root():
    """获取项目根目录"""
    if os.environ.get('COZE_WORKSPACE_PATH'):
        return os.environ.get('COZE_WORKSPACE_PATH')
    try:
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file)
        return os.path.dirname(src_dir)
    except:
        return os.getcwd()

PROJECT_ROOT = _get_project_root()
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_NAME = "multimodal_stock_predictor"


class MultiModalPredictor:
    """多模态股票预测模型"""

    _instance = None

    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, model_dir: str = None):
        """初始化多模态预测器"""
        if model_dir is None:
            model_dir = MODEL_DIR
        
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        self.model = None
        self.model_name = MODEL_NAME
        
        # 特征维度
        self.news_feature_dim = 64
        self.sector_feature_dim = 32
        self.tech_feature_dim = 12
        self.relevance_dim = 16
        
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
        return True  # 简化版始终可用

    def _load_model(self):
        """加载已训练的模型"""
        if not TENSORFLOW_AVAILABLE:
            logger.info("使用简化版预测模型（无需TensorFlow）")
            return

        model_path = os.path.join(self.model_dir, f"{self.model_name}.keras")
        metadata_path = os.path.join(self.model_dir, f"{self.model_name}_metadata.json")
        
        if os.path.exists(model_path):
            try:
                self.model = keras.models.load_model(model_path)
                logger.info(f"✓ 多模态模型加载成功: {model_path}")
                
                # 加载元数据
                if os.path.exists(metadata_path):
                    with open(metadata_path, 'r', encoding='utf-8') as f:
                        self.metadata = json.load(f)
            except Exception as e:
                logger.warning(f"加载模型失败: {e}")
                self.model = None
        else:
            logger.info(f"多模态模型文件不存在: {model_path}")

    def _build_model(self) -> keras.Model:
        """构建多模态神经网络模型"""
        if not TENSORFLOW_AVAILABLE:
            return None
        
        # 新闻特征输入
        news_input = layers.Input(shape=(self.news_feature_dim,), name='news_features')
        news_dense = layers.Dense(32, activation='relu')(news_input)
        news_dropout = layers.Dropout(0.2)(news_dense)
        
        # 领域影响特征输入
        sector_input = layers.Input(shape=(self.sector_feature_dim,), name='sector_features')
        sector_dense = layers.Dense(16, activation='relu')(sector_input)
        
        # 技术指标输入
        tech_input = layers.Input(shape=(self.tech_feature_dim,), name='tech_features')
        tech_dense = layers.Dense(32, activation='relu')(tech_input)
        tech_dropout = layers.Dropout(0.2)(tech_dense)
        
        # 相关性特征输入
        relevance_input = layers.Input(shape=(self.relevance_dim,), name='relevance_features')
        relevance_dense = layers.Dense(8, activation='relu')(relevance_input)
        
        # 多模态融合
        concat = layers.Concatenate()([news_dropout, sector_dense, tech_dropout, relevance_dense])
        
        # 全连接层
        x = layers.Dense(128, activation='relu')(concat)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.3)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # 输出层
        output = layers.Dense(1, activation='sigmoid', name='prediction')(x)
        
        # 构建模型
        model = models.Model(
            inputs=[news_input, sector_input, tech_input, relevance_input],
            outputs=output
        )
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        metadata_path = os.path.join(self.model_dir, f"{self.model_name}_metadata.json")
        
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                pass
        
        # 检查模型文件是否存在
        model_path = os.path.join(self.model_dir, f"{self.model_name}.keras")
        if os.path.exists(model_path):
            return {
                'available': True,
                'model_name': self.model_name,
                'tensorflow_available': TENSORFLOW_AVAILABLE
            }
        
        return {'available': False}

    def train(self, training_data: Dict, epochs: int = 50, batch_size: int = 32) -> Dict:
        """
        训练多模态模型
        
        Args:
            training_data: 训练数据字典，包含：
                - news_features: 新闻特征数组 (n_samples, news_feature_dim)
                - sector_features: 领域特征数组 (n_samples, sector_feature_dim)
                - tech_features: 技术指标特征数组 (n_samples, tech_feature_dim)
                - relevance_features: 相关性特征数组 (n_samples, relevance_dim)
                - labels: 标签数组 (n_samples,) 1=上涨, 0=下跌
            epochs: 训练轮数
            batch_size: 批量大小
            
        Returns:
            训练结果
        """
        if not TENSORFLOW_AVAILABLE:
            return {
                'success': False,
                'message': 'TensorFlow不可用，无法训练神经网络模型'
            }
        
        try:
            # 构建模型
            self.model = self._build_model()
            
            # 准备训练数据
            X_train = [
                np.array(training_data['news_features']),
                np.array(training_data['sector_features']),
                np.array(training_data['tech_features']),
                np.array(training_data['relevance_features'])
            ]
            y_train = np.array(training_data['labels'])
            
            # 划分验证集
            val_split = 0.2
            
            # 回调函数
            callback_list = [
                callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=10,
                    restore_best_weights=True
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-6
                )
            ]
            
            # 训练模型
            history = self.model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_split=val_split,
                callbacks=callback_list,
                verbose=1
            )
            
            # 保存模型
            model_path = os.path.join(self.model_dir, f"{self.model_name}.keras")
            self.model.save(model_path)
            
            # 保存元数据
            metadata = {
                'model_name': self.model_name,
                'available': True,
                'created_at': datetime.now().isoformat(),
                'training_samples': len(y_train),
                'epochs_trained': len(history.history['loss']),
                'final_metrics': {
                    'loss': float(history.history['loss'][-1]),
                    'accuracy': float(history.history['accuracy'][-1]),
                    'val_loss': float(history.history['val_loss'][-1]),
                    'val_accuracy': float(history.history['val_accuracy'][-1])
                },
                'feature_dims': {
                    'news': self.news_feature_dim,
                    'sector': self.sector_feature_dim,
                    'tech': self.tech_feature_dim,
                    'relevance': self.relevance_dim
                }
            }
            
            metadata_path = os.path.join(self.model_dir, f"{self.model_name}_metadata.json")
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✓ 多模态模型训练完成，保存至: {model_path}")
            
            return {
                'success': True,
                'model_path': model_path,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"训练模型失败: {e}")
            import traceback
            traceback.print_exc()
            return {
                'success': False,
                'message': str(e)
            }

    def encode_news_features(self, news_list: List[Dict]) -> np.ndarray:
        """编码新闻特征向量"""
        features = np.zeros(self.news_feature_dim)
        
        if not news_list:
            return features.reshape(1, -1)

        sentiments = [n.get('sentiment', 0) for n in news_list]
        importances = [n.get('importance', 0.5) for n in news_list]
        
        # 基础统计
        features[0] = np.mean(sentiments)
        features[1] = np.std(sentiments) if len(sentiments) > 1 else 0
        features[2] = np.mean(importances)
        features[3] = min(len(news_list) / 50, 1)
        
        # 情感分布
        features[4] = sum(1 for s in sentiments if s > 0.2) / max(len(sentiments), 1)
        features[5] = sum(1 for s in sentiments if s < -0.2) / max(len(sentiments), 1)
        
        # 时间加权情感
        current_time = datetime.now()
        time_weights = []
        for news in news_list:
            pub_time = news.get('publish_time', '')
            if pub_time:
                try:
                    dt = datetime.strptime(pub_time, '%Y-%m-%d %H:%M:%S')
                    hours_ago = (current_time - dt).total_seconds() / 3600
                    weight = np.exp(-hours_ago / 24)
                except:
                    weight = 0.5
            else:
                weight = 0.5
            time_weights.append(weight)
        
        weighted_sentiment = np.average(sentiments, weights=time_weights) if time_weights else 0
        features[6] = weighted_sentiment
        
        # 类别分布
        categories = {}
        for news in news_list:
            for cat in news.get('categories', []):
                categories[cat] = categories.get(cat, 0) + 1
        
        for i, cat in enumerate(list(categories.keys())[:20]):
            features[7 + i] = categories[cat] / len(news_list)
        
        return features.reshape(1, -1)

    def encode_sector_features(self, sector_impact: Dict[str, float]) -> np.ndarray:
        """编码领域影响特征向量"""
        features = np.zeros(self.sector_feature_dim)
        
        for i, sector in enumerate(self.sectors[:self.sector_feature_dim]):
            features[i] = sector_impact.get(sector, 0)
        
        return features.reshape(1, -1)

    def encode_technical_features(self, df: pd.DataFrame) -> np.ndarray:
        """编码技术指标特征"""
        features = np.zeros(self.tech_feature_dim)
        
        if df.empty or len(df) < 20:
            return features.reshape(1, -1)

        try:
            close = df['close']
            
            # 收益率
            features[0] = (close.iloc[-1] / close.iloc[-5] - 1) * 100
            features[1] = (close.iloc[-1] / close.iloc[-20] - 1) * 100
            
            # 波动率
            returns = close.pct_change()
            features[2] = returns.tail(5).std() * 100
            features[3] = returns.tail(20).std() * 100
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / (loss + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            features[4] = rsi.iloc[-1] / 100
            
            # MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = ema12 - ema26
            features[5] = macd.iloc[-1] / close.iloc[-1] * 100
            
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
            features[7] = min(volume_ratio / 5, 1)
            
            # 价格位置
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
        """编码相关性特征"""
        features = np.zeros(self.relevance_dim)
        
        if relevance_matrix is None or relevance_matrix.size == 0:
            return features.reshape(1, -1)

        try:
            correlations = relevance_matrix[stock_idx]
            
            features[0] = np.mean(correlations)
            features[1] = np.max(correlations)
            features[2] = np.std(correlations)
            features[3] = np.sum(correlations > 0.5) / len(correlations)
            features[4] = np.sum(correlations > 0.7) / len(correlations)
            
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
        """执行预测"""
        if TENSORFLOW_AVAILABLE and self.model is not None:
            prob = self.model.predict([
                news_features,
                sector_features,
                tech_features,
                relevance_features
            ], verbose=0)[0][0]
        else:
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
        """简化版预测（加权融合）"""
        # 新闻情感权重
        news_score = news_features[0, 0] * 0.5 + news_features[0, 6] * 0.5
        
        # 领域影响权重
        sector_score = np.mean(sector_features)
        
        # 技术指标权重
        tech_score = 0.5
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
        if relevance_features[0, 0] > 0.5:
            relevance_score += 0.2
        
        # 加权融合：新闻30%，领域20%，技术指标40%，相关性10%
        combined = (
            news_score * 0.3 +
            sector_score * 0.2 +
            tech_score * 0.4 +
            relevance_score * 0.1
        )
        
        # 归一化
        prob = 1 / (1 + np.exp(-combined * 5))
        
        return float(prob)

    def predict_stock(self, stock_code: str, 
                      kline_df: pd.DataFrame,
                      news_list: List[Dict] = None,
                      sector_impact: Dict[str, float] = None,
                      relevance_matrix: np.ndarray = None,
                      stock_idx: int = 0) -> Dict:
        """
        单股预测
        
        Args:
            stock_code: 股票代码
            kline_df: K线数据
            news_list: 新闻列表
            sector_impact: 领域影响
            relevance_matrix: 相关性矩阵
            stock_idx: 股票索引
            
        Returns:
            预测结果
        """
        if news_list is None:
            news_list = []
        if sector_impact is None:
            sector_impact = {}
            
        # 编码特征
        news_features = self.encode_news_features(news_list)
        sector_features = self.encode_sector_features(sector_impact)
        tech_features = self.encode_technical_features(kline_df)
        
        if relevance_matrix is not None:
            relevance_features = self.encode_relevance_features(relevance_matrix, stock_idx)
        else:
            relevance_features = np.zeros((1, self.relevance_dim))

        # 预测
        prediction = self.predict(
            news_features,
            sector_features,
            tech_features,
            relevance_features
        )

        # 添加详细信息
        latest_price = float(kline_df['close'].iloc[-1]) if not kline_df.empty else 0
        
        prediction.update({
            'stock_code': stock_code,
            'latest_price': latest_price,
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
            expected_return = prediction['confidence'] * 3
            predicted_price = latest_price * (1 + expected_return / 100)
        else:
            expected_return = -prediction['confidence'] * 2
            predicted_price = latest_price * (1 + expected_return / 100)

        prediction['expected_return'] = round(expected_return, 2)
        prediction['predicted_price'] = round(predicted_price, 2)
        prediction['success'] = True

        return prediction


# 单例获取函数
def get_multimodal_predictor() -> MultiModalPredictor:
    """获取多模态预测器单例"""
    return MultiModalPredictor()
