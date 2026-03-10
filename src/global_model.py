#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全局股票预测模型管理模块

功能：
1. 加载预训练的全局模型
2. 提供预测接口
3. 模型缓存管理
"""

import os
import json
import logging
from typing import Optional, Dict, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# 尝试导入TensorFlow
try:
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("TensorFlow未安装，全局模型不可用")

# ==================== 配置参数 ====================

# 获取项目根目录（优先使用环境变量，否则使用当前文件位置推断）
def _get_project_root():
    """获取项目根目录"""
    # 方法1：使用环境变量
    if os.environ.get('COZE_WORKSPACE_PATH'):
        return os.environ.get('COZE_WORKSPACE_PATH')
    
    # 方法2：使用当前文件位置推断
    try:
        current_file = os.path.abspath(__file__)
        src_dir = os.path.dirname(current_file)
        project_root = os.path.dirname(src_dir)
        return project_root
    except:
        # 方法3：使用当前工作目录
        return os.getcwd()

PROJECT_ROOT = _get_project_root()
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
MODEL_NAME = "global_stock_model"
SEQUENCE_LENGTH = 20
PREDICT_DAYS = 5
FEATURES = [
    'return_1d', 'return_5d', 'return_10d',
    'volatility_5d', 'volatility_10d', 'volatility_20d',
    'rsi', 'macd_norm', 'macd_signal_norm',
    'bb_position', 'volume_ratio', 'price_position'
]

# 调试输出
logger.info(f"全局模型配置 - 项目根目录: {PROJECT_ROOT}")
logger.info(f"全局模型配置 - 模型目录: {MODEL_DIR}")


class GlobalStockModel:
    """全局股票预测模型管理类"""
    
    _instance = None  # 单例实例
    _model = None     # 模型实例
    _metadata = None  # 模型元数据
    
    def __new__(cls):
        """单例模式"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """初始化"""
        self.model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}.keras")
        self.metadata_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_metadata.json")
    
    def is_available(self) -> bool:
        """检查模型是否可用"""
        if not TENSORFLOW_AVAILABLE:
            return False
        
        return os.path.exists(self.model_path)
    
    def load_model(self) -> bool:
        """
        加载预训练模型
        
        Returns:
            是否加载成功
        """
        if not TENSORFLOW_AVAILABLE:
            logger.warning("TensorFlow不可用，无法加载模型")
            return False
        
        if not os.path.exists(self.model_path):
            logger.warning(f"模型文件不存在: {self.model_path}")
            return False
        
        try:
            # 加载模型
            logger.info(f"加载全局模型: {self.model_path}")
            self._model = load_model(self.model_path)
            
            # 加载元数据
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    self._metadata = json.load(f)
                logger.info(f"模型元数据: {self._metadata.get('created_at', 'unknown')}")
            
            logger.info("✓ 全局模型加载成功")
            return True
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            return False
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        # 如果模型文件存在但未加载，先尝试加载
        if self._model is None and self.is_available():
            self.load_model()
        
        if self._metadata:
            return {
                'available': True,
                'model_name': self._metadata.get('model_name', MODEL_NAME),
                'sequence_length': self._metadata.get('sequence_length', SEQUENCE_LENGTH),
                'predict_days': self._metadata.get('predict_days', PREDICT_DAYS),
                'n_features': self._metadata.get('n_features', len(FEATURES)),
                'created_at': self._metadata.get('created_at', 'unknown'),
                'training_stats': self._metadata.get('training_stats', {}),
                'final_metrics': self._metadata.get('final_metrics', {})
            }
        
        # 如果模型文件存在但元数据不存在，返回基本信息
        if self.is_available():
            return {
                'available': True,
                'model_name': MODEL_NAME,
                'sequence_length': SEQUENCE_LENGTH,
                'predict_days': PREDICT_DAYS,
                'n_features': len(FEATURES),
                'created_at': 'unknown',
                'note': 'Model file exists but metadata not loaded'
            }
        
        return {'available': False}
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征（与训练时一致）
        
        Args:
            df: 股票K线数据
            
        Returns:
            添加特征后的DataFrame
        """
        df = df.copy()
        
        # 1. 收益率特征
        df['return_1d'] = df['close'].pct_change(1)
        df['return_5d'] = df['close'].pct_change(5)
        df['return_10d'] = df['close'].pct_change(10)
        
        # 2. 波动率特征
        df['volatility_5d'] = df['return_1d'].rolling(5).std()
        df['volatility_10d'] = df['return_1d'].rolling(10).std()
        df['volatility_20d'] = df['return_1d'].rolling(20).std()
        
        # 3. RSI指标
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        df['rsi'] = df['rsi'] / 100  # 标准化到0-1
        
        # 4. MACD指标（标准化）
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        
        df['macd_norm'] = macd / (df['close'] + 1e-10)
        df['macd_signal_norm'] = macd_signal / (df['close'] + 1e-10)
        
        # 5. 布林带位置
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        df['bb_position'] = df['bb_position'].clip(0, 1)
        
        # 6. 成交量比率
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        df['volume_ratio'] = df['volume_ratio'].clip(0, 10)
        
        # 7. 价格位置
        rolling_min = df['close'].rolling(60).min()
        rolling_max = df['close'].rolling(60).max()
        df['price_position'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
        df['price_position'] = df['price_position'].fillna(0.5)
        
        return df
    
    def prepare_input(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        准备模型输入数据
        
        Args:
            df: 股票K线数据（至少包含 sequence_length + 60 天）
            
        Returns:
            模型输入数据 (1, sequence_length, n_features) 或 None
        """
        if df.empty or len(df) < SEQUENCE_LENGTH + 60:
            return None
        
        try:
            # 计算特征
            df_features = self.calculate_features(df)
            
            # 填充NaN
            df_features = df_features.fillna(0)
            
            # 获取最后 sequence_length 天的特征
            feature_cols = [col for col in FEATURES if col in df_features.columns]
            
            if len(feature_cols) < len(FEATURES):
                missing = set(FEATURES) - set(feature_cols)
                logger.warning(f"缺少特征: {missing}")
                feature_cols = [col for col in FEATURES if col in df_features.columns]
            
            if len(feature_cols) == 0:
                return None
            
            # 提取特征序列
            X = df_features[feature_cols].tail(SEQUENCE_LENGTH).values
            
            # 调整形状为 (1, sequence_length, n_features)
            X = X.reshape(1, SEQUENCE_LENGTH, len(feature_cols))
            
            return X
            
        except Exception as e:
            logger.error(f"准备输入数据失败: {e}")
            return None
    
    def predict(self, df: pd.DataFrame) -> Dict:
        """
        使用全局模型预测股票走势
        
        Args:
            df: 股票K线数据
            
        Returns:
            预测结果字典
        """
        # 检查模型是否加载
        if self._model is None:
            if not self.load_model():
                return {
                    'success': False,
                    'message': '全局模型未加载'
                }
        
        # 准备输入数据
        X = self.prepare_input(df)
        
        if X is None:
            return {
                'success': False,
                'message': f'数据不足，至少需要 {SEQUENCE_LENGTH + 60} 天数据'
            }
        
        try:
            # 预测
            prob_array = self._model.predict(X, verbose=0)
            prob = float(prob_array[0][0])  # 确保转换为 Python float
            
            # 获取最新价格
            latest_price = float(df['close'].iloc[-1])
            
            # 预测结果
            prediction = 1 if prob > 0.5 else 0
            confidence = float(prob if prediction == 1 else 1 - prob)
            
            # 预测价格变化
            # 假设上涨概率对应不同的预期收益
            if prediction == 1:
                expected_return = 0.01 + (prob - 0.5) * 0.05  # 1% - 3.5%
                predicted_price = latest_price * (1 + expected_return)
            else:
                expected_return = -0.01 - (0.5 - prob) * 0.03  # -1% 到 -2.5%
                predicted_price = latest_price * (1 + expected_return)
            
            return {
                'success': True,
                'prediction': int(prediction),
                'prediction_text': '上涨' if prediction == 1 else '下跌',
                'probability': float(prob),
                'confidence': float(confidence),
                'latest_price': float(latest_price),
                'predicted_price': float(round(predicted_price, 2)),
                'expected_return': float(round(expected_return * 100, 2)),  # 百分比
                'predict_days': int(PREDICT_DAYS),
                'model_type': 'global',
                'model_info': self.get_model_info()
            }
            
        except Exception as e:
            logger.error(f"预测失败: {e}")
            return {
                'success': False,
                'message': f'预测失败: {str(e)}'
            }


# 全局实例
global_model = GlobalStockModel()


def get_global_model() -> GlobalStockModel:
    """获取全局模型实例"""
    return global_model
