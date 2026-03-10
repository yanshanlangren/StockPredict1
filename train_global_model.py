#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
全局股票预测模型训练脚本

功能：
1. 收集所有股票的历史数据
2. 特征工程（使用相对指标，支持多股票）
3. 训练LSTM全局模型
4. 保存模型和标准化参数

使用方法：
    python train_global_model.py [--stocks 100] [--days 300] [--epochs 100]

参数：
    --stocks: 训练使用的股票数量（默认100，最多5000+）
    --days: 每只股票获取的历史天数（默认300）
    --epochs: 训练轮数（默认100）
"""

import os
import sys
import json
import pickle
import argparse
import warnings
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# 尝试导入TensorFlow
try:
    import tensorflow as tf
    # 设置GPU内存增长，避免占用全部显存
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
    from tensorflow.keras.models import Sequential, save_model, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    
    TENSORFLOW_AVAILABLE = True
    print("✓ TensorFlow已加载")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("✗ TensorFlow未安装，请先安装: pip install tensorflow")

# 导入项目模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)
sys.path.insert(0, os.path.join(current_dir, 'src'))

from src.data_source_manager import DataSourceManager, DataSource

# ==================== 配置参数 ====================

MODEL_DIR = "models"
MODEL_NAME = "global_stock_model"
SEQUENCE_LENGTH = 20  # 使用20天序列预测
PREDICT_DAYS = 5      # 预测未来5天
MIN_SAMPLES_PER_STOCK = 50  # 每只股票最少样本数

# 特征列表（相对指标，适合多股票）
FEATURES = [
    'return_1d',       # 1日收益率
    'return_5d',       # 5日收益率
    'return_10d',      # 10日收益率
    'volatility_5d',   # 5日波动率
    'volatility_10d',  # 10日波动率
    'volatility_20d',  # 20日波动率
    'rsi',             # RSI指标
    'macd_norm',       # MACD标准化
    'macd_signal_norm',# MACD信号线标准化
    'bb_position',     # 布林带位置（0-1）
    'volume_ratio',    # 成交量比率
    'price_position',  # 价格在N天内的位置（0-1）
]

# ==================== 特征工程 ====================

class FeatureEngineer:
    """特征工程类"""
    
    @staticmethod
    def calculate_features(df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有特征（相对指标）
        
        Args:
            df: 股票K线数据（包含 open, high, low, close, volume）
            
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
        
        # MACD标准化（相对于价格）
        df['macd_norm'] = macd / (df['close'] + 1e-10)
        df['macd_signal_norm'] = macd_signal / (df['close'] + 1e-10)
        
        # 5. 布林带位置
        bb_middle = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_upper = bb_middle + (bb_std * 2)
        bb_lower = bb_middle - (bb_std * 2)
        df['bb_position'] = (df['close'] - bb_lower) / (bb_upper - bb_lower + 1e-10)
        df['bb_position'] = df['bb_position'].clip(0, 1)  # 限制在0-1
        
        # 6. 成交量比率
        df['volume_ma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_ma'] + 1e-10)
        df['volume_ratio'] = df['volume_ratio'].clip(0, 10)  # 限制异常值
        
        # 7. 价格位置（在60天内的相对位置）
        rolling_min = df['close'].rolling(60).min()
        rolling_max = df['close'].rolling(60).max()
        df['price_position'] = (df['close'] - rolling_min) / (rolling_max - rolling_min + 1e-10)
        df['price_position'] = df['price_position'].fillna(0.5)
        
        return df
    
    @staticmethod
    def prepare_samples(df: pd.DataFrame, sequence_length: int = 20, predict_days: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """
        准备训练样本
        
        Args:
            df: 包含特征的DataFrame
            sequence_length: 输入序列长度
            predict_days: 预测天数
            
        Returns:
            X: 特征序列 (samples, sequence_length, features)
            y: 标签 (samples,) - 1表示上涨，0表示下跌
        """
        # 获取特征列
        feature_cols = [col for col in FEATURES if col in df.columns]
        
        # 确保所有特征都存在
        if len(feature_cols) != len(FEATURES):
            missing = set(FEATURES) - set(feature_cols)
            print(f"警告：缺少特征 {missing}")
            feature_cols = [col for col in FEATURES if col in df.columns]
        
        if len(feature_cols) == 0:
            return np.array([]), np.array([])
        
        # 填充NaN值
        df_features = df[feature_cols].fillna(0)
        
        X, y = [], []
        
        for i in range(len(df_features) - sequence_length - predict_days):
            # 输入序列
            X.append(df_features.iloc[i:i+sequence_length].values)
            
            # 预测目标：未来N天的累计收益率
            current_price = df['close'].iloc[i + sequence_length - 1]
            future_price = df['close'].iloc[i + sequence_length + predict_days - 1]
            future_return = (future_price - current_price) / current_price
            
            # 标签：收益超过1%为1，否则为0
            y.append(1 if future_return > 0.01 else 0)
        
        return np.array(X), np.array(y)


# ==================== 数据收集 ====================

def collect_stock_data(data_manager: DataSourceManager, stock_list: pd.DataFrame, 
                       days: int = 300, max_stocks: int = 100) -> Dict[str, pd.DataFrame]:
    """
    收集多只股票的历史数据
    
    Args:
        data_manager: 数据源管理器
        stock_list: 股票列表
        days: 获取天数
        max_stocks: 最大股票数量
        
    Returns:
        字典：{股票代码: DataFrame}
    """
    stock_data = {}
    failed_count = 0
    
    print(f"\n开始收集股票数据（目标：{min(len(stock_list), max_stocks)} 只）...")
    
    for idx, row in stock_list.head(max_stocks).iterrows():
        stock_code = str(row['code'])
        stock_name = row.get('name', stock_code)
        
        try:
            # 获取股票数据
            df = data_manager.get_stock_kline(stock_code, days=days)
            
            if df.empty or len(df) < MIN_SAMPLES_PER_STOCK:
                print(f"  [{idx+1}/{min(len(stock_list), max_stocks)}] {stock_code} {stock_name}: 数据不足，跳过")
                failed_count += 1
                continue
            
            # 计算特征
            df = FeatureEngineer.calculate_features(df)
            
            # 去除NaN
            df = df.dropna()
            
            if len(df) < MIN_SAMPLES_PER_STOCK:
                print(f"  [{idx+1}/{min(len(stock_list), max_stocks)}] {stock_code} {stock_name}: 特征计算后数据不足，跳过")
                failed_count += 1
                continue
            
            stock_data[stock_code] = df
            print(f"  [{idx+1}/{min(len(stock_list), max_stocks)}] ✓ {stock_code} {stock_name}: {len(df)} 天数据")
            
        except Exception as e:
            print(f"  [{idx+1}/{min(len(stock_list), max_stocks)}] ✗ {stock_code} {stock_name}: {str(e)}")
            failed_count += 1
            continue
    
    print(f"\n数据收集完成：成功 {len(stock_data)} 只，失败 {failed_count} 只")
    return stock_data


def prepare_training_data(stock_data: Dict[str, pd.DataFrame], 
                          sequence_length: int = 20,
                          predict_days: int = 5) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    准备训练数据（合并所有股票）
    
    Args:
        stock_data: 股票数据字典
        sequence_length: 序列长度
        predict_days: 预测天数
        
    Returns:
        X: 特征数据
        y: 标签数据
        stats: 统计信息
    """
    print(f"\n准备训练数据...")
    
    all_X, all_y = [], []
    stats = {
        'total_stocks': len(stock_data),
        'total_samples': 0,
        'positive_samples': 0,
        'negative_samples': 0
    }
    
    for stock_code, df in stock_data.items():
        X, y = FeatureEngineer.prepare_samples(df, sequence_length, predict_days)
        
        if len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            stats['total_samples'] += len(y)
            stats['positive_samples'] += np.sum(y)
            stats['negative_samples'] += (len(y) - np.sum(y))
    
    # 合并所有数据
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    print(f"训练数据准备完成：")
    print(f"  总样本数：{stats['total_samples']}")
    print(f"  正样本：{stats['positive_samples']} ({stats['positive_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  负样本：{stats['negative_samples']} ({stats['negative_samples']/stats['total_samples']*100:.1f}%)")
    print(f"  特征维度：{X.shape[2]}")
    
    return X, y, stats


# ==================== 模型构建 ====================

def build_model(sequence_length: int, n_features: int) -> Sequential:
    """
    构建LSTM模型
    
    Args:
        sequence_length: 序列长度
        n_features: 特征数量
        
    Returns:
        Keras模型
    """
    model = Sequential([
        Input(shape=(sequence_length, n_features)),
        
        LSTM(128, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(64, return_sequences=True),
        BatchNormalization(),
        Dropout(0.3),
        
        LSTM(32, return_sequences=False),
        BatchNormalization(),
        Dropout(0.3),
        
        Dense(16, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        
        Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
    )
    
    return model


def train_model(X: np.ndarray, y: np.ndarray, epochs: int = 100, 
                model_dir: str = MODEL_DIR) -> Tuple[Sequential, Dict]:
    """
    训练模型
    
    Args:
        X: 特征数据
        y: 标签数据
        epochs: 训练轮数
        model_dir: 模型保存目录
        
    Returns:
        训练好的模型和训练历史
    """
    print(f"\n开始训练模型...")
    
    # 创建模型目录
    os.makedirs(model_dir, exist_ok=True)
    
    # 划分训练集和验证集
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    print(f"训练集：{len(X_train)} 样本")
    print(f"验证集：{len(X_val)} 样本")
    
    # 构建模型
    model = build_model(SEQUENCE_LENGTH, X.shape[2])
    model.summary()
    
    # 回调函数
    callbacks = [
        EarlyStopping(monitor='val_auc', patience=15, mode='max', restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
        ModelCheckpoint(
            filepath=os.path.join(model_dir, f'{MODEL_NAME}_best.keras'),
            monitor='val_auc',
            mode='max',
            save_best_only=True
        )
    ]
    
    # 训练
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=256,
        callbacks=callbacks,
        verbose=1
    )
    
    # 评估最终模型
    print("\n最终模型评估：")
    loss, accuracy, auc = model.evaluate(X_val, y_val, verbose=0)
    print(f"  验证集损失：{loss:.4f}")
    print(f"  验证集准确率：{accuracy:.4f}")
    print(f"  验证集AUC：{auc:.4f}")
    
    return model, history.history


def save_model_and_metadata(model: Sequential, stats: Dict, history: Dict, 
                            model_dir: str = MODEL_DIR):
    """
    保存模型和元数据
    
    Args:
        model: 训练好的模型
        stats: 统计信息
        history: 训练历史
        model_dir: 模型保存目录
    """
    print(f"\n保存模型和元数据...")
    
    # 保存Keras模型
    model_path = os.path.join(model_dir, f'{MODEL_NAME}.keras')
    model.save(model_path)
    print(f"  模型已保存：{model_path}")
    
    # 转换numpy类型为Python原生类型
    def convert_to_native(obj):
        """递归转换numpy类型为Python原生类型"""
        if isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(v) for v in obj]
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    # 保存元数据
    metadata = {
        'model_name': MODEL_NAME,
        'sequence_length': SEQUENCE_LENGTH,
        'predict_days': PREDICT_DAYS,
        'features': FEATURES,
        'n_features': len(FEATURES),
        'training_stats': convert_to_native(stats),
        'final_metrics': {
            'loss': float(history['loss'][-1]),
            'accuracy': float(history['accuracy'][-1]),
            'val_loss': float(history['val_loss'][-1]),
            'val_accuracy': float(history['val_accuracy'][-1])
        },
        'created_at': datetime.now().isoformat(),
        'tensorflow_version': tf.__version__
    }
    
    metadata_path = os.path.join(model_dir, f'{MODEL_NAME}_metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    print(f"  元数据已保存：{metadata_path}")
    
    # 保存训练历史
    history_path = os.path.join(model_dir, f'{MODEL_NAME}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    print(f"  训练历史已保存：{history_path}")


# ==================== 主函数 ====================

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='训练全局股票预测模型')
    parser.add_argument('--stocks', type=int, default=100, help='训练使用的股票数量')
    parser.add_argument('--days', type=int, default=300, help='每只股票获取的历史天数')
    parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("全局股票预测模型训练")
    print("=" * 60)
    print(f"参数配置：")
    print(f"  股票数量：{args.stocks}")
    print(f"  历史天数：{args.days}")
    print(f"  训练轮数：{args.epochs}")
    print(f"  序列长度：{SEQUENCE_LENGTH}")
    print(f"  预测天数：{PREDICT_DAYS}")
    print(f"  特征数量：{len(FEATURES)}")
    
    if not TENSORFLOW_AVAILABLE:
        print("\n错误：TensorFlow未安装，无法训练模型")
        print("请运行: pip install tensorflow")
        return
    
    # 初始化数据源管理器
    print("\n初始化数据源...")
    data_manager = DataSourceManager(preferred_source=DataSource.TENCENT)
    
    # 获取股票列表
    print("\n获取股票列表...")
    stock_list = data_manager.get_stock_list(limit=args.stocks)
    
    if stock_list.empty:
        print("错误：无法获取股票列表")
        return
    
    print(f"获取到 {len(stock_list)} 只股票")
    
    # 收集股票数据
    stock_data = collect_stock_data(data_manager, stock_list, days=args.days, max_stocks=args.stocks)
    
    if len(stock_data) == 0:
        print("错误：没有收集到有效的股票数据")
        return
    
    # 准备训练数据
    X, y, stats = prepare_training_data(stock_data, SEQUENCE_LENGTH, PREDICT_DAYS)
    
    # 训练模型
    model, history = train_model(X, y, epochs=args.epochs)
    
    # 保存模型和元数据
    save_model_and_metadata(model, stats, history)
    
    print("\n" + "=" * 60)
    print("训练完成！")
    print("=" * 60)
    print(f"模型文件保存在：{MODEL_DIR}/")
    print(f"  - {MODEL_NAME}.keras")
    print(f"  - {MODEL_NAME}_metadata.json")
    print(f"  - {MODEL_NAME}_history.pkl")


if __name__ == '__main__':
    main()
