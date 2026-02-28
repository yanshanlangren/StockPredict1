"""
深度学习模型模块 - LSTM/GRU股价预测模型
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Input, Bidirectional, Attention
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import numpy as np
import os
import sys
import logging
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import MODEL_DIR, LOG_DIR, BATCH_SIZE, EPOCHS, VALIDATION_SPLIT, EARLY_STOPPING_PATIENCE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'model.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StockPredictionModel:
    """股票预测模型基类"""

    def __init__(self, model_name: str = 'stock_model'):
        self.model_name = model_name
        self.model = None
        self.history = None
        self.model_path = os.path.join(MODEL_DIR, f'{model_name}.keras')
        self.config_path = os.path.join(MODEL_DIR, f'{model_name}_config.json')

    def build_lstm_model(self, input_shape: tuple, lstm_units: list = [128, 64, 32], 
                         dropout_rate: float = 0.3, learning_rate: float = 0.001):
        """
        构建LSTM模型
        
        Args:
            input_shape: 输入形状 (sequence_length, n_features)
            lstm_units: LSTM层单元数列表
            dropout_rate: Dropout比率
            learning_rate: 学习率
        """
        model = Sequential([
            LSTM(lstm_units[0], return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            LSTM(lstm_units[1], return_sequences=True),
            Dropout(dropout_rate),
            LSTM(lstm_units[2], return_sequences=False),
            Dropout(dropout_rate),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        
        self.model = model
        logger.info(f"LSTM模型构建完成，参数数量: {model.count_params()}")

    def build_gru_model(self, input_shape: tuple, gru_units: list = [128, 64, 32], 
                        dropout_rate: float = 0.3, learning_rate: float = 0.001):
        """
        构建GRU模型
        
        Args:
            input_shape: 输入形状
            gru_units: GRU层单元数列表
            dropout_rate: Dropout比率
            learning_rate: 学习率
        """
        model = Sequential([
            GRU(gru_units[0], return_sequences=True, input_shape=input_shape),
            Dropout(dropout_rate),
            GRU(gru_units[1], return_sequences=True),
            Dropout(dropout_rate),
            GRU(gru_units[2], return_sequences=False),
            Dropout(dropout_rate),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        
        self.model = model
        logger.info(f"GRU模型构建完成，参数数量: {model.count_params()}")

    def build_bidirectional_lstm_model(self, input_shape: tuple, lstm_units: list = [128, 64],
                                       dropout_rate: float = 0.3, learning_rate: float = 0.001):
        """
        构建双向LSTM模型
        
        Args:
            input_shape: 输入形状
            lstm_units: LSTM层单元数列表
            dropout_rate: Dropout比率
            learning_rate: 学习率
        """
        model = Sequential([
            Bidirectional(LSTM(lstm_units[0], return_sequences=True), input_shape=input_shape),
            Dropout(dropout_rate),
            Bidirectional(LSTM(lstm_units[1], return_sequences=False)),
            Dropout(dropout_rate),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae', 'mape'])
        
        self.model = model
        logger.info(f"双向LSTM模型构建完成，参数数量: {model.count_params()}")

    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: np.ndarray = None, y_val: np.ndarray = None,
              epochs: int = EPOCHS, batch_size: int = BATCH_SIZE):
        """
        训练模型
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_val: 验证特征
            y_val: 验证目标
            epochs: 训练轮数
            batch_size: 批次大小
        """
        if self.model is None:
            raise ValueError("模型未构建，请先调用build_xxx_model方法")
        
        # 回调函数
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-6,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=self.model_path,
                monitor='val_loss',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # 准备验证数据
        validation_data = None
        if X_val is not None and y_val is not None:
            validation_data = (X_val, y_val)
        else:
            validation_split = VALIDATION_SPLIT
        
        logger.info("开始训练模型...")
        logger.info(f"训练数据: X_train={X_train.shape}, y_train={y_train.shape}")
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=validation_data,
            validation_split=validation_split,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        logger.info("模型训练完成")
        
        # 保存训练配置
        self.save_config({
            'input_shape': list(X_train.shape[1:]),
            'epochs_trained': len(self.history.history['loss']),
            'final_train_loss': float(self.history.history['loss'][-1]),
            'final_val_loss': float(self.history.history['val_loss'][-1]) if 'val_loss' in self.history.history else None
        })

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 输入特征
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("模型未加载，请先训练或加载模型")
        
        return self.model.predict(X, verbose=0)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """
        评估模型
        
        Args:
            X_test: 测试特征
            y_test: 测试目标
            
        Returns:
            评估指标字典
        """
        if self.model is None:
            raise ValueError("模型未加载，请先训练或加载模型")
        
        loss, mae, mape = self.model.evaluate(X_test, y_test, verbose=0)
        
        y_pred = self.predict(X_test)
        
        # 计算自定义指标
        mse = np.mean(np.square(y_test - y_pred))
        rmse = np.sqrt(mse)
        
        metrics = {
            'loss': loss,
            'mae': mae,
            'mape': mape,
            'mse': mse,
            'rmse': rmse
        }
        
        logger.info(f"模型评估结果: {metrics}")
        
        return metrics

    def save_model(self, filepath: str = None):
        """
        保存模型
        
        Args:
            filepath: 保存路径
        """
        if self.model is None:
            raise ValueError("没有模型可保存")
        
        if filepath is None:
            filepath = self.model_path
        
        self.model.save(filepath)
        logger.info(f"模型已保存到 {filepath}")

    def load_model(self, filepath: str = None):
        """
        加载模型
        
        Args:
            filepath: 模型路径
        """
        if filepath is None:
            filepath = self.model_path
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"模型文件不存在: {filepath}")
        
        self.model = keras.models.load_model(filepath)
        logger.info(f"模型已从 {filepath} 加载")

    def save_config(self, config: dict):
        """
        保存配置
        
        Args:
            config: 配置字典
        """
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logger.info(f"配置已保存到 {self.config_path}")

    def load_config(self) -> dict:
        """
        加载配置
        
        Returns:
            配置字典
        """
        if not os.path.exists(self.config_path):
            return {}
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        return config


class MultiModelEnsemble:
    """多模型集成"""

    def __init__(self):
        self.models = {}

    def add_model(self, name: str, model: StockPredictionModel):
        """
        添加模型
        
        Args:
            name: 模型名称
            model: 模型实例
        """
        self.models[name] = model
        logger.info(f"已添加模型: {name}")

    def predict_ensemble(self, X: np.ndarray, weights: dict = None) -> np.ndarray:
        """
        集成预测
        
        Args:
            X: 输入特征
            weights: 各模型权重
            
        Returns:
            加权平均预测结果
        """
        predictions = []
        
        for name, model in self.models.items():
            pred = model.predict(X)
            predictions.append(pred)
        
        predictions = np.array(predictions)
        
        if weights:
            # 加权平均
            weighted_preds = []
            for i, (name, weight) in enumerate(weights.items()):
                weighted_preds.append(predictions[i] * weight)
            ensemble_pred = np.sum(weighted_preds, axis=0)
        else:
            # 简单平均
            ensemble_pred = np.mean(predictions, axis=0)
        
        return ensemble_pred


if __name__ == "__main__":
    # 测试代码
    logger.info("测试深度学习模型模块...")
    
    # 创建模拟数据
    np.random.seed(42)
    X_train = np.random.rand(1000, 60, 12)  # 1000个样本，60天，12个特征
    y_train = np.random.rand(1000, 1)
    X_test = np.random.rand(200, 60, 12)
    y_test = np.random.rand(200, 1)
    
    # 创建并构建LSTM模型
    model = StockPredictionModel('test_lstm')
    model.build_lstm_model(input_shape=(60, 12))
    
    # 训练模型（少量epoch用于测试）
    logger.info("开始训练...")
    model.train(X_train, y_train, epochs=5, batch_size=32)
    
    # 评估模型
    metrics = model.evaluate(X_test, y_test)
    logger.info(f"评估指标: {metrics}")
    
    # 保存和加载模型
    model.save_model()
    
    new_model = StockPredictionModel('test_lstm')
    new_model.load_model()
    
    # 预测
    predictions = new_model.predict(X_test[:5])
    logger.info(f"预测结果（前5个）: {predictions.flatten()}")
