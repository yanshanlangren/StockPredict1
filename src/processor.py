"""
数据处理模块 - 数据清洗和特征工程
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List, Dict
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import PROCESSED_DATA_DIR, TRAIN_DAYS, TEST_DAYS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataProcessor:
    """数据处理器"""

    def __init__(self):
        self.scalers = {}

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            清洗后的DataFrame
        """
        # 复制数据
        df_clean = df.copy()
        
        # 删除重复数据
        df_clean = df_clean.drop_duplicates()
        
        # 检查缺失值
        if df_clean.isnull().sum().sum() > 0:
            logger.warning(f"发现缺失值，将进行填充")
            # 向前填充
            df_clean = df_clean.fillna(method='ffill')
            # 如果还有缺失，向后填充
            df_clean = df_clean.fillna(method='bfill')
        
        # 确保日期排序
        if 'date' in df_clean.columns:
            df_clean = df_clean.sort_values('date').reset_index(drop=True)
        
        # 删除异常值（价格为0或负数）
        price_cols = ['open', 'close', 'high', 'low']
        for col in price_cols:
            if col in df_clean.columns:
                df_clean = df_clean[df_clean[col] > 0]
        
        return df_clean

    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        添加技术指标
        
        Args:
            df: 原始数据DataFrame
            
        Returns:
            添加技术指标后的DataFrame
        """
        df = df.copy()
        
        # 移动平均线
        df['ma5'] = df['close'].rolling(window=5).mean()
        df['ma10'] = df['close'].rolling(window=10).mean()
        df['ma20'] = df['close'].rolling(window=20).mean()
        df['ma60'] = df['close'].rolling(window=60).mean()
        
        # 相对强弱指标 (RSI)
        df['rsi'] = self._calculate_rsi(df['close'], period=14)
        
        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = self._calculate_macd(df['close'])
        
        # 布林带
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = self._calculate_bollinger_bands(df['close'], period=20)
        
        # 成交量移动平均
        df['volume_ma5'] = df['volume'].rolling(window=5).mean()
        df['volume_ma10'] = df['volume'].rolling(window=10).mean()
        
        # 价格变化率
        df['price_change_1d'] = df['close'].pct_change(1)
        df['price_change_5d'] = df['close'].pct_change(5)
        df['price_change_10d'] = df['close'].pct_change(10)
        
        # 振幅
        df['amplitude'] = (df['high'] - df['low']) / df['close']
        
        # 涨跌幅
        df['up_down'] = np.where(df['close'] >= df['open'], 1, 0)
        
        # 删除前面因计算指标产生的NaN
        df = df.dropna().reset_index(drop=True)
        
        return df

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """计算RSI指标"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_macd(self, prices: pd.Series, fast=12, slow=26, signal=9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算MACD指标"""
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal, adjust=False).mean()
        macd_hist = macd - macd_signal
        
        return macd, macd_signal, macd_hist

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """计算布林带"""
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        
        return upper, sma, lower

    def normalize_data(self, df: pd.DataFrame, feature_cols: List[str], method: str = 'minmax') -> Tuple[pd.DataFrame, Dict]:
        """
        标准化数据
        
        Args:
            df: 原始数据
            feature_cols: 特征列名列表
            method: 标准化方法，'minmax'或'standard'
            
        Returns:
            标准化后的数据和scaler字典
        """
        df_normalized = df.copy()
        scalers = {}
        
        for col in feature_cols:
            if method == 'minmax':
                scaler = MinMaxScaler()
            else:
                scaler = StandardScaler()
            
            # 使用训练数据拟合（假设前面80%是训练数据）
            train_size = int(len(df) * 0.8)
            scaler.fit(df[col].iloc[:train_size].values.reshape(-1, 1))
            
            df_normalized[col] = scaler.transform(df[col].values.reshape(-1, 1)).flatten()
            scalers[col] = scaler
        
        return df_normalized, scalers

    def create_sequences(self, df: pd.DataFrame, feature_cols: List[str], target_col: str, 
                        sequence_length: int = 60) -> Tuple[np.ndarray, np.ndarray]:
        """
        创建序列数据用于LSTM训练
        
        Args:
            df: 数据DataFrame
            feature_cols: 特征列名列表
            target_col: 目标列名
            sequence_length: 序列长度
            
        Returns:
            (X, y) 特征序列和目标值
        """
        sequences = []
        targets = []
        
        data = df[feature_cols + [target_col]].values
        
        for i in range(len(data) - sequence_length):
            sequences.append(data[i:i + sequence_length, :-1])  # 特征
            targets.append(data[i + sequence_length, -1])  # 目标值
        
        return np.array(sequences), np.array(targets)

    def split_data(self, X: np.ndarray, y: np.ndarray, test_ratio: float = 0.2) -> Tuple:
        """
        分割训练集和测试集
        
        Args:
            X: 特征数据
            y: 目标数据
            test_ratio: 测试集比例
            
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        split_idx = int(len(X) * (1 - test_ratio))
        
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return X_train, X_test, y_train, y_test

    def prepare_data_for_training(self, stock_data: Dict[str, pd.DataFrame], 
                                  sequence_length: int = TRAIN_DAYS) -> Dict[str, np.ndarray]:
        """
        准备训练数据
        
        Args:
            stock_data: 股票数据字典
            sequence_length: 序列长度
            
        Returns:
            包含训练数据的字典
        """
        all_X = []
        all_y = []
        
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma10', 'ma20', 
                       'rsi', 'macd', 'macd_signal', 'macd_hist']
        target_col = 'close'
        
        for stock_code, df in stock_data.items():
            try:
                # 清洗数据
                df_clean = self.clean_data(df)
                
                # 添加技术指标
                df_features = self.add_technical_indicators(df_clean)
                
                # 标准化
                df_normalized, _ = self.normalize_data(df_features, feature_cols)
                
                # 创建序列
                X, y = self.create_sequences(df_normalized, feature_cols, target_col, sequence_length)
                
                if len(X) > 0:
                    all_X.append(X)
                    all_y.append(y)
                    
            except Exception as e:
                logger.warning(f"处理股票 {stock_code} 时出错: {e}")
                continue
        
        # 合并所有股票数据
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)
        
        # 分割数据集
        X_train, X_test, y_train, y_test = self.split_data(X_combined, y_combined, test_ratio=0.2)
        
        logger.info(f"数据准备完成:")
        logger.info(f"  训练集: X_train={X_train.shape}, y_train={y_train.shape}")
        logger.info(f"  测试集: X_test={X_test.shape}, y_test={y_test.shape}")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }

    def prepare_data_for_backtest(self, df: pd.DataFrame, sequence_length: int = TRAIN_DAYS,
                                  test_days: int = TEST_DAYS) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        准备回测数据
        
        Args:
            df: 单只股票数据
            sequence_length: 序列长度
            test_days: 测试天数
            
        Returns:
            (X_test, y_test, original_data)
        """
        # 清洗数据
        df_clean = self.clean_data(df)
        
        # 添加技术指标
        df_features = self.add_technical_indicators(df_clean)
        
        # 保留最后test_days天的原始数据用于回测
        if len(df_features) < sequence_length + test_days:
            raise ValueError(f"数据不足，至少需要 {sequence_length + test_days} 天数据")
        
        test_data = df_features.iloc[-(sequence_length + test_days):]
        original_test_data = test_data.copy()
        
        feature_cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma10', 'ma20', 
                       'rsi', 'macd', 'macd_signal', 'macd_hist']
        target_col = 'close'
        
        # 标准化（仅使用前面的数据）
        df_normalized, _ = self.normalize_data(df_features, feature_cols)
        test_normalized = df_normalized.iloc[-(sequence_length + test_days):]
        
        # 创建序列（滑动窗口）
        X_test = []
        y_test = []
        
        for i in range(test_days):
            start_idx = i
            end_idx = i + sequence_length
            X_test.append(test_normalized.iloc[start_idx:end_idx][feature_cols].values)
            y_test.append(test_normalized.iloc[end_idx][target_col])
        
        return np.array(X_test), np.array(y_test), original_test_data


if __name__ == "__main__":
    # 测试代码
    from crawler import EastMoneyCrawler
    
    logger.info("测试数据处理模块...")
    
    # 创建测试数据
    test_data = {
        'date': pd.date_range(start='2023-01-01', periods=100),
        'open': np.random.rand(100) * 10 + 20,
        'close': np.random.rand(100) * 10 + 20,
        'high': np.random.rand(100) * 10 + 25,
        'low': np.random.rand(100) * 10 + 15,
        'volume': np.random.randint(100000, 1000000, 100),
        'turnover': np.random.rand(100) * 10000000
    }
    df = pd.DataFrame(test_data)
    
    processor = DataProcessor()
    
    # 清洗数据
    df_clean = processor.clean_data(df)
    logger.info(f"清洗后数据: {len(df_clean)} 行")
    
    # 添加技术指标
    df_features = processor.add_technical_indicators(df_clean)
    logger.info(f"添加技术指标后列数: {len(df_features.columns)}")
    logger.info(f"列名: {list(df_features.columns)}")
    
    # 创建序列
    feature_cols = ['open', 'high', 'low', 'close', 'volume', 'ma5', 'ma10', 'ma20']
    X, y = processor.create_sequences(df_features, feature_cols, 'close', sequence_length=20)
    logger.info(f"序列数据 shape: X={X.shape}, y={y.shape}")
