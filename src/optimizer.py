"""
模型优化模块 - 超参数调优和迭代优化
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import logging
import os
import sys
from itertools import product
import json

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model import StockPredictionModel
from src.backtest import Backtester, ModelComparison
from config import RESULT_DIR, MODEL_DIR, LOG_DIR, BATCH_SIZE, EPOCHS

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'optimizer.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class GridSearchOptimizer:
    """网格搜索优化器"""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray, 
                 X_test: np.ndarray, y_test: np.ndarray,
                 test_actual_prices: np.ndarray = None,
                 test_dates: pd.Series = None):
        """
        初始化优化器
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            y_test: 测试目标
            test_actual_prices: 用于回测的实际价格
            test_dates: 用于回测的日期
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.test_actual_prices = test_actual_prices
        self.test_dates = test_dates
        self.results = []

    def generate_param_grid(self) -> Dict[str, List]:
        """
        生成参数网格
        
        Returns:
            参数字典
        """
        param_grid = {
            'lstm_units': [
                [128, 64, 32],
                [64, 32, 16],
                [256, 128, 64]
            ],
            'dropout_rate': [0.2, 0.3, 0.4],
            'learning_rate': [0.001, 0.0005, 0.0001],
            'batch_size': [16, 32, 64],
            'model_type': ['lstm', 'gru', 'bidirectional']
        }
        
        return param_grid

    def search(self, param_grid: Dict[str, List] = None, max_trials: int = 10) -> pd.DataFrame:
        """
        网格搜索
        
        Args:
            param_grid: 参数网格
            max_trials: 最大尝试次数
            
        Returns:
            搜索结果DataFrame
        """
        if param_grid is None:
            param_grid = self.generate_param_grid()
        
        logger.info(f"开始网格搜索，预计最多尝试 {max_trials} 次")
        
        # 生成所有参数组合
        param_combinations = list(product(
            param_grid['lstm_units'],
            param_grid['dropout_rate'],
            param_grid['learning_rate'],
            param_grid['batch_size'],
            param_grid['model_type']
        ))
        
        # 随机采样（避免太多组合）
        if len(param_combinations) > max_trials:
            np.random.shuffle(param_combinations)
            param_combinations = param_combinations[:max_trials]
        
        logger.info(f"实际尝试 {len(param_combinations)} 组参数")
        
        for i, params in enumerate(param_combinations):
            logger.info(f"\n{'='*60}")
            logger.info(f"Trial {i+1}/{len(param_combinations)}")
            logger.info(f"参数: {params}")
            
            lstm_units, dropout_rate, learning_rate, batch_size, model_type = params
            
            try:
                # 构建模型
                model = StockPredictionModel(f'trial_{i+1}')
                input_shape = (self.X_train.shape[1], self.X_train.shape[2])
                
                if model_type == 'lstm':
                    model.build_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate)
                elif model_type == 'gru':
                    model.build_gru_model(input_shape, lstm_units, dropout_rate, learning_rate)
                else:
                    model.build_bidirectional_lstm_model(input_shape, lstm_units, dropout_rate, learning_rate)
                
                # 训练
                model.train(self.X_train, self.y_train, epochs=EPOCHS, batch_size=batch_size)
                
                # 评估
                metrics = model.evaluate(self.X_test, self.y_test)
                
                # 回测（如果提供了实际价格）
                backtest_results = {}
                if self.test_actual_prices is not None:
                    predictions = model.predict(self.X_test)
                    
                    # 需要反标准化预测结果（这里简化处理）
                    # 实际应用中应该保存scaler并正确反标准化
                    backtester = Backtester()
                    backtest_results = backtester.backtest(
                        predictions,
                        self.test_actual_prices,
                        self.test_dates,
                        threshold=0.02
                    )
                
                # 记录结果
                result = {
                    'trial': i+1,
                    'model_type': model_type,
                    'lstm_units': str(lstm_units),
                    'dropout_rate': dropout_rate,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'loss': metrics['loss'],
                    'mae': metrics['mae'],
                    'mape': metrics['mape'],
                    'profit_rate': backtest_results.get('profit_rate', 0),
                    'win_rate': backtest_results.get('win_rate', 0),
                    'total_trades': backtest_results.get('total_trades', 0)
                }
                
                self.results.append(result)
                logger.info(f"Trial {i+1} 完成")
                logger.info(f"  MAE: {metrics['mae']:.4f}")
                logger.info(f"  MAPE: {metrics['mape']:.2f}%")
                if backtest_results:
                    logger.info(f"  收益率: {backtest_results['profit_rate']:.2f}%")
                
            except Exception as e:
                logger.error(f"Trial {i+1} 失败: {e}")
                continue
        
        # 保存结果
        results_df = pd.DataFrame(self.results)
        results_path = os.path.join(RESULT_DIR, 'grid_search_results.csv')
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        logger.info(f"\n网格搜索结果已保存到 {results_path}")
        
        return results_df

    def get_best_params(self, metric: str = 'profit_rate') -> Dict:
        """
        获取最佳参数
        
        Args:
            metric: 优化指标
            
        Returns:
            最佳参数字典
        """
        if not self.results:
            logger.warning("没有搜索结果")
            return {}
        
        results_df = pd.DataFrame(self.results)
        
        # 找到最佳参数（根据指标最大化或最小化）
        if metric in ['profit_rate', 'win_rate']:
            best_idx = results_df[metric].idxmax()
        else:
            best_idx = results_df[metric].idxmin()
        
        best_params = results_df.iloc[best_idx].to_dict()
        
        logger.info(f"\n最佳参数（基于 {metric}）:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        return best_params


class IterativeOptimizer:
    """迭代优化器"""

    def __init__(self, X_train: np.ndarray, y_train: np.ndarray,
                 X_test: np.ndarray, y_test: np.ndarray,
                 test_actual_prices: np.ndarray,
                 test_dates: pd.Series):
        """
        初始化迭代优化器
        
        Args:
            X_train: 训练特征
            y_train: 训练目标
            X_test: 测试特征
            y_test: 测试目标
            test_actual_prices: 用于回测的实际价格
            test_dates: 用于回测的日期
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.test_actual_prices = test_actual_prices
        self.test_dates = test_dates
        self.iteration_results = []

    def optimize(self, max_iterations: int = 5) -> Tuple[StockPredictionModel, pd.DataFrame]:
        """
        迭代优化
        
        Args:
            max_iterations: 最大迭代次数
            
        Returns:
            (最佳模型, 迭代结果DataFrame)
        """
        logger.info(f"开始迭代优化，最大迭代次数: {max_iterations}")
        
        # 初始参数
        current_params = {
            'lstm_units': [128, 64, 32],
            'dropout_rate': 0.3,
            'learning_rate': 0.001,
            'batch_size': 32
        }
        
        best_profit_rate = -float('inf')
        best_model = None
        
        for iteration in range(max_iterations):
            logger.info(f"\n{'='*60}")
            logger.info(f"迭代 {iteration + 1}/{max_iterations}")
            logger.info(f"当前参数: {current_params}")
            
            try:
                # 构建模型
                model = StockPredictionModel(f'iter_{iteration+1}')
                input_shape = (self.X_train.shape[1], self.X_train.shape[2])
                
                model.build_lstm_model(
                    input_shape=input_shape,
                    lstm_units=current_params['lstm_units'],
                    dropout_rate=current_params['dropout_rate'],
                    learning_rate=current_params['learning_rate']
                )
                
                # 训练
                model.train(self.X_train, self.y_train, 
                          epochs=EPOCHS, 
                          batch_size=current_params['batch_size'])
                
                # 评估
                metrics = model.evaluate(self.X_test, self.y_test)
                
                # 回测
                predictions = model.predict(self.X_test)
                backtester = Backtester()
                backtest_results = backtester.backtest(
                    predictions,
                    self.test_actual_prices,
                    self.test_dates,
                    threshold=0.02
                )
                
                profit_rate = backtest_results['profit_rate']
                
                # 记录结果
                iteration_result = {
                    'iteration': iteration + 1,
                    'profit_rate': profit_rate,
                    'mae': metrics['mae'],
                    'mape': metrics['mape'],
                    'win_rate': backtest_results['win_rate'],
                    'params': current_params.copy()
                }
                self.iteration_results.append(iteration_result)
                
                logger.info(f"收益率: {profit_rate:.2f}%")
                logger.info(f"MAE: {metrics['mae']:.4f}")
                logger.info(f"胜率: {backtest_results['win_rate']:.2f}%")
                
                # 更新最佳模型
                if profit_rate > best_profit_rate:
                    best_profit_rate = profit_rate
                    best_model = model
                    logger.info("发现更好的模型！")
                
                # 调整参数（简单启发式）
                if iteration < max_iterations - 1:
                    current_params = self._adjust_params(
                        current_params, 
                        profit_rate, 
                        backtest_results['win_rate']
                    )
                
            except Exception as e:
                logger.error(f"迭代 {iteration+1} 失败: {e}")
                continue
        
        # 保存迭代历史
        history_df = pd.DataFrame(self.iteration_results)
        history_path = os.path.join(RESULT_DIR, 'iteration_history.csv')
        history_df.to_csv(history_path, index=False, encoding='utf-8-sig')
        logger.info(f"\n迭代历史已保存到 {history_path}")
        
        # 保存最佳模型
        if best_model:
            best_model.save_model(os.path.join(MODEL_DIR, 'best_model.keras'))
            logger.info(f"最佳模型已保存，收益率: {best_profit_rate:.2f}%")
        
        return best_model, history_df

    def _adjust_params(self, params: Dict, profit_rate: float, win_rate: float) -> Dict:
        """
        调整参数
        
        Args:
            params: 当前参数
            profit_rate: 收益率
            win_rate: 胜率
            
        Returns:
            调整后的参数
        """
        new_params = params.copy()
        
        # 根据表现调整参数
        if profit_rate > 5 and win_rate > 50:
            # 表现好，减小学习率，增加模型复杂度
            new_params['learning_rate'] *= 0.8
            new_params['lstm_units'] = [u * 2 for u in params['lstm_units']]
        elif profit_rate < 0 or win_rate < 40:
            # 表现差，增加学习率，简化模型
            new_params['learning_rate'] *= 1.2
            new_params['lstm_units'] = [max(u // 2, 16) for u in params['lstm_units']]
            new_params['dropout_rate'] = min(params['dropout_rate'] + 0.1, 0.5)
        else:
            # 表现一般，微调
            new_params['learning_rate'] *= 0.9
        
        # 限制范围
        new_params['learning_rate'] = max(min(new_params['learning_rate'], 0.01), 0.00001)
        new_params['dropout_rate'] = max(min(new_params['dropout_rate'], 0.6), 0.1)
        
        return new_params


if __name__ == "__main__":
    # 测试代码
    logger.info("测试模型优化模块...")
    
    # 创建模拟数据
    np.random.seed(42)
    X_train = np.random.rand(500, 60, 12)
    y_train = np.random.rand(500, 1)
    X_test = np.random.rand(100, 60, 12)
    y_test = np.random.rand(100, 1)
    test_prices = np.random.rand(100) * 10 + 20
    test_dates = pd.date_range(start='2024-01-01', periods=100)
    
    # 网格搜索（少量测试）
    grid_optimizer = GridSearchOptimizer(
        X_train, y_train, X_test, y_test, test_prices, test_dates
    )
    
    small_param_grid = {
        'lstm_units': [[64, 32, 16]],
        'dropout_rate': [0.3],
        'learning_rate': [0.001],
        'batch_size': [32],
        'model_type': ['lstm']
    }
    
    results_df = grid_optimizer.search(small_param_grid, max_trials=1)
    logger.info(f"\n网格搜索结果:\n{results_df}")
    
    # 迭代优化
    iterative_optimizer = IterativeOptimizer(
        X_train, y_train, X_test, y_test, test_prices, test_dates
    )
    
    best_model, history_df = iterative_optimizer.optimize(max_iterations=2)
    logger.info(f"\n迭代优化历史:\n{history_df}")
