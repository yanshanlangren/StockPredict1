"""
股票交易AI系统 - 主程序入口
整合数据爬取、模型训练、回测和优化
"""
import os
import sys
import logging
import argparse
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.crawler import EastMoneyCrawler
from src.processor import DataProcessor
from src.model import StockPredictionModel, MultiModelEnsemble
from src.backtest import Backtester, ModelComparison
from src.optimizer import GridSearchOptimizer, IterativeOptimizer
from config import (
    TRAIN_DAYS, TEST_DAYS, RAW_DATA_DIR, 
    MODEL_DIR, RESULT_DIR, LOG_DIR
)

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'main.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class StockTradingAI:
    """股票交易AI系统主类"""

    def __init__(self, max_stocks: int = 20):
        """
        初始化系统
        
        Args:
            max_stocks: 最大处理股票数量
        """
        self.max_stocks = max_stocks
        self.crawler = EastMoneyCrawler()
        self.processor = DataProcessor()
        self.models = {}
        self.backtest_results = {}

    def step1_download_data(self, force_redownload: bool = False) -> dict:
        """
        步骤1: 下载数据
        """
        logger.info("\n" + "="*60)
        logger.info("步骤1: 下载股票数据")
        logger.info("="*60)
        
        # 获取股票列表
        stock_list_df = self.crawler.get_stock_list()
        
        if stock_list_df.empty:
            logger.error("获取股票列表失败")
            return {}
        
        logger.info(f"共获取到 {len(stock_list_df)} 只股票")
        
        # 选择前N只股票进行训练（避免数据量过大）
        stock_codes = stock_list_df['code'].head(self.max_stocks).tolist()
        logger.info(f"将下载前 {len(stock_codes)} 只股票的数据")
        
        # 批量下载
        stocks_data = self.crawler.batch_download_stocks(stock_codes, max_stocks=self.max_stocks)
        
        # 保存数据
        if stocks_data:
            self.crawler.save_data(stocks_data, 'stock_data.csv')
            logger.info(f"成功下载 {len(stocks_data)} 只股票的数据")
        else:
            logger.error("没有成功下载任何股票数据")
        
        return stocks_data

    def step2_train_models(self, stocks_data: dict, use_existing: bool = True) -> dict:
        """
        步骤2: 训练多个模型
        """
        logger.info("\n" + "="*60)
        logger.info("步骤2: 训练深度学习模型")
        logger.info("="*60)
        
        # 准备训练数据
        train_data = self.processor.prepare_data_for_training(stocks_data, sequence_length=TRAIN_DAYS)
        
        # 输入形状
        input_shape = (train_data['X_train'].shape[1], train_data['X_train'].shape[2])
        logger.info(f"输入形状: {input_shape}")
        
        # 创建多个模型
        model_configs = [
            {'name': 'LSTM_128_64_32', 'type': 'lstm', 'units': [128, 64, 32]},
            {'name': 'LSTM_64_32_16', 'type': 'lstm', 'units': [64, 32, 16]},
            {'name': 'GRU_128_64_32', 'type': 'gru', 'units': [128, 64, 32]},
            {'name': 'BiLSTM_128_64', 'type': 'bidirectional', 'units': [128, 64]},
        ]
        
        for config in model_configs:
            model_name = config['name']
            model_path = os.path.join(MODEL_DIR, f'{model_name}.keras')
            
            # 检查是否已存在模型
            if use_existing and os.path.exists(model_path):
                logger.info(f"模型 {model_name} 已存在，跳过训练")
                model = StockPredictionModel(model_name)
                model.load_model(model_path)
                self.models[model_name] = model
                continue
            
            logger.info(f"\n开始训练模型: {model_name}")
            
            # 创建模型
            model = StockPredictionModel(model_name)
            
            if config['type'] == 'lstm':
                model.build_lstm_model(input_shape, config['units'])
            elif config['type'] == 'gru':
                model.build_gru_model(input_shape, config['units'])
            else:
                model.build_bidirectional_lstm_model(input_shape, config['units'])
            
            # 训练
            model.train(
                train_data['X_train'],
                train_data['y_train'],
                epochs=50,
                batch_size=32
            )
            
            # 评估
            metrics = model.evaluate(
                train_data['X_test'],
                train_data['y_test']
            )
            
            # 保存模型
            model.save_model()
            
            self.models[model_name] = model
        
        logger.info(f"\n所有模型训练完成！共 {len(self.models)} 个模型")
        
        return train_data

    def step3_backtest(self, stocks_data: dict, train_data: dict) -> ModelComparison:
        """
        步骤3: 回测模型
        """
        logger.info("\n" + "="*60)
        logger.info("步骤3: 回测模型")
        logger.info("="*60)
        
        # 选择一只股票进行回测（选择数据量最大的）
        best_stock_code = max(stocks_data.items(), key=lambda x: len(x[1]))[0]
        logger.info(f"选择股票 {best_stock_code} 进行回测")
        
        # 准备回测数据
        X_test, y_test, original_test_data = self.processor.prepare_data_for_backtest(
            stocks_data[best_stock_code],
            sequence_length=TRAIN_DAYS,
            test_days=TEST_DAYS
        )
        
        logger.info(f"回测数据准备完成: X_test={X_test.shape}")
        
        # 模型比较器
        comparator = ModelComparison()
        
        # 对每个模型进行回测
        for model_name, model in self.models.items():
            logger.info(f"\n回测模型: {model_name}")
            
            # 预测
            predictions = model.predict(X_test)
            
            # 反标准化预测结果（这里简化处理）
            # 实际应用中应该保存scaler并正确反标准化
            test_prices = original_test_data['close'].values[-TEST_DAYS:]
            test_dates = original_test_data['date'].iloc[-TEST_DAYS:]
            
            # 执行回测
            backtester = Backtester()
            results = backtester.backtest(
                predictions,
                test_prices,
                test_dates,
                threshold=0.02
            )
            
            # 保存结果
            backtester.save_results(results, model_name)
            
            # 添加到比较器
            comparator.add_result(model_name, results)
            
            self.backtest_results[model_name] = results
        
        # 比较模型
        comparison_df = comparator.compare()
        
        # 获取最佳模型
        best_model_name, best_result = comparator.get_best_model()
        
        return comparator

    def step4_optimize(self, stocks_data: dict, train_data: dict):
        """
        步骤4: 模型优化
        """
        logger.info("\n" + "="*60)
        logger.info("步骤4: 模型优化")
        logger.info("="*60)
        
        # 选择一只股票进行优化
        best_stock_code = max(stocks_data.items(), key=lambda x: len(x[1]))[0]
        
        # 准备优化数据
        X_test, y_test, original_test_data = self.processor.prepare_data_for_backtest(
            stocks_data[best_stock_code],
            sequence_length=TRAIN_DAYS,
            test_days=TEST_DAYS
        )
        
        test_prices = original_test_data['close'].values[-TEST_DAYS:]
        test_dates = original_test_data['date'].iloc[-TEST_DAYS:]
        
        # 网格搜索
        logger.info("\n开始网格搜索优化...")
        grid_optimizer = GridSearchOptimizer(
            train_data['X_train'],
            train_data['y_train'],
            train_data['X_test'],
            train_data['y_test'],
            test_prices,
            test_dates
        )
        
        # 定义较小的搜索空间（避免运行时间过长）
        small_param_grid = {
            'lstm_units': [[128, 64, 32], [64, 32, 16]],
            'dropout_rate': [0.2, 0.3],
            'learning_rate': [0.001, 0.0005],
            'batch_size': [32, 64],
            'model_type': ['lstm', 'gru']
        }
        
        grid_results = grid_optimizer.search(small_param_grid, max_trials=5)
        
        # 获取最佳参数
        best_params = grid_optimizer.get_best_params(metric='profit_rate')
        
        # 迭代优化
        logger.info("\n开始迭代优化...")
        iterative_optimizer = IterativeOptimizer(
            train_data['X_train'],
            train_data['y_train'],
            train_data['X_test'],
            train_data['y_test'],
            test_prices,
            test_dates
        )
        
        best_model, history_df = iterative_optimizer.optimize(max_iterations=3)
        
        return grid_results, history_df

    def run_full_pipeline(self):
        """
        运行完整流程
        """
        logger.info("\n" + "="*60)
        logger.info("开始运行股票交易AI系统")
        logger.info(f"最大处理股票数: {self.max_stocks}")
        logger.info(f"训练天数: {TRAIN_DAYS}, 测试天数: {TEST_DAYS}")
        logger.info("="*60)
        
        try:
            # 步骤1: 下载数据
            stocks_data = self.step1_download_data()
            
            if not stocks_data:
                logger.error("数据下载失败，终止流程")
                return
            
            # 步骤2: 训练模型
            train_data = self.step2_train_models(stocks_data)
            
            # 步骤3: 回测
            comparator = self.step3_backtest(stocks_data, train_data)
            
            # 步骤4: 优化
            grid_results, history_df = self.step4_optimize(stocks_data, train_data)
            
            # 总结
            logger.info("\n" + "="*60)
            logger.info("流程完成！")
            logger.info("="*60)
            logger.info(f"训练了 {len(self.models)} 个模型")
            logger.info(f"所有结果已保存到 {RESULT_DIR}")
            logger.info(f"所有模型已保存到 {MODEL_DIR}")
            
        except Exception as e:
            logger.error(f"流程执行失败: {e}", exc_info=True)
            raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='股票交易AI系统')
    parser.add_argument('--stocks', type=int, default=20, 
                       help='最大处理股票数量 (默认: 20)')
    parser.add_argument('--download-only', action='store_true',
                       help='仅下载数据')
    parser.add_argument('--train-only', action='store_true',
                       help='仅训练模型')
    
    args = parser.parse_args()
    
    # 创建系统实例
    ai_system = StockTradingAI(max_stocks=args.stocks)
    
    # 执行对应功能
    if args.download_only:
        ai_system.step1_download_data()
    elif args.train_only:
        # 需要先有数据
        stocks_data = ai_system.step1_download_data()
        if stocks_data:
            ai_system.step2_train_models(stocks_data)
    else:
        # 运行完整流程
        ai_system.run_full_pipeline()


if __name__ == "__main__":
    main()
