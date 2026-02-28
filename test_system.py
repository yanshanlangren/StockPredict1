"""
测试脚本 - 快速验证系统功能
使用模拟数据进行测试，避免长时间等待网络请求
"""
import numpy as np
import pandas as pd
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.processor import DataProcessor
from src.model import StockPredictionModel
from src.backtest import Backtester, ModelComparison
from src.optimizer import GridSearchOptimizer, IterativeOptimizer

def generate_mock_stock_data(days=300):
    """生成模拟股票数据"""
    np.random.seed(42)
    
    dates = pd.date_range(start='2023-01-01', periods=days)
    
    # 随机游走价格
    price = 100.0
    prices = []
    for _ in range(days):
        change = np.random.normal(0, 0.02)
        price = price * (1 + change)
        prices.append(max(price, 1.0))  # 确保价格为正
    
    # 生成OHLC数据
    data = {
        'date': dates,
        'open': np.array(prices) * (1 + np.random.uniform(-0.01, 0.01, days)),
        'close': np.array(prices),
        'high': np.array(prices) * (1 + np.random.uniform(0, 0.02, days)),
        'low': np.array(prices) * (1 - np.random.uniform(0, 0.02, days)),
        'volume': np.random.randint(100000, 1000000, days),
        'turnover': np.random.rand(days) * 10000000,
        'amplitude': np.random.rand(days) * 0.05,
        'change_percent': np.random.uniform(-0.05, 0.05, days),
        'change_amount': np.random.uniform(-2, 2, days),
        'turnover_rate': np.random.rand(days) * 10
    }
    
    return pd.DataFrame(data)

def test_pipeline():
    """测试完整流程"""
    print("\n" + "="*60)
    print("开始测试股票交易AI系统")
    print("="*60)
    
    # 1. 生成模拟数据
    print("\n[步骤1] 生成模拟数据...")
    stocks_data = {}
    num_stocks = 5
    
    for i in range(num_stocks):
        stock_data = generate_mock_stock_data(days=300)
        stocks_data[f'60000{i}'] = stock_data
        print(f"  生成股票 60000{i}: {len(stock_data)} 天数据")
    
    # 2. 数据处理
    print("\n[步骤2] 数据处理...")
    processor = DataProcessor()
    
    train_data = processor.prepare_data_for_training(stocks_data, sequence_length=60)
    print(f"  训练数据准备完成:")
    print(f"    X_train: {train_data['X_train'].shape}")
    print(f"    X_test: {train_data['X_test'].shape}")
    
    # 3. 训练模型
    print("\n[步骤3] 训练模型...")
    models = {}
    model_configs = [
        {'name': 'LSTM_128_64_32', 'type': 'lstm', 'units': [128, 64, 32]},
        {'name': 'GRU_128_64_32', 'type': 'gru', 'units': [128, 64, 32]},
    ]
    
    input_shape = (train_data['X_train'].shape[1], train_data['X_train'].shape[2])
    
    for config in model_configs:
        print(f"\n  训练模型: {config['name']}")
        
        model = StockPredictionModel(config['name'])
        
        if config['type'] == 'lstm':
            model.build_lstm_model(input_shape, config['units'])
        else:
            model.build_gru_model(input_shape, config['units'])
        
        # 训练（使用少量epoch进行测试）
        model.train(train_data['X_train'], train_data['y_train'], epochs=5, batch_size=32)
        
        # 评估
        metrics = model.evaluate(train_data['X_test'], train_data['y_test'])
        print(f"    MAE: {metrics['mae']:.4f}")
        print(f"    MAPE: {metrics['mape']:.2f}%")
        
        models[config['name']] = model
    
    # 4. 回测
    print("\n[步骤4] 回测模型...")
    test_stock = '600000'
    X_test, y_test, original_test_data = processor.prepare_data_for_backtest(
        stocks_data[test_stock],
        sequence_length=60,
        test_days=20
    )
    
    comparator = ModelComparison()
    
    for model_name, model in models.items():
        print(f"\n  回测模型: {model_name}")
        
        predictions = model.predict(X_test)
        test_prices = original_test_data['close'].values[-20:]
        test_dates = original_test_data['date'].iloc[-20:]
        
        backtester = Backtester(initial_capital=100000)
        results = backtester.backtest(predictions, test_prices, test_dates, threshold=0.015)
        
        print(f"    收益率: {results['profit_rate']:.2f}%")
        print(f"    胜率: {results['win_rate']:.2f}%")
        print(f"    交易次数: {results['total_trades']}")
        
        comparator.add_result(model_name, results)
    
    # 比较模型
    comparison_df = comparator.compare()
    print("\n模型性能比较:")
    print(comparison_df.to_string(index=False))
    
    # 5. 模型优化（简化测试）
    print("\n[步骤5] 模型优化（网格搜索）...")
    test_prices = original_test_data['close'].values[-20:]
    test_dates = original_test_data['date'].iloc[-20:]
    
    grid_optimizer = GridSearchOptimizer(
        train_data['X_train'],
        train_data['y_train'],
        train_data['X_test'],
        train_data['y_test'],
        test_prices,
        test_dates
    )
    
    small_param_grid = {
        'lstm_units': [[64, 32, 16]],
        'dropout_rate': [0.3],
        'learning_rate': [0.001],
        'batch_size': [32],
        'model_type': ['lstm']
    }
    
    grid_results = grid_optimizer.search(small_param_grid, max_trials=1)
    print("\n网格搜索结果:")
    print(grid_results[['trial', 'model_type', 'mae', 'profit_rate']].to_string(index=False))
    
    # 完成
    print("\n" + "="*60)
    print("测试完成！")
    print("="*60)
    print("\n系统功能验证:")
    print("✓ 数据处理模块: 正常")
    print("✓ 深度学习模型: 正常")
    print("✓ 回测系统: 正常")
    print("✓ 模型优化: 正常")
    print("\n提示: 使用真实数据时，请运行 python main.py")
    print("  可选参数: --stocks N (指定处理股票数量)")
    print("  可选参数: --download-only (仅下载数据)")
    print("  可选参数: --train-only (仅训练模型)")

if __name__ == "__main__":
    test_pipeline()
