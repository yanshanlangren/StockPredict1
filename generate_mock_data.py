"""
备用数据生成器 - 当网络不可用时提供模拟数据
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def generate_mock_stock_data(stock_code: str, days: int = 300) -> pd.DataFrame:
    """
    生成模拟股票数据
    
    Args:
        stock_code: 股票代码
        days: 生成天数
        
    Returns:
        模拟股票数据DataFrame
    """
    np.random.seed(hash(stock_code) % (2**32))
    
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
    
    df = pd.DataFrame(data)
    return df

def generate_mock_dataset(num_stocks: int = 20) -> dict:
    """
    生成模拟股票数据集
    
    Args:
        num_stocks: 股票数量
        
    Returns:
        股票数据字典 {stock_code: DataFrame}
    """
    print(f"\n生成 {num_stocks} 只股票的模拟数据...")
    
    stocks_data = {}
    for i in range(num_stocks):
        stock_code = f'6000{i:02d}'  # 生成股票代码：600000, 600001, ...
        df = generate_mock_stock_data(stock_code, days=300)
        stocks_data[stock_code] = df
        print(f"  ✓ 生成 {stock_code}: {len(df)} 天数据")
    
    print(f"\n成功生成 {len(stocks_data)} 只股票的模拟数据")
    return stocks_data

def save_mock_data(stocks_data: dict, filename: str = 'mock_stock_data.csv'):
    """
    保存模拟数据
    
    Args:
        stocks_data: 股票数据字典
        filename: 文件名
    """
    from config import RAW_DATA_DIR
    
    filepath = os.path.join(RAW_DATA_DIR, filename)
    
    # 合并所有股票数据
    all_data = []
    for stock_code, df in stocks_data.items():
        df_copy = df.copy()
        df_copy['stock_code'] = stock_code
        all_data.append(df_copy)
    
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        merged_df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"\n模拟数据已保存到 {filepath}")

if __name__ == "__main__":
    print("="*60)
    print("模拟数据生成器")
    print("="*60)
    print("\n当网络不可用时，使用模拟数据可以：")
    print("  ✓ 测试系统所有功能")
    print("  ✓ 训练深度学习模型")
    print("  ✓ 执行回测")
    print("  ✓ 优化模型参数")
    
    # 生成20只股票的模拟数据
    stocks_data = generate_mock_dataset(num_stocks=20)
    
    # 保存数据
    save_mock_data(stocks_data, 'mock_stock_data.csv')
    
    print("\n" + "="*60)
    print("生成完成！")
    print("="*60)
    print("\n现在可以使用这些模拟数据运行系统：")
    print("  python main.py --train-only")
    print("\n或者使用测试脚本：")
    print("  python test_system.py")
