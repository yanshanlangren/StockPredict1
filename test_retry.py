"""
测试重试机制
"""
import sys
sys.path.insert(0, '.')

from src.crawler import EastMoneyCrawler
import logging

logging.basicConfig(level=logging.INFO)

print("测试重试机制...")
print("="*60)

crawler = EastMoneyCrawler()

print("\n测试 1: 获取股票列表（会自动重试）")
try:
    stock_list = crawler.get_stock_list()
    if not stock_list.empty:
        print(f"✓ 成功获取 {len(stock_list)} 只股票")
    else:
        print("✓ 未获取到股票数据（API返回空）")
except Exception as e:
    print(f"✗ 最终失败: {e}")

print("\n测试 2: 获取单只股票K线（会自动重试）")
try:
    kline_data = crawler.get_stock_kline('600000')
    if not kline_data.empty:
        print(f"✓ 成功获取 {len(kline_data)} 天数据")
    else:
        print("✓ 未获取到K线数据")
except Exception as e:
    print(f"✗ 最终失败: {e}")

print("\n" + "="*60)
print("测试完成")
print("如果看到多次 '请求失败，第X/Y次重试' 的日志，说明重试机制正常工作")
