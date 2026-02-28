"""
测试下载真实数据 - 重点测试
"""
import sys
sys.path.insert(0, '.')

from src.crawler import EastMoneyCrawler
import pandas as pd
import time
from datetime import datetime

print("="*80)
print("东方财富数据下载测试")
print("="*80)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

crawler = EastMoneyCrawler()

# 测试1: 下载1只股票
print("\n" + "="*80)
print("测试1: 下载1只股票数据 (600000 浦发银行)")
print("="*80)

start_time = time.time()
try:
    kline_data = crawler.get_stock_kline('600000')
    elapsed_time = time.time() - start_time
    
    if not kline_data.empty:
        print(f"✓ 下载成功!")
        print(f"  耗时: {elapsed_time:.2f}秒")
        print(f"  数据行数: {len(kline_data)}")
        print(f"  列数: {len(kline_data.columns)}")
        print(f"  时间范围: {kline_data.index[0]} 到 {kline_data.index[-1]}")
        print(f"  前5行数据:")
        print(kline_data.head())
        
        # 保存到文件
        output_file = 'data/raw/test_600000.csv'
        kline_data.to_csv(output_file)
        print(f"\n  已保存到: {output_file}")
    else:
        print(f"✗ 下载失败 - 返回空数据")
        print(f"  耗时: {elapsed_time:.2f}秒")
        
except Exception as e:
    elapsed_time = time.time() - start_time
    print(f"✗ 下载失败: {e}")
    print(f"  耗时: {elapsed_time:.2f}秒")

# 等待一会儿
print("\n等待5秒后进行下一个测试...")
time.sleep(5)

# 测试2: 尝试下载5只股票
print("\n" + "="*80)
print("测试2: 批量下载5只股票")
print("="*80)

test_stocks = ['600000', '600001', '600004', '600005', '600006']
success_count = 0
fail_count = 0
start_time = time.time()

results = []

for stock_code in test_stocks:
    stock_start = time.time()
    try:
        print(f"\n正在下载 {stock_code}...")
        kline_data = crawler.get_stock_kline(stock_code)
        stock_elapsed = time.time() - stock_start
        
        if not kline_data.empty:
            print(f"  ✓ 成功 ({len(kline_data)}天, {stock_elapsed:.2f}秒)")
            success_count += 1
            results.append({
                'code': stock_code,
                'status': 'success',
                'days': len(kline_data),
                'time': stock_elapsed
            })
        else:
            print(f"  ✗ 空数据 ({stock_elapsed:.2f}秒)")
            fail_count += 1
            results.append({
                'code': stock_code,
                'status': 'empty',
                'days': 0,
                'time': stock_elapsed
            })
    except Exception as e:
        stock_elapsed = time.time() - stock_start
        print(f"  ✗ 失败: {str(e)[:50]}... ({stock_elapsed:.2f}秒)")
        fail_count += 1
        results.append({
            'code': stock_code,
            'status': 'error',
            'days': 0,
            'time': stock_elapsed
        })

total_time = time.time() - start_time

# 输出统计
print("\n" + "="*80)
print("测试结果统计")
print("="*80)
print(f"总尝试: {len(test_stocks)} 只股票")
print(f"成功: {success_count} 只")
print(f"失败: {fail_count} 只")
print(f"总耗时: {total_time:.2f}秒")
print(f"平均耗时: {total_time/len(test_stocks):.2f}秒/股")

if success_count > 0:
    print(f"\n成功率: {success_count/len(test_stocks)*100:.1f}%")
else:
    print(f"\n成功率: 0%")

# 详细结果
print("\n详细结果:")
print("-"*80)
for r in results:
    status_icon = "✓" if r['status'] == 'success' else "✗"
    print(f"{status_icon} {r['code']}: {r['status']} ({r['days']}天, {r['time']:.2f}秒)")

# 结论
print("\n" + "="*80)
print("测试结论")
print("="*80)

if success_count == len(test_stocks):
    print("✓ 东方财富API完全可用，可以正常下载所有数据")
elif success_count > 0:
    print(f"⚠ 东方财富API部分可用，成功率 {success_count/len(test_stocks)*100:.1f}%")
    print("  建议: 使用模拟数据，或者少量重试下载失败的数据")
else:
    print("✗ 东方财富API当前不可用")
    print("  建议: 使用模拟数据进行开发和测试")
    print("  命令: python test_system.py")

print("\n" + "="*80)
