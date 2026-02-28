"""
测试AKShare数据获取
"""
import akshare as ak
import pandas as pd
from datetime import datetime
import time

print("="*80)
print("AKShare 数据获取测试")
print("="*80)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 测试1: 获取A股历史行情
print("\n" + "="*80)
print("测试1: 获取A股历史行情数据")
print("="*80)

test_stocks = ['600000', '600001', '600004', '600005', '600006']
stock_names = ['浦发银行', '邯郸钢铁', '白云机场', '武钢股份', '东风汽车']

results = []

for i, stock_code in enumerate(test_stocks):
    stock_name = stock_names[i]
    print(f"\n正在获取 {stock_code} ({stock_name})...")
    start_time = time.time()

    try:
        # 使用AKShare获取历史数据
        # ak.stock_zh_a_hist: 获取A股历史行情数据
        df = ak.stock_zh_a_hist(
            symbol=stock_code,
            period="daily",  # 日线数据
            start_date="20240101",  # 开始日期
            end_date="20251231",    # 结束日期
            adjust="qfq"  # 前复权
        )

        elapsed_time = time.time() - start_time

        if not df.empty:
            print(f"  ✓ 成功获取 {len(df)} 天数据")
            print(f"  耗时: {elapsed_time:.2f}秒")
            print(f"  列数: {len(df.columns)}")
            print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")
            print(f"  前5行数据:")
            print(df.head())

            # 保存到文件
            output_file = f'data/raw/akshare_{stock_code}.csv'
            df.to_csv(output_file)
            print(f"  已保存到: {output_file}")

            results.append({
                'code': stock_code,
                'name': stock_name,
                'status': 'success',
                'days': len(df),
                'time': elapsed_time
            })
        else:
            print(f"  ✗ 返回空数据")
            results.append({
                'code': stock_code,
                'name': stock_name,
                'status': 'empty',
                'days': 0,
                'time': elapsed_time
            })

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        print(f"  ✗ 失败: {error_msg[:100]}")
        print(f"  耗时: {elapsed_time:.2f}秒")
        results.append({
            'code': stock_code,
            'name': stock_name,
            'status': 'error',
            'error': error_msg[:100],
            'days': 0,
            'time': elapsed_time
        })

    # 添加延迟避免请求过快
    time.sleep(1)

# 统计结果
print("\n" + "="*80)
print("测试结果统计")
print("="*80)

success_count = sum(1 for r in results if r['status'] == 'success')
empty_count = sum(1 for r in results if r['status'] == 'empty')
error_count = sum(1 for r in results if r['status'] == 'error')

print(f"总尝试: {len(test_stocks)} 只股票")
print(f"成功: {success_count} 只")
print(f"空数据: {empty_count} 只")
print(f"错误: {error_count} 只")

if len(test_stocks) > 0:
    total_time = sum(r['time'] for r in results)
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
    if r['status'] == 'success':
        status_icon = "✓"
        print(f"{status_icon} {r['code']} ({r['name']}): 成功 ({r['days']}天, {r['time']:.2f}秒)")
    elif r['status'] == 'empty':
        status_icon = "⚠"
        print(f"{status_icon} {r['code']} ({r['name']}): 空数据 ({r['time']:.2f}秒)")
    else:
        status_icon = "✗"
        print(f"{status_icon} {r['code']} ({r['name']}): {r.get('error', '未知错误')} ({r['time']:.2f}秒)")

# 测试2: 检查数据格式
print("\n" + "="*80)
print("测试2: 检查数据格式")
print("="*80)

if success_count > 0:
    # 使用第一个成功的数据检查格式
    success_result = next(r for r in results if r['status'] == 'success')
    stock_code = success_result['code']

    print(f"\n使用 {stock_code} 的数据检查格式...")

    df = pd.read_csv(f'data/raw/akshare_{stock_code}.csv', index_col=0)
    print(f"\n数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"数据类型:")
    print(df.dtypes)
    print(f"\n基本统计:")
    print(df.describe())

# 结论
print("\n" + "="*80)
print("测试结论")
print("="*80)

if success_count == len(test_stocks):
    print("✓ AKShare完全可用，可以正常获取所有数据")
    print("\n建议行动:")
    print("  1. 创建AKShare数据适配器")
    print("  2. 集成到现有系统")
    print("  3. 实现数据源切换机制")
elif success_count > 0:
    print(f"⚠ AKShare部分可用，成功率 {success_count/len(test_stocks)*100:.1f}%")
    print("\n建议行动:")
    print("  1. 使用AKShare作为主数据源")
    print("  2. 模拟数据作为备用")
    print("  3. 检查失败原因")
else:
    print("✗ AKShare当前不可用")
    print("\n建议行动:")
    print("  1. 继续使用模拟数据")
    print("  2. 检查AKShare版本和网络")
    print("  3. 考虑其他数据源")

print("\n" + "="*80)
