"""
测试腾讯财经数据批量下载
"""
import akshare as ak
import pandas as pd
from datetime import datetime
import time
import os

print("="*80)
print("腾讯财经数据批量下载测试")
print("="*80)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 确保输出目录存在
os.makedirs('data/raw', exist_ok=True)

# 测试批量下载
test_stocks = [
    ('sh600000', '浦发银行'),
    ('sh600001', '邯郸钢铁'),
    ('sh600004', '白云机场'),
    ('sh600005', '武钢股份'),
    ('sh600006', '东风汽车'),
    ('sh600007', '中国国贸'),
    ('sh600008', '首创股份'),
    ('sh600009', '上海机场'),
    ('sh600010', '包钢股份'),
    ('sh600011', '华能国际'),
]

results = []

print(f"\n计划下载 {len(test_stocks)} 只股票...")
print("-"*80)

for i, (stock_code, stock_name) in enumerate(test_stocks, 1):
    print(f"\n[{i}/{len(test_stocks)}] 正在获取 {stock_code} ({stock_name})...")
    start_time = time.time()

    try:
        # 使用腾讯财经数据源
        df = ak.stock_zh_a_daily(
            symbol=stock_code,
            start_date="20240101",
            end_date="20251231",
            adjust="qfq"
        )

        elapsed_time = time.time() - start_time

        if not df.empty:
            print(f"  ✓ 成功! ({len(df)}天, {elapsed_time:.2f}秒)")

            # 标准化列名以匹配系统格式
            # 系统期望的列名: open, high, low, close, volume, amount
            df = df.reset_index()
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')

            # 重命名列
            column_mapping = {
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }

            # 确保所有需要的列都存在
            available_columns = {k: v for k, v in column_mapping.items() if k in df.columns}
            df = df[list(available_columns.keys())].rename(columns=available_columns)

            # 计算涨跌幅
            if 'close' in df.columns:
                df['change_pct'] = df['close'].pct_change() * 100

            # 保存到文件
            output_file = f'data/raw/tencent_{stock_code}.csv'
            df.to_csv(output_file)
            print(f"  已保存到: {output_file}")

            results.append({
                'code': stock_code,
                'name': stock_name,
                'status': 'success',
                'days': len(df),
                'time': elapsed_time,
                'file': output_file
            })
        else:
            print(f"  ⚠ 返回空数据 ({elapsed_time:.2f}秒)")
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
        print(f"  ✗ 失败: {error_msg[:80]}...")
        results.append({
            'code': stock_code,
            'name': stock_name,
            'status': 'error',
            'error': error_msg[:80],
            'time': elapsed_time
        })

    # 延迟避免请求过快
    time.sleep(0.5)

# 统计结果
print("\n" + "="*80)
print("批量下载统计")
print("="*80)

success_count = sum(1 for r in results if r['status'] == 'success')
empty_count = sum(1 for r in results if r['status'] == 'empty')
error_count = sum(1 for r in results if r['status'] == 'error')

print(f"总尝试: {len(test_stocks)} 只股票")
print(f"成功: {success_count} 只")
print(f"空数据: {empty_count} 只")
print(f"错误: {error_count} 只")

if success_count > 0:
    total_time = sum(r['time'] for r in results if r['status'] == 'success')
    avg_time = total_time / success_count
    total_days = sum(r['days'] for r in results if r['status'] == 'success')

    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均耗时: {avg_time:.2f}秒/股")
    print(f"总数据量: {total_days} 天")
    print(f"成功率: {success_count/len(test_stocks)*100:.1f}%")

# 详细结果
print("\n详细结果:")
print("-"*80)
for r in results:
    if r['status'] == 'success':
        status_icon = "✓"
        print(f"{status_icon} {r['code']:12} {r['name']:8} - {r['days']:3}天, {r['time']:.2f}秒")
    elif r['status'] == 'empty':
        status_icon = "⚠"
        print(f"{status_icon} {r['code']:12} {r['name']:8} - 空数据, {r['time']:.2f}秒")
    else:
        status_icon = "✗"
        error = r.get('error', '未知错误')
        print(f"{status_icon} {r['code']:12} {r['name']:8} - {error}")

# 检查数据质量
if success_count > 0:
    print("\n" + "="*80)
    print("数据质量检查")
    print("="*80)

    # 使用第一个成功的数据检查
    success_result = next(r for r in results if r['status'] == 'success')
    df = pd.read_csv(success_result['file'], index_col=0, parse_dates=True)

    print(f"\n示例数据: {success_result['code']} ({success_result['name']})")
    print(f"数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    print(f"时间范围: {df.index[0]} 到 {df.index[-1]}")
    print(f"缺失值: {df.isnull().sum().sum()}")
    print(f"\n前5行数据:")
    print(df.head())
    print(f"\n基本统计:")
    print(df.describe())

# 结论
print("\n" + "="*80)
print("测试结论")
print("="*80)

if success_count >= len(test_stocks) * 0.8:  # 80%以上成功率
    print("✓ 腾讯财经API完全可用!")
    print("\n建议行动:")
    print("  1. 创建腾讯财经数据适配器")
    print("  2. 集成到系统作为主数据源")
    print("  3. 实现多数据源切换机制")
elif success_count > 0:
    print(f"⚠ 腾讯财经API部分可用，成功率 {success_count/len(test_stocks)*100:.1f}%")
    print("\n建议行动:")
    print("  1. 使用腾讯财经作为主数据源")
    print("  2. 模拟数据作为备用")
    print("  3. 检查失败原因")
else:
    print("✗ 腾讯财经API当前不可用")
    print("\n建议行动:")
    print("  1. 继续使用模拟数据")
    print("  2. 等待网络恢复")

print("\n" + "="*80)
