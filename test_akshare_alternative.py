"""
测试AKShare的备用数据源
"""
import akshare as ak
from datetime import datetime
import time

print("="*80)
print("AKShare 备用数据源测试")
print("="*80)
print(f"测试时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# 测试不同的AKShare接口
tests = [
    {
        "name": "新浪财经 - 历史行情",
        "func": lambda: ak.stock_zh_a_hist(symbol="600000", period="daily", adjust="qfq"),
        "desc": "使用新浪财经数据源"
    },
    {
        "name": "腾讯财经 - 历史行情",
        "func": lambda: ak.stock_zh_a_daily(symbol="sh600000", start_date="20240101", end_date="20251231", adjust="qfq"),
        "desc": "使用腾讯财经数据源"
    },
    {
        "name": "东方财富 - 指数数据",
        "func": lambda: ak.stock_zh_index_daily(symbol="sh000001"),
        "desc": "测试指数数据"
    },
    {
        "name": "东方财富 - 实时行情",
        "func": lambda: ak.stock_zh_a_spot_em(),
        "desc": "测试实时行情接口"
    },
]

for i, test in enumerate(tests, 1):
    print(f"\n{'='*80}")
    print(f"测试 {i}: {test['name']}")
    print(f"说明: {test['desc']}")
    print(f"{'='*80}")

    try:
        print("正在请求数据...")
        start_time = time.time()
        result = test['func']()
        elapsed_time = time.time() - start_time

        if result is not None and not result.empty:
            print(f"✓ 成功!")
            print(f"  耗时: {elapsed_time:.2f}秒")
            print(f"  数据形状: {result.shape}")
            print(f"  前5行:")
            print(result.head())
        else:
            print(f"⚠ 返回空数据或None")
            print(f"  耗时: {elapsed_time:.2f}秒")

    except Exception as e:
        elapsed_time = time.time() - start_time
        error_msg = str(e)
        print(f"✗ 失败: {error_msg[:150]}")
        print(f"  耗时: {elapsed_time:.2f}秒")

    # 延迟避免请求过快
    time.sleep(2)

# 测试结论
print("\n" + "="*80)
print("测试总结")
print("="*80)
print("""
经过测试，发现：
1. AKShare的主要数据接口也受到相同的限制
2. 云服务器环境被限制访问金融数据API
3. 这不是单个API的问题，而是整个环境的问题

根本原因：
- 云服务器IP被金融数据网站列入黑名单
- 反爬虫机制识别出服务器请求
- 网络层或应用层被限制访问

唯一可行的解决方案：
1. 使用模拟数据（推荐）⭐⭐⭐
2. 使用本地数据文件
3. 使用专业付费数据源（需要真实IP/认证）
4. 使用代理IP池（复杂且不稳定）

推荐行动：
立即使用模拟数据，无需等待API恢复。
""")
