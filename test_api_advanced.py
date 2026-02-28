"""
高级API测试 - 尝试不同的请求方式和参数
"""
import requests
import json
import time

print("="*80)
print("东方财富API高级测试")
print("="*80)

# 测试1: 使用不同的User-Agent
print("\n测试1: 使用不同的User-Agent")

user_agents = [
    # 标准浏览器
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    # 移动端
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1",
    # 简单User-Agent
    "Mozilla/5.0",
    # 无User-Agent
    "",
]

# 测试URL
url = "http://push2.eastmoney.com/api/qt/stock/kline/get"
params = {
    'fields1': 'f1,f2,f3,f4,f5,f6',
    'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
    'klt': '101',  # 日K线
    'fqt': '1',    # 前复权
    'secid': '1.600000',  # 600000 (浦发银行)
    'beg': '0',
    'end': '20500101'
}

for i, ua in enumerate(user_agents, 1):
    print(f"\n尝试 User-Agent {i}...", end="")
    if not ua:
        print(" (无User-Agent)", end="")
    else:
        print(f" ({ua[:50]}...)", end="")

    headers = {}
    if ua:
        headers['User-Agent'] = ua

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and data['data'].get('klines'):
                print(f" → ✓ 成功! 返回 {len(data['data']['klines'])} 天数据")
            else:
                print(f" → 响应成功但无数据")
        else:
            print(f" → 状态码 {response.status_code}")
    except requests.exceptions.ConnectionError as e:
        print(f" → ✗ 连接错误")
    except Exception as e:
        print(f" → ✗ {type(e).__name__}")

# 测试2: 测试不同的API端点
print("\n" + "="*80)
print("测试2: 不同的API端点")

endpoints = [
    ("东方财富 K线API", "http://push2.eastmoney.com/api/qt/stock/kline/get"),
    ("东方财富 K线API (备用)", "http://82.push2.eastmoney.com/api/qt/stock/kline/get"),
    ("东方财富 K线API (HTTPS)", "https://push2.eastmoney.com/api/qt/stock/kline/get"),
]

for name, endpoint_url in endpoints:
    print(f"\n测试: {name}")
    print(f"URL: {endpoint_url}")

    try:
        response = requests.get(endpoint_url, params=params, timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and data['data'].get('klines'):
                print(f"  ✓ 成功! 返回 {len(data['data']['klines'])} 天数据")
            else:
                print(f"  ✓ 响应成功但无数据")
        else:
            print(f"  ✗ 状态码 {response.status_code}")
    except Exception as e:
        print(f"  ✗ 失败: {type(e).__name__}")

# 测试3: 添加常见的请求头
print("\n" + "="*80)
print("测试3: 添加常见请求头")

headers_variations = [
    {
        "name": "基本浏览器头",
        "headers": {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': '*/*',
            'Accept-Language': 'zh-CN,zh;q=0.9',
        }
    },
    {
        "name": "完整浏览器头",
        "headers": {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate',
            'Referer': 'https://quote.eastmoney.com/sh600000.html',
            'Connection': 'keep-alive',
        }
    },
]

for variation in headers_variations:
    print(f"\n测试: {variation['name']}")
    try:
        response = requests.get(url, params=params, headers=variation['headers'], timeout=10)
        if response.status_code == 200:
            data = response.json()
            if data.get('data') and data['data'].get('klines'):
                print(f"  ✓ 成功! 返回 {len(data['data']['klines'])} 天数据")
            else:
                print(f"  ✓ 响应成功但无数据")
        else:
            print(f"  ✗ 状态码 {response.status_code}")
    except Exception as e:
        print(f"  ✗ 失败: {type(e).__name__}")

# 测试4: 检查是否是IP限制
print("\n" + "="*80)
print("测试4: 检查是否是IP或地域限制")

# 测试不同的股票
test_codes = ['0.000001', '1.600000', '0.000002', '1.600001']  # 深圳和上海各两只

print("\n测试不同股票代码:")
for code in test_codes:
    params['secid'] = code
    print(f"\n股票代码: {code}", end="")
    try:
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            print(" → 响应成功", end="")
            data = response.json()
            if data.get('data') and data['data'].get('klines'):
                print(f" ✓ 数据返回")
            else:
                print(f" (无数据)")
        else:
            print(f" → 状态码 {response.status_code}")
    except Exception as e:
        print(f" → {type(e).__name__}")

# 测试5: 原始连接测试（无库包装）
print("\n" + "="*80)
print("测试5: 底层连接诊断")

import socket

api_host = "push2.eastmoney.com"
api_port = 80

print(f"\n尝试TCP连接到 {api_host}:{api_port}")
try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    result = sock.connect_ex((api_host, api_port))
    sock.close()

    if result == 0:
        print("  ✓ TCP连接成功")
    else:
        print(f"  ✗ TCP连接失败，错误码: {result}")
except Exception as e:
    print(f"  ✗ 连接异常: {e}")

# 测试6: 检查DNS解析
print("\n测试6: DNS解析测试")
import socket

try:
    ip = socket.gethostbyname(api_host)
    print(f"  ✓ DNS解析成功: {api_host} → {ip}")
except Exception as e:
    print(f"  ✗ DNS解析失败: {e}")

# 结论
print("\n" + "="*80)
print("测试总结")
print("="*80)
print("""
根据以上测试，如果：
1. 所有 User-Agent 都失败 → 不是 User-Agent 问题
2. 所有端点都失败 → 不是端点问题
3. 所有股票代码都失败 → 不是参数问题
4. TCP连接失败 → 网络连接被阻止
5. 所有测试都出现 RemoteDisconnected → 服务器主动拒绝连接

可能的根本原因：
- 服务器检测到请求来自云服务器/代理IP
- 服务器对频繁请求实施了限制
- 服务器实施了地域限制
- 需要特定的认证或Token
- 服务器暂时不可用/维护中

建议方案：
1. 使用模拟数据进行开发和测试
2. 寻找其他数据源（如Tushare、AKShare等）
3. 使用代理IP池
4. 等待服务器恢复
""")
