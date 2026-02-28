"""
网络诊断脚本 - 帮助诊断网络连接问题
"""
import requests
import time
from datetime import datetime

def test_connection(url, timeout=10):
    """测试连接"""
    try:
        print(f"\n测试连接: {url}")
        print(f"超时设置: {timeout}秒")
        
        start_time = time.time()
        response = requests.get(url, timeout=timeout)
        elapsed_time = time.time() - start_time
        
        print(f"✓ 连接成功")
        print(f"  状态码: {response.status_code}")
        print(f"  响应时间: {elapsed_time:.2f}秒")
        print(f"  响应大小: {len(response.content)} 字节")
        
        return True
    except requests.exceptions.Timeout:
        print(f"✗ 连接超时 ({timeout}秒)")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"✗ 连接错误: {e}")
        return False
    except Exception as e:
        print(f"✗ 其他错误: {e}")
        return False

def test_eastmoney_apis():
    """测试东方财富API"""
    print("\n" + "="*60)
    print("测试东方财富API连接")
    print("="*60)
    
    # 测试股票列表API
    stock_list_url = "http://82.push2.eastmoney.com/api/qt/clist/get"
    params = {
        'pn': 1,
        'pz': 10,
        'fields': 'f12,f14,f2,f3,f5,f6'
    }
    
    try:
        print("\n测试股票列表API...")
        print(f"URL: {stock_list_url}")
        response = requests.get(stock_list_url, params=params, timeout=30)
        data = response.json()
        
        print(f"✓ API响应成功")
        print(f"  状态码: {response.status_code}")
        
        if data.get('data'):
            stocks = data['data'].get('diff', [])
            print(f"  获取股票数量: {len(stocks)}")
            if stocks:
                print(f"  示例股票: {stocks[0].get('f12')} - {stocks[0].get('f14')}")
        
        return True
    except Exception as e:
        print(f"✗ API测试失败: {e}")
        return False

def main():
    print("\n股票交易AI系统 - 网络诊断工具")
    print("="*60)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 测试基本网络连接
    print("\n" + "="*60)
    print("测试基本网络连接")
    print("="*60)
    
    test_urls = [
        ("百度", "https://www.baidu.com"),
        ("GitHub", "https://www.github.com"),
        ("东方财富", "https://quote.eastmoney.com"),
    ]
    
    results = {}
    for name, url in test_urls:
        results[name] = test_connection(url, timeout=10)
        time.sleep(1)
    
    # 测试东方财富API
    api_success = test_eastmoney_apis()
    
    # 总结
    print("\n" + "="*60)
    print("诊断总结")
    print("="*60)
    
    all_success = all(results.values())
    
    if all_success:
        print("\n✓ 基本网络连接正常")
    else:
        print("\n✗ 部分网络连接失败:")
        for name, success in results.items():
            status = "✓" if success else "✗"
            print(f"  {status} {name}")
    
    if api_success:
        print("✓ 东方财富API连接正常")
    else:
        print("✗ 东方财富API连接失败")
        print("\n可能的原因:")
        print("  1. 网络防火墙或代理阻止了连接")
        print("  2. 东方财富服务器暂时不可用")
        print("  3. IP地址被临时限制（频繁请求）")
        print("\n建议解决方案:")
        print("  1. 稍后重试")
        print("  2. 使用模拟数据测试系统: python test_system.py")
        print("  3. 检查网络连接和代理设置")
        print("  4. 联系网络管理员")
    
    print("\n" + "="*60)
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

if __name__ == "__main__":
    main()
