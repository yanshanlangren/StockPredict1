"""
快速修复脚本 - 网络问题快速处理
"""
import os
import sys

def print_menu():
    print("\n" + "="*60)
    print("网络问题快速修复")
    print("="*60)
    print("\n请选择操作：")
    print("1. 使用模拟数据测试系统（推荐）")
    print("2. 诊断网络连接")
    print("3. 减少下载数量测试（5只股票）")
    print("4. 查看网络问题FAQ")
    print("5. 检查系统日志")
    print("0. 退出")

def run_mock_test():
    print("\n正在运行模拟数据测试...")
    print("这将使用模拟数据验证系统功能，无需网络连接")
    os.system("python test_system.py")

def diagnose_network():
    print("\n正在诊断网络连接...")
    os.system("python diagnose_network.py")

def test_with_fewer_stocks():
    print("\n尝试下载少量股票数据（5只）...")
    print("如果仍然失败，建议使用模拟数据")
    os.system("python main.py --download-only --stocks 5")

def show_faq():
    print("\n打开网络问题FAQ...")
    if os.path.exists('NETWORK_FAQ.md'):
        with open('NETWORK_FAQ.md', 'r', encoding='utf-8') as f:
            content = f.read()
            # 显示前50行
            lines = content.split('\n')[:50]
            print('\n'.join(lines))
            print("\n...（更多内容请查看 NETWORK_FAQ.md 文件）...")
    else:
        print("FAQ文件不存在")

def check_logs():
    print("\n检查系统日志...")
    log_files = [
        'logs/crawler.log',
        'logs/model.log',
        'logs/backtest.log',
        'logs/main.log'
    ]

    found_logs = False
    for log_file in log_files:
        if os.path.exists(log_file):
            print(f"\n{log_file} (最后20行):")
            print("-" * 40)
            os.system(f"tail -n 20 {log_file}")
            found_logs = True

    if not found_logs:
        print("没有找到日志文件")

def main():
    print("\n股票交易AI系统 - 网络问题快速修复")
    print("="*60)
    print("\n检测到网络连接问题")
    print("本工具将帮助你快速解决问题\n")

    while True:
        print_menu()
        choice = input("\n请输入选项（0-5）：").strip()

        if choice == '1':
            run_mock_test()
        elif choice == '2':
            diagnose_network()
        elif choice == '3':
            test_with_fewer_stocks()
        elif choice == '4':
            show_faq()
        elif choice == '5':
            check_logs()
        elif choice == '0':
            print("\n退出工具")
            print("\n推荐操作：")
            print("  运行: python test_system.py")
            print("  使用模拟数据测试系统功能")
            break
        else:
            print("\n无效选项，请重新选择")

        input("\n按Enter继续...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n程序已中断")
        sys.exit(0)
