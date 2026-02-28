#!/usr/bin/env python
"""
快速开始脚本 - 引导用户使用系统
"""
import os
import sys

def print_banner():
    """打印横幅"""
    banner = """
╔════════════════════════════════════════════════════════════╗
║         股票交易AI系统 - Quick Start Guide                  ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_options():
    """打印选项"""
    print("\n请选择操作：")
    print("1. 快速测试（使用模拟数据，推荐首次运行）")
    print("2. 下载股票数据（使用真实数据）")
    print("3. 训练模型（需要先下载数据）")
    print("4. 运行完整流程（下载+训练+回测+优化）")
    print("5. 查看使用说明")
    print("0. 退出")

def run_test():
    """运行测试"""
    print("\n正在运行快速测试...")
    print("这将使用模拟数据验证系统功能...")
    os.system("python test_system.py")

def download_data():
    """下载数据"""
    print("\n开始下载股票数据...")
    print("提示：首次运行建议使用 --stocks 10 参数")
    print("      完整运行可以指定 --stocks 50")
    stocks = input("\n请输入要下载的股票数量（默认20）：").strip()
    if not stocks:
        stocks = "20"
    os.system(f"python main.py --download-only --stocks {stocks}")

def train_models():
    """训练模型"""
    print("\n开始训练模型...")
    print("提示：确保已经下载了数据")
    print("      如果数据不存在，会先自动下载")
    stocks = input("\n请输入股票数量（默认20）：").strip()
    if not stocks:
        stocks = "20"
    os.system(f"python main.py --train-only --stocks {stocks}")

def run_full_pipeline():
    """运行完整流程"""
    print("\n开始运行完整流程...")
    print("这将执行：数据下载 → 模型训练 → 回测 → 优化")
    print("预计需要 20-40 分钟（取决于股票数量）")
    confirm = input("\n确认继续？(y/n): ").strip().lower()
    if confirm == 'y':
        stocks = input("\n请输入股票数量（默认20）：").strip()
        if not stocks:
            stocks = "20"
        os.system(f"python main.py --stocks {stocks}")
    else:
        print("已取消")

def show_usage():
    """显示使用说明"""
    print("\n使用说明：")
    print("=" * 60)
    with open('USAGE_GUIDE.md', 'r', encoding='utf-8') as f:
        content = f.read()
        # 只显示前50行
        lines = content.split('\n')[:50]
        print('\n'.join(lines))
        print("\n...（更多内容请查看 USAGE_GUIDE.md 文件）...")

def main():
    """主函数"""
    print_banner()
    
    while True:
        print_options()
        choice = input("\n请输入选项（0-5）：").strip()
        
        if choice == '1':
            run_test()
        elif choice == '2':
            download_data()
        elif choice == '3':
            train_models()
        elif choice == '4':
            run_full_pipeline()
        elif choice == '5':
            show_usage()
        elif choice == '0':
            print("\n感谢使用！")
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
