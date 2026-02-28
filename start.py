#!/usr/bin/env python
"""
快速开始脚本 - 增强版，支持网络错误处理
"""
import os
import sys
import traceback

def print_banner():
    """打印横幅"""
    banner = """
╔════════════════════════════════════════════════════════════╗
║         股票交易AI系统 - Enhanced Quick Start Guide        ║
╚════════════════════════════════════════════════════════════╝
"""
    print(banner)

def print_options():
    """打印选项"""
    print("\n请选择操作：")
    print("1. 快速测试（使用模拟数据，推荐）")
    print("2. 下载真实数据（需要网络）")
    print("3. 生成模拟数据（离线使用）")
    print("4. 训练模型（使用已有数据）")
    print("5. 运行完整流程（真实数据）")
    print("6. 诊断网络问题")
    print("7. 查看使用说明")
    print("0. 退出")

def run_test():
    """运行测试"""
    print("\n正在运行快速测试...")
    print("这将使用模拟数据验证系统功能...")
    os.system("python test_system.py")

def download_data():
    """下载数据"""
    print("\n开始下载股票数据...")
    print("提示：首次运行建议使用较少股票数量")
    print("      如果网络失败，建议使用模拟数据")
    
    confirm = input("\n确认继续下载？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    stocks = input("\n请输入股票数量（默认10，推荐）: ").strip()
    if not stocks:
        stocks = "10"
    
    print(f"\n尝试下载 {stocks} 只股票的数据...")
    print("如果遇到网络错误，系统会自动重试（最多3次）")
    
    result = os.system(f"python main.py --download-only --stocks {stocks}")
    
    if result != 0:
        print("\n✗ 下载失败！")
        print("\n建议：")
        print("  1. 使用模拟数据（选项3）")
        print("  2. 运行测试脚本（选项1）")
        print("  3. 诊断网络问题（选项6）")

def generate_mock_data():
    """生成模拟数据"""
    print("\n生成模拟数据...")
    print("这将创建与真实数据格式相同的模拟股票数据")
    
    confirm = input("\n确认生成？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    stocks = input("\n请输入股票数量（默认20）: ").strip()
    if not stocks:
        stocks = "20"
    
    print(f"\n生成 {stocks} 只股票的模拟数据...")
    os.system(f"python generate_mock_data.py")
    
    print("\n模拟数据已保存，现在可以：")
    print("  - 运行训练: python main.py --train-only")
    print("  - 运行测试: python test_system.py")

def train_models():
    """训练模型"""
    print("\n开始训练模型...")
    print("提示：需要先有数据（真实数据或模拟数据）")
    
    # 检查数据是否存在
    data_dir = 'data/raw'
    has_data = os.path.exists(data_dir) and len(os.listdir(data_dir)) > 0
    
    if not has_data:
        print("\n⚠ 警告：没有找到数据文件")
        print("\n请先：")
        print("  1. 下载真实数据（选项2）")
        print("  2. 生成模拟数据（选项3）")
        print("  3. 运行测试脚本（选项1）")
        return
    
    print(f"\n发现数据文件，开始训练...")
    os.system("python main.py --train-only")

def run_full_pipeline():
    """运行完整流程"""
    print("\n开始运行完整流程...")
    print("这将执行：数据下载 → 模型训练 → 回测 → 优化")
    print("⚠ 警告：需要网络连接，可能需要20-40分钟")
    
    confirm = input("\n确认继续？(y/n): ").strip().lower()
    if confirm != 'y':
        print("已取消")
        return
    
    stocks = input("\n请输入股票数量（默认10，推荐）: ").strip()
    if not stocks:
        stocks = "10"
    
    print(f"\n开始处理 {stocks} 只股票...")
    result = os.system(f"python main.py --stocks {stocks}")
    
    if result != 0:
        print("\n✗ 流程执行失败")
        print("\n建议：")
        print("  1. 使用模拟数据测试系统")
        print("  2. 诊断网络问题")
        print("  3. 减少股票数量")

def diagnose_network():
    """诊断网络"""
    print("\n诊断网络连接...")
    os.system("python diagnose_network.py")

def show_usage():
    """显示使用说明"""
    print("\n使用说明：")
    print("="*60)
    with open('USAGE_GUIDE.md', 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')[:50]
        print('\n'.join(lines))
        print("\n...（更多内容请查看 USAGE_GUIDE.md 文件）...")

def main():
    """主函数"""
    print_banner()
    
    print("\n提示：")
    print("  • 如果遇到网络问题，推荐使用模拟数据（选项1或3）")
    print("  • 模拟数据可以完整测试所有系统功能")
    print("  • 真实数据仅用于实际应用场景")
    
    while True:
        print_options()
        choice = input("\n请输入选项（0-7）：").strip()
        
        if choice == '1':
            run_test()
        elif choice == '2':
            download_data()
        elif choice == '3':
            generate_mock_data()
        elif choice == '4':
            train_models()
        elif choice == '5':
            run_full_pipeline()
        elif choice == '6':
            diagnose_network()
        elif choice == '7':
            show_usage()
        elif choice == '0':
            print("\n感谢使用！")
            print("\n推荐操作：")
            print("  python test_system.py  # 使用模拟数据测试")
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
    except Exception as e:
        print(f"\n发生错误: {e}")
        traceback.print_exc()
        sys.exit(1)
