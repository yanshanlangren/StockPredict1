"""
股票交易AI系统 - Flask Web应用 (测试版)
"""
import sys
import os
from flask import Flask, render_template, jsonify, request, send_from_directory
import pandas as pd
import json
from datetime import datetime, timedelta
import numpy as np

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_source_manager import DataSourceManager, DataSource
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__,
            template_folder='app/templates',
            static_folder='app/static')

# 初始化组件
data_manager = None
predictor = None

def init_components():
    """初始化系统组件"""
    global data_manager

    try:
        logger.info("初始化数据源管理器...")
        data_manager = DataSourceManager(preferred_source=DataSource.TENCENT)
        logger.info("✓ 数据源管理器初始化成功")

        logger.info("所有组件初始化完成")
        return True
    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        return False

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/stock')
def stock():
    """股票查询页面"""
    return render_template('stock.html')

@app.route('/model')
def model():
    """模型预测页面"""
    return render_template('model.html')

@app.route('/backtest')
def backtest():
    """回测页面"""
    return render_template('backtest.html')

@app.route('/api/health')
def health():
    """健康检查"""
    status = {
        'status': 'ok',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'version': '2.0 (test mode - no TensorFlow)',
        'message': 'Flask Web应用运行正常'
    }
    return jsonify(status)

@app.route('/api/stocks')
def get_stocks():
    """获取股票列表"""
    try:
        # 返回常用股票列表
        stock_list = [
            {'code': '000001', 'name': '平安银行'},
            {'code': '000002', 'name': '万科A'},
            {'code': '000725', 'name': '京东方A'},
            {'code': '000858', 'name': '五粮液'},
            {'code': '600000', 'name': '浦发银行'},
            {'code': '600519', 'name': '贵州茅台'},
            {'code': '601318', 'name': '中国平安'},
            {'code': '600036', 'name': '招商银行'}
        ]
        return jsonify({'success': True, 'data': stock_list})
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/stock/<stock_code>')
def get_stock_data(stock_code):
    """获取股票K线数据"""
    try:
        # 获取查询参数
        days = request.args.get('days', 30, type=int)

        if not data_manager:
            return jsonify({'success': False, 'error': '数据源管理器未初始化'})

        # 获取股票数据
        data = data_manager.get_stock_kline(stock_code, days=days)

        if data is None or len(data) == 0:
            return jsonify({'success': False, 'error': f'无法获取股票 {stock_code} 的数据'})

        # 计算基本统计信息
        stats = {
            'highest': float(data['close'].max()),
            'lowest': float(data['close'].min()),
            'average': float(data['close'].mean()),
            'latest': float(data['close'].iloc[-1]),
            'change': float((data['close'].iloc[-1] - data['close'].iloc[-2]) / data['close'].iloc[-2] * 100)
        }

        # 转换数据格式
        chart_data = []
        for i, row in data.iterrows():
            chart_data.append({
                'date': row['date'],
                'open': float(row['open']),
                'high': float(row['high']),
                'low': float(row['low']),
                'close': float(row['close']),
                'volume': float(row['volume'])
            })

        return jsonify({
            'success': True,
            'data': chart_data,
            'stats': stats
        })

    except Exception as e:
        logger.error(f"获取股票数据失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/predict/<stock_code>', methods=['POST'])
def predict_stock(stock_code):
    """预测股票价格（测试版）"""
    try:
        # 获取请求参数
        params = request.get_json()
        days = params.get('days', 60)

        if not data_manager:
            return jsonify({'success': False, 'error': '数据源管理器未初始化'})

        # 获取股票数据
        data = data_manager.get_stock_kline(stock_code, days=days)

        if data is None or len(data) < 50:
            return jsonify({
                'success': False,
                'error': '数据不足',
                'message': f'当前只有 {len(data) if data is not None else 0} 天数据，至少需要50天',
                'suggestions': [
                    '增加训练天数（建议使用 60-180 天）',
                    '选择其他股票代码',
                    '等待数据源更新'
                ]
            })

        # 模拟预测结果（测试版）
        latest_price = float(data['close'].iloc[-1])
        predictions = []

        for i in range(5):
            # 模拟预测价格（随机波动）
            change = np.random.uniform(-0.02, 0.02)
            predicted_price = latest_price * (1 + change * (i + 1))
            predictions.append(predicted_price)

        # 模拟准确率
        accuracy = round(np.random.uniform(0.65, 0.85) * 100, 2)

        # 生成实际值和预测值图表数据
        chart_data = []
        for i, row in data.tail(30).iterrows():
            chart_data.append({
                'date': row['date'],
                'actual': float(row['close']),
                'predicted': None
            })

        # 添加预测值
        last_date = datetime.strptime(data['date'].iloc[-1], '%Y-%m-%d')
        for i, pred in enumerate(predictions):
            pred_date = last_date + timedelta(days=i+1)
            chart_data.append({
                'date': pred_date.strftime('%Y-%m-%d'),
                'actual': None,
                'predicted': pred
            })

        return jsonify({
            'success': True,
            'predictions': predictions,
            'accuracy': accuracy,
            'chart_data': chart_data,
            'latest_price': latest_price,
            'days_used': len(data)
        })

    except Exception as e:
        logger.error(f"预测失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/backtest/<stock_code>', methods=['POST'])
def run_backtest(stock_code):
    """运行回测（测试版）"""
    try:
        # 获取请求参数
        params = request.get_json()
        days = params.get('days', 30)
        initial_capital = params.get('initial_capital', 100000)

        if not data_manager:
            return jsonify({'success': False, 'error': '数据源管理器未初始化'})

        # 获取股票数据
        data = data_manager.get_stock_kline(stock_code, days=days)

        if data is None or len(data) < 10:
            return jsonify({'success': False, 'error': '数据不足，无法进行回测'})

        # 模拟回测结果
        capital = initial_capital
        trades = []
        max_drawdown = 0
        max_capital = capital

        for i in range(1, len(data)):
            # 简单策略：涨则买，跌则卖
            today_close = float(data['close'].iloc[i])
            yesterday_close = float(data['close'].iloc[i-1])
            change = (today_close - yesterday_close) / yesterday_close

            if change > 0.02:  # 上涨超过2%，买入
                shares = int(capital * 0.1 / today_close)  # 使用10%资金
                if shares > 0:
                    capital -= shares * today_close
                    trades.append({
                        'date': data['date'].iloc[i],
                        'type': 'buy',
                        'price': today_close,
                        'shares': shares,
                        'capital': capital
                    })
            elif change < -0.02:  # 下跌超过2%，卖出
                if capital < initial_capital:  # 有持仓
                    sell_shares = int((initial_capital - capital) / today_close)
                    capital += sell_shares * today_close
                    trades.append({
                        'date': data['date'].iloc[i],
                        'type': 'sell',
                        'price': today_close,
                        'shares': sell_shares,
                        'capital': capital
                    })

            # 计算最大回撤
            if capital > max_capital:
                max_capital = capital
            drawdown = (max_capital - capital) / max_capital
            if drawdown > max_drawdown:
                max_drawdown = drawdown

        # 最终收益
        final_return = (capital - initial_capital) / initial_capital * 100
        win_rate = len([t for t in trades if t['type'] == 'sell' and t['capital'] > t['shares'] * t['price']]) / len(trades) * 100 if trades else 0

        # 生成收益曲线
        equity_curve = []
        for i in range(len(data)):
            if i < len(trades):
                equity_curve.append({
                    'date': data['date'].iloc[i],
                    'capital': trades[i]['capital']
                })
            else:
                equity_curve.append({
                    'date': data['date'].iloc[i],
                    'capital': capital
                })

        return jsonify({
            'success': True,
            'trades': trades,
            'final_capital': capital,
            'final_return': final_return,
            'max_drawdown': max_drawdown * 100,
            'win_rate': win_rate,
            'equity_curve': equity_curve
        })

    except Exception as e:
        logger.error(f"回测失败: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

def main():
    """主函数"""
    print("=" * 60)
    print("股票交易AI系统 - Flask Web应用")
    print("版本: 2.0 (测试版 - 无TensorFlow)")
    print("=" * 60)

    # 初始化组件
    if not init_components():
        print("⚠️  组件初始化失败，部分功能可能不可用")
        print("建议: 请安装完整依赖以获得完整功能")
        print("  pip install -r requirements.txt")

    print("\n🚀 启动Flask服务...")
    print("📝 访问地址: http://localhost:5000/")
    print("📡 API地址: http://localhost:5000/api/health")
    print("=" * 60)

    # 启动应用
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    main()
