"""
股票交易AI系统 - Flask Web应用
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
from src.model import StockPredictionModel
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
    global data_manager, predictor

    try:
        logger.info("初始化数据源管理器...")
        data_manager = DataSourceManager(preferred_source=DataSource.TENCENT)
        logger.info("✓ 数据源管理器初始化成功")

        logger.info("初始化模型预测器...")
        predictor = StockPredictionModel('stock_model')
        logger.info("✓ 模型预测器初始化成功")

        logger.info("所有组件初始化完成")
        return True
    except Exception as e:
        logger.error(f"组件初始化失败: {e}")
        return False

# 在第一次请求时初始化组件
@app.before_request
def initialize_if_needed():
    """在第一次请求前初始化组件"""
    global data_manager
    if data_manager is None:
        init_components()

# ==================== 页面路由 ====================

@app.route('/')
def index():
    """首页"""
    return render_template('index.html')

@app.route('/stock')
def stock_page():
    """股票信息页面"""
    return render_template('stock.html')

@app.route('/model')
def model_page():
    """模型测试页面"""
    return render_template('model.html')

@app.route('/backtest')
def backtest_page():
    """回测页面"""
    return render_template('backtest.html')

# ==================== API路由 ====================

@app.route('/api/health')
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'data_manager': data_manager is not None,
            'predictor': predictor is not None
        }
    })

@app.route('/api/stocks')
def get_stocks():
    """获取股票列表（不限制数量）"""
    try:
        # 不传limit参数，获取所有可用股票
        stock_list = data_manager.get_stock_list()

        if stock_list.empty:
            return jsonify({
                'success': False,
                'message': '获取股票列表失败'
            }), 500

        return jsonify({
            'success': True,
            'data': stock_list.to_dict('records'),
            'total': len(stock_list)
        })
    except Exception as e:
        logger.error(f"获取股票列表失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/stock/<stock_code>')
def get_stock_info(stock_code):
    """获取股票信息"""
    try:
        days = request.args.get('days', 100, type=int)

        # 获取股票数据
        df = data_manager.get_stock_kline(stock_code, days=days)

        if df.empty:
            return jsonify({
                'success': False,
                'message': f'获取股票 {stock_code} 数据失败'
            }), 500

        # 计算基本信息
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        change = (latest['close'] - prev['close']) / prev['close'] * 100 if prev['close'] != 0 else 0

        # 计算统计信息
        stats = {
            'current': round(latest['close'], 2),
            'change': round(change, 2),
            'high': round(df['high'].max(), 2),
            'low': round(df['low'].min(), 2),
            'volume': int(df['volume'].sum()),
            'avg_volume': int(df['volume'].mean()),
            'days': len(df)
        }

        # 准备K线数据
        kline_data = []
        for date, row in df.iterrows():
            kline_data.append({
                'date': date.strftime('%Y-%m-%d'),
                'open': round(row['open'], 2),
                'high': round(row['high'], 2),
                'low': round(row['low'], 2),
                'close': round(row['close'], 2),
                'volume': int(row['volume'])
            })

        return jsonify({
            'success': True,
            'data': {
                'code': stock_code,
                'stats': stats,
                'kline': kline_data
            }
        })
    except Exception as e:
        logger.error(f"获取股票信息失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/predict/<stock_code>', methods=['POST'])
def predict_stock(stock_code):
    """预测股票价格"""
    try:
        days = request.json.get('days', 30) if request.json else 30

        # 获取数据
        df = data_manager.get_stock_kline(stock_code, days=days * 2)

        if df.empty or len(df) < days:
            return jsonify({
                'success': False,
                'message': f'数据不足，至少需要 {days} 天数据'
            }), 400

        # 简单的数据预处理
        df_processed = df[['close']].copy()

        # 计算技术指标
        df_processed['ma5'] = df_processed['close'].rolling(5).mean()
        df_processed['ma10'] = df_processed['close'].rolling(10).mean()
        df_processed['ma20'] = df_processed['close'].rolling(20).mean()

        df_processed = df_processed.dropna()

        # 准备训练数据
        sequence_length = 10
        X, y = [], []

        for i in range(len(df_processed) - sequence_length):
            X.append(df_processed['close'].iloc[i:i+sequence_length].values)
            y.append(df_processed['close'].iloc[i+sequence_length])

        X = np.array(X)
        y = np.array(y)

        if len(X) < 20:
            return jsonify({
                'success': False,
                'message': '数据不足，无法训练模型'
            }), 400

        # 简化训练（只训练10个epoch）
        predictor.build_lstm_model(input_shape=(sequence_length, 1))
        predictor.train(X, y, epochs=10, batch_size=16)

        # 预测
        split_idx = int(len(X) * 0.8)
        X_test = X[split_idx:]
        y_test = y[split_idx:]

        predictions = predictor.predict(X_test)

        # 准备返回数据
        actual_dates = df_processed.index[split_idx + sequence_length:].strftime('%Y-%m-%d').tolist()

        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code,
                'train_days': len(X) - len(X_test),
                'test_days': len(X_test),
                'dates': actual_dates[:len(predictions)],
                'actual': [round(float(p), 2) for p in y_test[:len(predictions)]],
                'predicted': [round(float(p), 2) for p in predictions.flatten()],
                'accuracy': round(np.mean(np.abs(y_test[:len(predictions)] - predictions.flatten()) / y_test[:len(predictions)]), 4)
            }
        })
    except Exception as e:
        logger.error(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/backtest/<stock_code>', methods=['POST'])
def backtest_stock(stock_code):
    """回测股票策略"""
    try:
        days = request.json.get('days', 100) if request.json else 100
        initial_capital = request.json.get('initial_capital', 100000) if request.json else 100000

        # 获取数据
        df = data_manager.get_stock_kline(stock_code, days=days)

        if df.empty:
            return jsonify({
                'success': False,
                'message': '获取数据失败'
            }), 500

        # 简单的回测策略：移动平均交叉
        df['ma5'] = df['close'].rolling(5).mean()
        df['ma20'] = df['close'].rolling(20).mean()
        df = df.dropna()

        # 生成信号
        df['signal'] = 0
        df.loc[df['ma5'] > df['ma20'], 'signal'] = 1  # 买入
        df.loc[df['ma5'] < df['ma20'], 'signal'] = -1  # 卖出

        # 模拟交易
        capital = initial_capital
        position = 0
        trades = []
        portfolio_values = []
        benchmark_values = []

        for i in range(len(df)):
            current_price = df['close'].iloc[i]
            signal = df['signal'].iloc[i]
            prev_signal = df['signal'].iloc[i-1] if i > 0 else 0

            # 买入
            if signal == 1 and prev_signal != 1 and position == 0:
                shares = int(capital / current_price)
                position = shares
                capital -= shares * current_price
                trades.append({
                    'date': df.index[i].strftime('%Y-%m-%d'),
                    'action': 'buy',
                    'price': round(current_price, 2),
                    'shares': shares,
                    'amount': round(shares * current_price, 2)
                })

            # 卖出
            elif signal == -1 and prev_signal != -1 and position > 0:
                capital += position * current_price
                trades.append({
                    'date': df.index[i].strftime('%Y-%m-%d'),
                    'action': 'sell',
                    'price': round(current_price, 2),
                    'shares': position,
                    'amount': round(position * current_price, 2)
                })
                position = 0

            # 计算组合价值
            portfolio_value = capital + position * current_price
            portfolio_values.append(portfolio_value)

            # 基准收益（买入持有）
            benchmark_value = initial_capital * (current_price / df['close'].iloc[0])
            benchmark_values.append(benchmark_value)

        # 最终平仓
        if position > 0:
            capital += position * df['close'].iloc[-1]
            trades.append({
                'date': df.index[-1].strftime('%Y-%m-%d'),
                'action': 'sell',
                'price': round(df['close'].iloc[-1], 2),
                'shares': position,
                'amount': round(position * df['close'].iloc[-1], 2)
            })

        # 计算指标
        final_capital = capital
        total_return = (final_capital - initial_capital) / initial_capital * 100
        benchmark_return = (benchmark_values[-1] - initial_capital) / initial_capital * 100

        winning_trades = len([t for t in trades if t['action'] == 'sell' and t['price'] > trades[trades.index(t)-1]['price']]) if len(trades) > 1 else 0
        total_trades_count = len([t for t in trades if t['action'] == 'sell'])
        win_rate = (winning_trades / total_trades_count * 100) if total_trades_count > 0 else 0

        max_value = max(portfolio_values)
        min_after_max = min(portfolio_values[portfolio_values.index(max_value):]) if max_value < portfolio_values[-1] else portfolio_values[-1]
        max_drawdown = (min_after_max - max_value) / max_value * 100 if max_value > 0 else 0

        return jsonify({
            'success': True,
            'data': {
                'initial_capital': initial_capital,
                'final_capital': round(final_capital, 2),
                'total_return': round(total_return, 2),
                'win_rate': round(win_rate, 2),
                'max_drawdown': round(max_drawdown, 2),
                'total_trades': len(trades),
                'trades': trades,
                'dates': df.index.strftime('%Y-%m-%d').tolist(),
                'strategy_returns': [round(v, 2) for v in portfolio_values],
                'benchmark_returns': [round(v, 2) for v in benchmark_values]
            }
        })
    except Exception as e:
        logger.error(f"回测失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# ==================== 错误处理 ====================

@app.errorhandler(404)
def not_found(error):
    """404错误"""
    return jsonify({
        'success': False,
        'message': '页面未找到'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    """500错误"""
    return jsonify({
        'success': False,
        'message': '服务器内部错误'
    }), 500

# ==================== 启动应用 ====================

if __name__ == '__main__':
    # 初始化组件
    if init_components():
        print("="*60)
        print("股票交易AI系统 - Web服务")
        print("="*60)
        print("✓ 所有组件初始化成功")
        print("✓ 启动Web服务...")
        print("\n访问地址:")
        print("  首页: http://localhost:5000/")
        print("  股票: http://localhost:5000/stock")
        print("  模型: http://localhost:5000/model")
        print("  回测: http://localhost:5000/backtest")
        print("\n按 Ctrl+C 停止服务")
        print("="*60)

        # 启动Flask应用
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("✗ 组件初始化失败，无法启动服务")
        sys.exit(1)
