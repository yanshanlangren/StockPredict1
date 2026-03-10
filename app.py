#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票交易AI系统 - Flask Web应用（全局模型版）

功能：
1. 全局模型预测
2. 股票数据查询
3. 模型信息展示
"""
import sys
import os
from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
from datetime import datetime
import logging

# 添加src目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.data_source_manager import DataSourceManager, DataSource

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 尝试导入全局模型
GLOBAL_MODEL_AVAILABLE = False
try:
    from src.global_model import get_global_model
    global_model = get_global_model()
    GLOBAL_MODEL_AVAILABLE = global_model.is_available()
    if GLOBAL_MODEL_AVAILABLE:
        logger.info("✓ 全局模型已加载")
    else:
        logger.info("ℹ️  全局模型未找到")
        logger.info("提示: 训练全局模型请运行: python train_global_model.py")
except Exception as e:
    logger.warning(f"全局模型加载失败: {e}")

# 创建Flask应用
app = Flask(__name__,
            template_folder='app/templates',
            static_folder='app/static')

# 初始化组件
data_manager = None

def init_components():
    """初始化系统组件"""
    global data_manager

    try:
        logger.info("初始化数据源管理器...")
        data_manager = DataSourceManager(preferred_source=DataSource.TENCENT)
        logger.info("✓ 数据源管理器初始化成功")
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

@app.route('/model')
def model_page():
    """模型测试页面"""
    return render_template('model.html')

@app.route('/stock')
def stock_page():
    """股票信息页面"""
    return render_template('stock.html')

# ==================== API路由 ====================

@app.route('/api/health')
def health_check():
    """健康检查"""
    # 获取全局模型信息
    global_model_info = {}
    if GLOBAL_MODEL_AVAILABLE:
        try:
            global_model_info = global_model.get_model_info()
        except:
            pass
    
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'data_manager': data_manager is not None,
            'global_model': GLOBAL_MODEL_AVAILABLE
        },
        'global_model': global_model_info if global_model_info else None
    })

@app.route('/api/model/info')
def get_model_info():
    """获取模型信息"""
    result = {
        'global_model_available': GLOBAL_MODEL_AVAILABLE
    }
    
    if GLOBAL_MODEL_AVAILABLE:
        try:
            model_info = global_model.get_model_info()
            result['global_model'] = model_info
        except Exception as e:
            result['global_model_error'] = str(e)
    
    return jsonify(result)

@app.route('/api/stocks')
def get_stocks():
    """获取股票列表"""
    try:
        # 获取刷新参数
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'

        # 获取股票列表
        if force_refresh:
            logger.info("强制刷新股票列表...")
            stock_list = data_manager.get_stock_list(force_refresh=True)
        else:
            stock_list = data_manager.get_stock_list()

        if stock_list.empty:
            return jsonify({
                'success': False,
                'message': '获取股票列表失败'
            }), 500

        # 确保 code 列为字符串类型
        if 'code' in stock_list.columns:
            stock_list = stock_list.copy()
            stock_list['code'] = stock_list['code'].astype(str)

        return jsonify({
            'success': True,
            'data': stock_list.to_dict('records'),
            'total': len(stock_list),
            'refreshed': force_refresh
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
    """
    使用全局模型预测股票价格趋势
    
    请求参数:
    {
        "days": 100  # 获取的历史数据天数（至少80天）
    }
    
    返回:
    {
        "success": true,
        "mode": "global_model",
        "data": {
            "stock_code": "600690",
            "prediction": 1,  # 1:上涨, 0:下跌
            "prediction_text": "上涨",
            "probability": 0.65,
            "confidence": 0.7,
            "latest_price": 10.5,
            "predicted_price": 10.85,
            "expected_return": 0.033,
            "predict_days": 5,
            "days_used": 120,
            "model_type": "global"
        }
    }
    """
    try:
        days = request.json.get('days', 100) if request.json else 100
        
        # 检查全局模型是否可用
        if not GLOBAL_MODEL_AVAILABLE:
            return jsonify({
                'success': False,
                'message': '全局模型不可用，请先训练模型: python train_global_model.py'
            }), 400
        
        logger.info(f"使用全局模型预测股票 {stock_code}")
        
        # 获取足够的历史数据（至少80天）
        df = data_manager.get_stock_kline(stock_code, days=max(days, 100))
        
        if df.empty or len(df) < 80:
            # 数据不足，尝试重新获取
            logger.info(f"数据不足，尝试重新获取...")
            df = data_manager.get_stock_kline(stock_code, days=days * 4, force_refresh=True)
        
        if df.empty or len(df) < 80:
            return jsonify({
                'success': False,
                'message': f'数据不足，全局模型至少需要 80 天数据（当前：{len(df) if not df.empty else 0}天）'
            }), 400
        
        # 使用全局模型预测
        result = global_model.predict(df)
        
        if result['success']:
            # 获取模型信息
            model_info = result.get('model_info', {})
            training_stats = model_info.get('training_stats', {})
            
            # 构建响应
            return jsonify({
                'success': True,
                'mode': 'global_model',
                'data': {
                    'stock_code': stock_code,
                    'prediction': result['prediction'],
                    'prediction_text': result['prediction_text'],
                    'probability': float(round(result['probability'], 4)),
                    'confidence': float(round(result['confidence'], 4)),
                    'latest_price': float(result['latest_price']),
                    'predicted_price': float(result['predicted_price']),
                    'expected_return': float(result['expected_return']),
                    'predict_days': int(result['predict_days']),
                    'days_used': int(len(df)),
                    'model_type': 'global',
                    'model_created': str(model_info.get('created_at', 'unknown')),
                    'training_samples': int(training_stats.get('total_samples', 0)) if training_stats else 0
                }
            })
        else:
            return jsonify({
                'success': False,
                'message': result.get('message', '预测失败')
            }), 500
            
    except Exception as e:
        logger.error(f"预测失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# ==================== 启动应用 ====================

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("股票交易AI系统（全局模型版）启动中...")
    logger.info("=" * 50)
    
    # 初始化组件
    init_components()
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=True)
