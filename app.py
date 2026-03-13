#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票交易AI系统 - Flask Web应用（全局模型版）

功能：
1. 全局模型训练
2. 全局模型预测
3. 股票数据查询
4. 模型信息展示
"""
import sys
import os
import threading
import subprocess
from flask import Flask, render_template, jsonify, request
import pandas as pd
import numpy as np
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

# 训练状态管理
training_status = {
    'is_training': False,
    'progress': 0,
    'message': '',
    'start_time': None,
    'end_time': None,
    'error': None
}
training_lock = threading.Lock()

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

@app.route('/predict')
def predict_page():
    """批量预测页面"""
    return render_template('predict.html')

@app.route('/multimodal')
def multimodal_page():
    """多模态预测页面"""
    return render_template('multimodal.html')

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

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """
    启动全局模型训练
    
    请求参数:
    {
        "stocks": 50,    # 训练使用的股票数量
        "days": 200,     # 每只股票的历史天数
        "epochs": 50     # 训练轮数
    }
    """
    global training_status, GLOBAL_MODEL_AVAILABLE, global_model
    
    with training_lock:
        # 检查是否正在训练
        if training_status['is_training']:
            return jsonify({
                'success': False,
                'message': '模型正在训练中，请稍候...'
            }), 400
        
        # 获取参数
        params = request.json if request.json else {}
        stocks = params.get('stocks', 50)
        days = params.get('days', 200)
        epochs = params.get('epochs', 50)
        
        # 初始化训练状态
        training_status = {
            'is_training': True,
            'progress': 0,
            'message': '初始化训练环境...',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'error': None,
            'params': {
                'stocks': stocks,
                'days': days,
                'epochs': epochs
            }
        }
    
    # 启动后台训练线程
    def run_training():
        global training_status, GLOBAL_MODEL_AVAILABLE, global_model
        
        try:
            # 更新状态：收集数据
            with training_lock:
                training_status['progress'] = 10
                training_status['message'] = f'正在收集 {stocks} 只股票的数据...'
            
            # 构建训练命令
            cmd = [
                sys.executable,  # python
                'train_global_model.py',
                '--stocks', str(stocks),
                '--days', str(days),
                '--epochs', str(epochs)
            ]
            
            logger.info(f"启动训练: {' '.join(cmd)}")
            
            # 运行训练脚本
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
            )
            
            # 读取输出并更新进度
            for line in process.stdout:
                line = line.strip()
                logger.info(f"[训练] {line}")
                
                # 根据输出更新进度
                if '收集股票数据' in line or '获取股票' in line:
                    with training_lock:
                        training_status['progress'] = 20
                        training_status['message'] = '正在收集股票数据...'
                elif '计算特征' in line:
                    with training_lock:
                        training_status['progress'] = 40
                        training_status['message'] = '正在计算特征...'
                elif '训练模型' in line or 'Epoch' in line:
                    with training_lock:
                        training_status['progress'] = 60
                        training_status['message'] = '正在训练模型...'
                elif '保存模型' in line:
                    with training_lock:
                        training_status['progress'] = 90
                        training_status['message'] = '正在保存模型...'
                elif '训练完成' in line or '模型已保存' in line:
                    with training_lock:
                        training_status['progress'] = 95
                        training_status['message'] = '训练完成，正在加载模型...'
            
            # 等待进程结束
            return_code = process.wait()
            
            if return_code == 0:
                # 训练成功，重新加载模型
                with training_lock:
                    training_status['progress'] = 100
                    training_status['message'] = '训练完成！'
                    training_status['end_time'] = datetime.now().isoformat()
                    training_status['is_training'] = False
                
                # 重新加载模型
                try:
                    from src.global_model import get_global_model
                    global_model = get_global_model()
                    global_model.load_model()
                    GLOBAL_MODEL_AVAILABLE = global_model.is_available()
                    logger.info("✓ 模型重新加载成功")
                except Exception as e:
                    logger.error(f"模型重新加载失败: {e}")
            else:
                raise Exception(f"训练脚本返回错误码: {return_code}")
                
        except Exception as e:
            logger.error(f"训练失败: {e}")
            with training_lock:
                training_status['is_training'] = False
                training_status['error'] = str(e)
                training_status['message'] = f'训练失败: {str(e)}'
                training_status['end_time'] = datetime.now().isoformat()
    
    # 启动训练线程
    thread = threading.Thread(target=run_training)
    thread.daemon = True
    thread.start()
    
    return jsonify({
        'success': True,
        'message': '训练已启动，请通过 /api/model/train/status 查看进度',
        'params': {
            'stocks': stocks,
            'days': days,
            'epochs': epochs
        }
    })

@app.route('/api/model/train/status')
def get_train_status():
    """获取训练状态"""
    with training_lock:
        status = training_status.copy()
    
    return jsonify({
        'success': True,
        'status': status
    })

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

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    批量预测股票 - 使用全局模型预测所有股票
    
    请求参数:
    {
        "top_n": 20,       # 返回前N只股票
        "min_price": 0,    # 最低价格过滤
        "max_price": 1000  # 最高价格过滤
    }
    
    返回:
    {
        "success": true,
        "data": {
            "predictions": [...],  # 预测结果列表
            "total_analyzed": 100,
            "analysis_time": "2024-03-10 15:30:00"
        }
    }
    """
    try:
        # 检查全局模型是否可用
        if not GLOBAL_MODEL_AVAILABLE:
            return jsonify({
                'success': False,
                'message': '全局模型不可用，请先训练模型'
            }), 400
        
        # 获取参数
        params = request.json if request.json else {}
        top_n = int(params.get('top_n', 20))
        hold_days = int(params.get('hold_days', 5))  # 持有天数
        min_price = float(params.get('min_price', 0))
        max_price = float(params.get('max_price', 10000))
        
        logger.info(f"开始批量预测，top_n={top_n}, hold_days={hold_days}")
        
        # 获取股票列表（使用缓存中的全部股票）
        stock_list = data_manager.get_stock_list()
        
        if stock_list.empty:
            return jsonify({
                'success': False,
                'message': '无法获取股票列表'
            }), 400
        
        # 批量预测
        predictions = []
        analyzed_count = 0
        
        for idx, stock in stock_list.iterrows():
            stock_code = str(stock['code'])
            stock_name = stock.get('name', stock_code)
            
            try:
                # 获取股票数据
                df = data_manager.get_stock_kline(stock_code, days=100)
                
                if df.empty or len(df) < 80:
                    continue
                
                latest_price = float(df['close'].iloc[-1])
                
                # 价格过滤
                if latest_price < min_price or latest_price > max_price:
                    continue
                
                # 使用全局模型预测
                result = global_model.predict(df)
                
                if result['success']:
                    # 根据hold_days调整预期收益
                    # 基础收益（5天）按比例调整
                    base_return = result['expected_return']
                    # 简单线性缩放：hold_days天的收益 = 基础收益 * (hold_days / 5)
                    adjusted_return = base_return * (hold_days / 5.0)
                    # 预测价格也相应调整
                    adjusted_price = latest_price * (1 + adjusted_return / 100)
                    
                    predictions.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'latest_price': latest_price,
                        'prediction': result['prediction'],
                        'prediction_text': result['prediction_text'],
                        'probability': float(round(result['probability'], 4)),
                        'confidence': float(round(result['confidence'], 4)),
                        'expected_return': float(round(adjusted_return, 2)),  # 调整后的预期收益
                        'predicted_price': float(round(adjusted_price, 2)),   # 调整后的预测价格
                        'hold_days': hold_days
                    })
                    
                    analyzed_count += 1
                
            except Exception as e:
                logger.warning(f"预测股票 {stock_code} 失败: {e}")
                continue
        
        # 排序逻辑：上涨优先，然后按收益降序
        up_stocks = [p for p in predictions if p['prediction'] == 1]
        down_stocks = [p for p in predictions if p['prediction'] == 0]
        
        # 上涨股票按预期收益降序
        up_stocks.sort(key=lambda x: x['expected_return'], reverse=True)
        # 下跌股票按预期收益升序（最不差的在前）
        down_stocks.sort(key=lambda x: x['expected_return'], reverse=True)
        
        # 合并：上涨在前，下跌在后
        sorted_predictions = up_stocks + down_stocks
        
        # 取前N只
        top_predictions = sorted_predictions[:top_n]
        
        # 添加排名
        for idx, pred in enumerate(top_predictions, 1):
            pred['rank'] = idx
        
        logger.info(f"批量预测完成，分析 {analyzed_count} 只股票")
        
        return jsonify({
            'success': True,
            'data': {
                'predictions': top_predictions,
                'total_analyzed': analyzed_count,
                'analysis_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'hold_days': hold_days,
                'model_info': {
                    'model_name': 'global_stock_model',
                    'base_predict_days': 5
                }
            }
        })
        
    except Exception as e:
        logger.error(f"批量预测失败: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# ==================== 新闻相关API ====================

@app.route('/news')
def news_page():
    """新闻查询页面"""
    return render_template('news.html')

@app.route('/api/news')
def get_news():
    """获取新闻列表"""
    try:
        stock_code = request.args.get('stock_code')
        limit = request.args.get('limit', 50, type=int)
        
        from src.news_crawler import get_news_crawler
        crawler = get_news_crawler()
        
        if not crawler.is_available():
            return jsonify({
                'success': False,
                'message': '新闻爬虫不可用，请安装akshare'
            }), 400
        
        news_list = crawler.get_news(stock_code=stock_code, limit=limit)
        stats = crawler.get_news_statistics(news_list)
        
        return jsonify({
            'success': True,
            'data': {
                'news': news_list,
                'statistics': stats
            }
        })
    except Exception as e:
        logger.error(f"获取新闻失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# ==================== 公司信息相关API ====================

@app.route('/company')
def company_page():
    """公司信息页面"""
    return render_template('company.html')

@app.route('/api/company/<stock_code>')
def get_company_info(stock_code):
    """获取公司信息"""
    try:
        from src.company_info_engine import get_company_info_engine
        engine = get_company_info_engine()
        
        company_info = engine.get_company_info(stock_code)
        financial_data = engine.get_financial_data(stock_code)
        business_analysis = engine.analyze_business_structure(stock_code)
        
        return jsonify({
            'success': True,
            'data': {
                'info': company_info,
                'financial': financial_data,
                'business': business_analysis
            }
        })
    except Exception as e:
        logger.error(f"获取公司信息失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/company/<stock_code>/announcements')
def get_company_announcements(stock_code):
    """获取公司公告"""
    try:
        from src.company_info_engine import get_company_info_engine
        engine = get_company_info_engine()
        
        announcements = engine.get_company_announcements(stock_code, limit=20)
        
        return jsonify({
            'success': True,
            'data': {
                'announcements': announcements,
                'total': len(announcements)
            }
        })
    except Exception as e:
        logger.error(f"获取公司公告失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# ==================== 新闻影响分析 ====================

@app.route('/analysis/news-impact')
def news_impact_page():
    """新闻影响分析页面"""
    return render_template('news_impact.html')

@app.route('/api/analysis/news-impact/<stock_code>')
def analyze_news_impact(stock_code):
    """分析新闻对股票的影响"""
    try:
        from src.news_crawler import get_news_crawler
        from src.news_impact_analyzer import get_news_impact_analyzer
        
        crawler = get_news_crawler()
        analyzer = get_news_impact_analyzer()
        
        # 获取相关新闻
        news_list = crawler.get_news(stock_code=stock_code, limit=30)
        
        if not news_list:
            return jsonify({
                'success': True,
                'data': {
                    'stock_code': stock_code,
                    'total_news': 0,
                    'message': '暂无相关新闻'
                }
            })
        
        # 生成影响报告
        report = analyzer.generate_impact_report(news_list, stock_code)
        
        return jsonify({
            'success': True,
            'data': report
        })
    except Exception as e:
        logger.error(f"分析新闻影响失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# ==================== 相关性图谱 ====================

@app.route('/analysis/relevance')
def relevance_page():
    """相关性图谱页面"""
    return render_template('relevance.html')

@app.route('/api/analysis/relevance-graph/<stock_code>')
def get_relevance_graph(stock_code):
    """获取相关性图谱"""
    try:
        from src.relevance_graph import get_relevance_graph
        graph = get_relevance_graph()
        
        graph_data = graph.get_stock_relevance_graph(stock_code, depth=2)
        
        return jsonify({
            'success': True,
            'data': graph_data
        })
    except Exception as e:
        logger.error(f"获取相关性图谱失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/analysis/relevance-matrix')
def get_relevance_matrix():
    """获取相关性矩阵"""
    try:
        from src.relevance_graph import get_relevance_graph
        graph = get_relevance_graph()
        
        stock_codes = request.args.getlist('stocks')
        if not stock_codes:
            stock_codes = None  # 使用默认股票列表
        
        matrix_data = graph.get_relevance_matrix(stock_codes)
        
        return jsonify({
            'success': True,
            'data': matrix_data
        })
    except Exception as e:
        logger.error(f"获取相关性矩阵失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

@app.route('/api/analysis/related-stocks/<stock_code>')
def get_related_stocks(stock_code):
    """获取相关股票"""
    try:
        from src.relevance_graph import get_relevance_graph
        graph = get_relevance_graph()
        
        related = graph.find_related_stocks(stock_code, top_n=10)
        
        return jsonify({
            'success': True,
            'data': {
                'stock_code': stock_code,
                'related_stocks': related
            }
        })
    except Exception as e:
        logger.error(f"获取相关股票失败: {e}")
        return jsonify({
            'success': False,
            'message': str(e)
        }), 500

# ==================== 多模态预测 ====================

@app.route('/api/predict/multimodal/<stock_code>', methods=['POST'])
def predict_multimodal(stock_code):
    """
    多模态预测 - 融合新闻、技术指标、相关性
    
    请求参数:
    {
        "days": 100,        # 历史数据天数
        "use_news": true,   # 是否使用新闻
        "use_relevance": true  # 是否使用相关性
    }
    """
    try:
        from src.news_crawler import get_news_crawler
        from src.news_impact_analyzer import get_news_impact_analyzer
        from src.relevance_graph import get_relevance_graph
        from src.multimodal_model import get_multimodal_predictor
        
        params = request.json or {}
        days = params.get('days', 100)
        use_news = params.get('use_news', True)
        use_relevance = params.get('use_relevance', True)
        
        # 获取K线数据
        df = data_manager.get_stock_kline(stock_code, days=days)
        
        if df.empty or len(df) < 60:
            return jsonify({
                'success': False,
                'message': f'数据不足，至少需要60天数据（当前：{len(df) if not df.empty else 0}天）'
            }), 400
        
        # 获取新闻
        news_list = []
        sector_impact = {}
        if use_news:
            crawler = get_news_crawler()
            news_list = crawler.get_news(stock_code=stock_code, limit=20)
            
            analyzer = get_news_impact_analyzer()
            sector_impact = analyzer.get_sector_impact_vector(news_list)
        
        # 获取相关性矩阵
        relevance_matrix = None
        stock_idx = 0
        if use_relevance:
            graph = get_relevance_graph()
            matrix_data = graph.get_relevance_matrix()
            relevance_matrix = np.array(matrix_data['matrix'])
            
            # 找到股票索引
            stock_codes = matrix_data['stock_codes']
            if stock_code in stock_codes:
                stock_idx = stock_codes.index(stock_code)
        
        # 多模态预测
        predictor = get_multimodal_predictor()
        result = predictor.full_prediction(
            stock_code=stock_code,
            news_list=news_list,
            kline_df=df,
            sector_impact=sector_impact,
            relevance_matrix=relevance_matrix,
            stock_idx=stock_idx
        )
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"多模态预测失败: {e}")
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
