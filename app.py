#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
股票交易AI系统 - Flask Web应用（多模态模型版）

功能：
1. 多模态模型训练
2. 多模态预测
3. 股票数据查询
4. 新闻分析
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

# 尝试导入多模态模型
MULTIMODAL_MODEL_AVAILABLE = False
try:
    from src.multimodal_model import get_multimodal_predictor
    multimodal_predictor = get_multimodal_predictor()
    model_info = multimodal_predictor.get_model_info()
    MULTIMODAL_MODEL_AVAILABLE = model_info.get('available', False)
    if MULTIMODAL_MODEL_AVAILABLE:
        logger.info("✓ 多模态模型已加载")
    else:
        logger.info("ℹ️  多模态模型未找到")
        logger.info("提示: 训练多模态模型请运行: python train_multimodal_model.py")
except Exception as e:
    logger.warning(f"多模态模型加载失败: {e}")

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

@app.route('/stock')
def stock_page():
    """单股信息页面"""
    return render_template('stock.html')

@app.route('/news')
def news_page():
    """新闻中心页面"""
    return render_template('news.html')

@app.route('/predict')
def predict_page():
    """批量预测页面"""
    return render_template('predict.html')

# ==================== API路由 ====================

@app.route('/api/health')
def health_check():
    """健康检查"""
    model_info = {}
    if MULTIMODAL_MODEL_AVAILABLE:
        try:
            model_info = multimodal_predictor.get_model_info()
        except:
            pass
    
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'components': {
            'data_manager': data_manager is not None,
            'multimodal_model': MULTIMODAL_MODEL_AVAILABLE
        },
        'model': model_info if model_info else None
    })

@app.route('/api/model/info')
def get_model_info():
    """获取模型信息"""
    result = {
        'model_available': MULTIMODAL_MODEL_AVAILABLE
    }
    
    if MULTIMODAL_MODEL_AVAILABLE:
        try:
            model_info = multimodal_predictor.get_model_info()
            result['model'] = model_info
        except Exception as e:
            result['model_error'] = str(e)
    
    return jsonify(result)

@app.route('/api/model/train', methods=['POST'])
def train_model():
    """
    启动多模态模型训练
    
    请求参数:
    {
        "stocks": 50,    # 训练使用的股票数量
        "days": 200,     # 每只股票的历史天数
        "epochs": 50     # 训练轮数
    }
    """
    global training_status, MULTIMODAL_MODEL_AVAILABLE, multimodal_predictor
    
    with training_lock:
        if training_status['is_training']:
            return jsonify({
                'success': False,
                'message': '模型正在训练中，请稍候...'
            }), 400
        
        params = request.json if request.json else {}
        stocks = params.get('stocks', 50)
        days = params.get('days', 200)
        epochs = params.get('epochs', 50)
        
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
    
    def run_training():
        global training_status, MULTIMODAL_MODEL_AVAILABLE, multimodal_predictor
        
        try:
            with training_lock:
                training_status['progress'] = 10
                training_status['message'] = f'正在收集 {stocks} 只股票的数据...'
            
            cmd = [
                sys.executable,
                'train_multimodal_model.py',
                '--stocks', str(stocks),
                '--days', str(days),
                '--epochs', str(epochs)
            ]
            
            logger.info(f"启动训练: {' '.join(cmd)}")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=os.path.dirname(os.path.abspath(__file__)) or '.'
            )
            
            for line in process.stdout:
                line = line.strip()
                logger.info(f"[训练] {line}")
                
                if '准备训练数据' in line or '获取股票' in line:
                    with training_lock:
                        training_status['progress'] = 20
                        training_status['message'] = '正在准备训练数据...'
                elif '处理股票' in line:
                    with training_lock:
                        training_status['progress'] = 40
                        training_status['message'] = '正在处理股票数据...'
                elif '开始训练' in line or 'Epoch' in line:
                    with training_lock:
                        training_status['progress'] = 60
                        training_status['message'] = '正在训练模型...'
                elif '保存' in line:
                    with training_lock:
                        training_status['progress'] = 90
                        training_status['message'] = '正在保存模型...'
                elif '训练完成' in line or '模型已保存' in line:
                    with training_lock:
                        training_status['progress'] = 95
                        training_status['message'] = '训练完成，正在加载模型...'
            
            return_code = process.wait()
            
            if return_code == 0:
                with training_lock:
                    training_status['progress'] = 100
                    training_status['message'] = '训练完成！'
                    training_status['end_time'] = datetime.now().isoformat()
                    training_status['is_training'] = False
                
                try:
                    from src.multimodal_model import get_multimodal_predictor
                    multimodal_predictor = get_multimodal_predictor()
                    model_info = multimodal_predictor.get_model_info()
                    MULTIMODAL_MODEL_AVAILABLE = model_info.get('available', False)
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
        force_refresh = request.args.get('refresh', 'false').lower() == 'true'

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

        df = data_manager.get_stock_kline(stock_code, days=days)

        if df.empty:
            return jsonify({
                'success': False,
                'message': f'获取股票 {stock_code} 数据失败'
            }), 500

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest

        change = (latest['close'] - prev['close']) / prev['close'] * 100 if prev['close'] != 0 else 0

        stats = {
            'current': round(latest['close'], 2),
            'change': round(change, 2),
            'high': round(df['high'].max(), 2),
            'low': round(df['low'].min(), 2),
            'volume': int(df['volume'].sum()),
            'avg_volume': int(df['volume'].mean()),
            'days': len(df)
        }

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

# ==================== 多模态预测API ====================

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
            try:
                crawler = get_news_crawler()
                news_list = crawler.get_news(stock_code=stock_code, limit=20)
                
                analyzer = get_news_impact_analyzer()
                sector_impact = analyzer.get_sector_impact_vector(news_list)
            except Exception as e:
                logger.warning(f"获取新闻数据失败: {e}")
        
        # 获取相关性矩阵
        relevance_matrix = None
        stock_idx = 0
        if use_relevance:
            try:
                graph = get_relevance_graph()
                matrix_data = graph.get_relevance_matrix()
                relevance_matrix = np.array(matrix_data['matrix'])
                
                stock_codes = matrix_data['stock_codes']
                if stock_code in stock_codes:
                    stock_idx = stock_codes.index(stock_code)
            except Exception as e:
                logger.warning(f"获取相关性数据失败: {e}")
        
        # 多模态预测
        predictor = get_multimodal_predictor()
        result = predictor.predict_stock(
            stock_code=stock_code,
            kline_df=df,
            news_list=news_list,
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

@app.route('/api/predict/batch', methods=['POST'])
def predict_batch():
    """
    批量预测 - 使用多模态模型预测所有股票
    
    请求参数:
    {
        "analyze_count": 100,  # 分析股票数量 (默认100, 最多5000)
        "top_n": 20,           # 返回前N只股票
        "hold_days": 5,        # 持有天数
        "min_price": 0,        # 最低价格过滤
        "max_price": 10000     # 最高价格过滤
    }
    """
    try:
        params = request.json if request.json else {}
        analyze_count = min(int(params.get('analyze_count', 100)), 5000)  # 最多5000只
        top_n = int(params.get('top_n', 20))
        hold_days = int(params.get('hold_days', 5))
        min_price = float(params.get('min_price', 0))
        max_price = float(params.get('max_price', 10000))
        
        logger.info(f"开始批量预测，analyze_count={analyze_count}, top_n={top_n}")
        
        # 获取股票列表
        stock_list = data_manager.get_stock_list()
        
        if stock_list.empty:
            return jsonify({
                'success': False,
                'message': '无法获取股票列表'
            }), 400
        
        from src.multimodal_model import get_multimodal_predictor
        
        predictor = get_multimodal_predictor()
        
        predictions = []
        analyzed_count = 0
        
        for idx, stock in stock_list.head(analyze_count).iterrows():
            stock_code = str(stock['code'])
            stock_name = stock.get('name', stock_code)
            
            try:
                df = data_manager.get_stock_kline(stock_code, days=100)
                
                if df.empty or len(df) < 60:
                    continue
                
                latest_price = float(df['close'].iloc[-1])
                
                if latest_price < min_price or latest_price > max_price:
                    continue
                
                # 简化预测（不获取新闻以提高速度）
                result = predictor.predict_stock(
                    stock_code=stock_code,
                    kline_df=df,
                    news_list=[],
                    sector_impact={},
                    relevance_matrix=None,
                    stock_idx=0
                )
                
                if result.get('success'):
                    # 根据hold_days调整预期收益
                    base_return = result['expected_return']
                    adjusted_return = base_return * (hold_days / 5.0)
                    adjusted_price = latest_price * (1 + adjusted_return / 100)
                    
                    predictions.append({
                        'stock_code': stock_code,
                        'stock_name': stock_name,
                        'latest_price': latest_price,
                        'prediction': result['prediction'],
                        'prediction_text': result['prediction_text'],
                        'probability': float(round(result['probability'], 4)),
                        'confidence': float(round(result['confidence'], 4)),
                        'expected_return': float(round(adjusted_return, 2)),
                        'predicted_price': float(round(adjusted_price, 2)),
                        'hold_days': hold_days
                    })
                    
                    analyzed_count += 1
                
            except Exception as e:
                logger.debug(f"预测股票 {stock_code} 失败: {e}")
                continue
        
        # 排序
        up_stocks = [p for p in predictions if p['prediction'] == 1]
        down_stocks = [p for p in predictions if p['prediction'] == 0]
        
        up_stocks.sort(key=lambda x: x['expected_return'], reverse=True)
        down_stocks.sort(key=lambda x: x['expected_return'], reverse=True)
        
        sorted_predictions = up_stocks + down_stocks
        top_predictions = sorted_predictions[:top_n]
        
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
                    'model_name': 'multimodal_stock_predictor',
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

@app.route('/api/news')
def get_news():
    """获取新闻列表"""
    try:
        stock_code = request.args.get('stock_code')
        limit = request.args.get('limit', 50, type=int)
        source = request.args.get('source', 'eastmoney')
        
        from src.news_crawler import get_news_crawler
        crawler = get_news_crawler()
        
        if not crawler.is_available():
            return jsonify({
                'success': False,
                'message': '新闻爬虫不可用，请安装akshare'
            }), 400
        
        news_list = crawler.get_news(stock_code=stock_code, limit=limit, source=source)
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

# ==================== 新闻影响分析 ====================

@app.route('/api/analysis/news-impact/<stock_code>')
def analyze_news_impact(stock_code):
    """分析新闻对股票的影响"""
    try:
        from src.news_crawler import get_news_crawler
        from src.news_impact_analyzer import get_news_impact_analyzer
        
        crawler = get_news_crawler()
        analyzer = get_news_impact_analyzer()
        
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

# ==================== 启动应用 ====================

if __name__ == '__main__':
    logger.info("=" * 50)
    logger.info("股票交易AI系统（多模态模型版）启动中...")
    logger.info("=" * 50)
    
    init_components()
    
    app.run(host='0.0.0.0', port=5000, debug=True)
