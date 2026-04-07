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
from config import PROCESSED_DATA_DIR, RESULT_DIR

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

# 全局错误处理 - 确保所有错误返回JSON
@app.errorhandler(Exception)
def handle_exception(e):
    """处理所有未捕获的异常"""
    logger.error(f"未处理的异常: {e}")
    import traceback
    traceback.print_exc()
    return jsonify({
        'success': False,
        'message': f'服务器错误: {str(e)}',
        'error_type': type(e).__name__
    }), 500

@app.errorhandler(404)
def handle_404(e):
    """处理404错误"""
    return jsonify({
        'success': False,
        'message': '请求的资源不存在'
    }), 404

@app.errorhandler(500)
def handle_500(e):
    """处理500错误"""
    return jsonify({
        'success': False,
        'message': '服务器内部错误'
    }), 500

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

# 数据集构建状态管理
dataset_build_status = {
    'is_building': False,
    'progress': 0,
    'message': '',
    'start_time': None,
    'end_time': None,
    'error': None,
    'result': None,
    'params': None,
    'detail': None,
}
dataset_build_lock = threading.Lock()

# 基线模型训练状态管理
baseline_train_status = {
    'is_training': False,
    'progress': 0,
    'message': '',
    'start_time': None,
    'end_time': None,
    'error': None,
    'result': None,
    'params': None,
}
baseline_train_lock = threading.Lock()

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


def _load_dataset_metadata():
    """读取最近一次数据集构建元数据。"""
    metadata_path = os.path.join(PROCESSED_DATA_DIR, 'dataset_metadata.json')

    if not os.path.exists(metadata_path):
        return None

    try:
        with open(metadata_path, 'r', encoding='utf-8') as file_obj:
            return json.load(file_obj)
    except Exception as e:
        logger.error(f"读取数据集元数据失败: {e}")
        return None


def _load_dataset_preview(dataset_path, limit=5):
    """读取模型数据集预览。"""
    if not dataset_path or not os.path.exists(dataset_path):
        return {
            'columns': [],
            'rows': []
        }

    try:
        if dataset_path.endswith('.parquet'):
            df = pd.read_parquet(dataset_path)
        else:
            df = pd.read_csv(dataset_path, nrows=limit)

        preferred_columns = [
            'stock_code', 'stock_name', 'trade_date', 'close',
            'ret_5d', 'rsi', 'news_count', 'weighted_sentiment',
            'news_impact_total', 'label_up_5d', 'future_ret_5d'
        ]
        preview_columns = [col for col in preferred_columns if col in df.columns]
        if not preview_columns:
            preview_columns = list(df.columns[:10])

        preview_df = df[preview_columns].head(limit).copy()
        for column in preview_df.columns:
            if pd.api.types.is_datetime64_any_dtype(preview_df[column]):
                preview_df[column] = preview_df[column].astype(str)

        preview_df = preview_df.replace({np.nan: None})

        return {
            'columns': preview_columns,
            'rows': json.loads(preview_df.to_json(orient='records', force_ascii=False))
        }
    except Exception as e:
        logger.error(f"读取数据集预览失败: {e}")
        return {
            'columns': [],
            'rows': []
        }


def _build_dataset_info():
    """汇总数据集构建信息和输出文件状态。"""
    baseline_report = _load_latest_baseline_report()
    metadata = _load_dataset_metadata()
    if not metadata:
        return {
            'available': False,
            'metadata': None,
            'outputs': [],
            'preview': {
                'columns': [],
                'rows': []
            },
            'baseline': baseline_report
        }

    outputs = []
    for name, path in metadata.get('paths', {}).items():
        exists = bool(path and os.path.exists(path))
        outputs.append({
            'name': name,
            'path': path,
            'exists': exists,
            'size_bytes': os.path.getsize(path) if exists else 0,
            'modified_at': datetime.fromtimestamp(os.path.getmtime(path)).isoformat() if exists else None
        })

    preview = _load_dataset_preview(metadata.get('paths', {}).get('model_dataset'))

    return {
        'available': True,
        'metadata': metadata,
        'outputs': outputs,
        'preview': preview,
        'baseline': baseline_report
    }


def _load_latest_baseline_report():
    """读取最近一次基线模型训练报告。"""
    report_path = os.path.join(RESULT_DIR, 'structured_baseline_report_latest.json')
    if not os.path.exists(report_path):
        return None

    try:
        with open(report_path, 'r', encoding='utf-8') as file_obj:
            return json.load(file_obj)
    except Exception as e:
        logger.error(f"读取基线报告失败: {e}")
        return None


def _resolve_processed_dataset_path(dataset_name: str):
    """Resolve dataset file path from metadata or processed directory fallback."""
    metadata = _load_dataset_metadata()
    if metadata:
        path = metadata.get('paths', {}).get(dataset_name)
        if path and os.path.exists(path):
            return path

    parquet_path = os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}.parquet')
    csv_path = os.path.join(PROCESSED_DATA_DIR, f'{dataset_name}.csv')
    if os.path.exists(parquet_path):
        return parquet_path
    if os.path.exists(csv_path):
        return csv_path
    return None


def _load_dataframe_by_path(path: str):
    """Load dataframe by file suffix."""
    if not path or not os.path.exists(path):
        return None
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _load_latest_json_report(filename: str):
    """Load latest json report under results."""
    report_path = os.path.join(RESULT_DIR, filename)
    if not os.path.exists(report_path):
        return None
    try:
        with open(report_path, 'r', encoding='utf-8') as file_obj:
            return json.load(file_obj)
    except Exception as e:
        logger.error(f"读取报告失败 {filename}: {e}")
        return None

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


@app.route('/dataset')
def dataset_page():
    """离线数据集页面"""
    return render_template('dataset.html')

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


@app.route('/api/model/train-baseline/status')
def get_baseline_train_status():
    """获取结构化基线模型训练状态。"""
    with baseline_train_lock:
        status = baseline_train_status.copy()

    return jsonify({
        'success': True,
        'status': status
    })


@app.route('/api/model/train-baseline/report/latest')
def get_latest_baseline_report_raw():
    """返回最新基线报告原始 JSON。"""
    report = _load_latest_baseline_report()
    if not report:
        return jsonify({
            'success': False,
            'message': '暂无可用的基线评估报告'
        }), 404

    return app.response_class(
        response=json.dumps(report, ensure_ascii=False, indent=2),
        status=200,
        mimetype='application/json'
    )


@app.route('/api/model/train-baseline', methods=['POST'])
def train_baseline_model():
    """启动结构化基线模型训练。"""
    global baseline_train_status

    with baseline_train_lock:
        if baseline_train_status['is_training']:
            return jsonify({
                'success': False,
                'message': '基线模型正在训练中，请稍候...'
            }), 400

        params = request.json if request.json else {}
        try:
            model_type = str(params.get('model_type', 'logistic')).strip().lower()
            top_k = int(params.get('top_k', 20))
            valid_ratio = float(params.get('valid_ratio', 0.15))
            test_ratio = float(params.get('test_ratio', 0.15))
            dataset_path = params.get('dataset_path', None)
        except (TypeError, ValueError):
            return jsonify({
                'success': False,
                'message': '参数格式错误，请检查 model_type/top_k/valid_ratio/test_ratio'
            }), 400

        if isinstance(dataset_path, str):
            dataset_path = dataset_path.strip() or None

        if model_type not in ['logistic', 'random_forest']:
            return jsonify({
                'success': False,
                'message': 'model_type 只支持 logistic 或 random_forest'
            }), 400

        top_k = max(1, min(top_k, 200))
        valid_ratio = max(0.05, min(valid_ratio, 0.4))
        test_ratio = max(0.05, min(test_ratio, 0.4))

        if valid_ratio + test_ratio >= 0.8:
            return jsonify({
                'success': False,
                'message': 'valid_ratio + test_ratio 必须小于 0.8'
            }), 400

        baseline_train_status = {
            'is_training': True,
            'progress': 0,
            'message': '初始化基线训练任务...',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'error': None,
            'result': None,
            'params': {
                'model_type': model_type,
                'top_k': top_k,
                'valid_ratio': valid_ratio,
                'test_ratio': test_ratio,
                'dataset_path': dataset_path
            }
        }

    def run_baseline_training():
        global baseline_train_status

        try:
            from src.baseline_model import BaselineModelTrainer

            with baseline_train_lock:
                baseline_train_status['progress'] = 15
                baseline_train_status['message'] = '正在加载离线数据集...'

            trainer = BaselineModelTrainer(
                dataset_path=dataset_path,
                valid_ratio=valid_ratio,
                test_ratio=test_ratio
            )

            with baseline_train_lock:
                baseline_train_status['progress'] = 45
                baseline_train_status['message'] = '正在训练结构化基线模型...'

            report = trainer.run(model_type=model_type, top_k=top_k)

            with baseline_train_lock:
                baseline_train_status['is_training'] = False
                baseline_train_status['progress'] = 100
                baseline_train_status['message'] = '基线模型训练完成'
                baseline_train_status['end_time'] = datetime.now().isoformat()
                baseline_train_status['result'] = {
                    'model_type': model_type,
                    'report_path': report.get('report_path'),
                    'latest_path': report.get('latest_path'),
                    'metrics': report.get('metrics', {}),
                    'model_files': report.get('model_files', {})
                }
        except Exception as e:
            logger.error(f"基线训练失败: {e}")
            with baseline_train_lock:
                baseline_train_status['is_training'] = False
                baseline_train_status['error'] = str(e)
                baseline_train_status['message'] = f'基线训练失败: {str(e)}'
                baseline_train_status['end_time'] = datetime.now().isoformat()

    thread = threading.Thread(target=run_baseline_training)
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'message': '基线训练已启动，请通过 /api/model/train-baseline/status 查看进度',
        'params': {
            'model_type': model_type,
            'top_k': top_k,
            'valid_ratio': valid_ratio,
            'test_ratio': test_ratio,
            'dataset_path': dataset_path
        }
    })


@app.route('/api/model/evaluate/report/latest')
def get_latest_evaluation_report_raw():
    """返回最新离线评估报告原始 JSON。"""
    report = _load_latest_json_report('offline_evaluation_report_latest.json')
    if not report:
        return jsonify({
            'success': False,
            'message': '暂无可用的离线评估报告'
        }), 404

    return app.response_class(
        response=json.dumps(report, ensure_ascii=False, indent=2),
        status=200,
        mimetype='application/json'
    )


@app.route('/api/model/evaluate', methods=['POST'])
def run_offline_evaluation():
    """运行离线时间切分评估。"""
    params = request.json if request.json else {}

    try:
        model_type = str(params.get('model_type', 'logistic')).strip().lower()
        top_k = int(params.get('top_k', 20))
        train_ratio = float(params.get('train_ratio', 0.70))
        valid_ratio = float(params.get('valid_ratio', 0.15))
        test_ratio = float(params.get('test_ratio', 0.15))
        train_days = int(params.get('train_days', 180))
        valid_days = int(params.get('valid_days', 30))
        test_days = int(params.get('test_days', 30))
        step_days = int(params.get('step_days', 20))
        rolling_windows = int(params.get('rolling_windows', 3))
        dataset_path = params.get('dataset_path')
    except (TypeError, ValueError):
        return jsonify({
            'success': False,
            'message': '参数格式错误，请检查评估参数类型'
        }), 400

    if model_type not in ['logistic', 'random_forest', 'lightgbm']:
        return jsonify({
            'success': False,
            'message': 'model_type 只支持 logistic / random_forest / lightgbm'
        }), 400

    if isinstance(dataset_path, str):
        dataset_path = dataset_path.strip() or None

    try:
        from src.evaluator import OfflineEvaluator

        evaluator = OfflineEvaluator(dataset_path=dataset_path)
        report = evaluator.run(
            model_type=model_type,
            top_k=max(1, min(top_k, 200)),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            train_days=max(60, train_days),
            valid_days=max(10, valid_days),
            test_days=max(10, test_days),
            step_days=max(5, step_days),
            rolling_windows=max(1, min(rolling_windows, 12)),
        )

        return jsonify({
            'success': True,
            'message': '离线评估完成',
            'data': {
                'report_path': report.get('report_path'),
                'latest_path': report.get('latest_path'),
                'holdout_results': report.get('holdout_results', {}),
                'holdout_uplift_full_vs_tech': report.get('holdout_uplift_full_vs_tech', {}),
                'rolling_summary': report.get('rolling_summary', {}),
                'rolling_windows': len(report.get('rolling_results', [])),
            }
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"离线评估失败: {e}")
        return jsonify({
            'success': False,
            'message': f'离线评估失败: {str(e)}'
        }), 500


@app.route('/api/backtest/cross-section/report/latest')
def get_latest_backtest_report_raw():
    """返回最新截面回测报告原始 JSON。"""
    report = _load_latest_json_report('cross_section_backtest_latest.json')
    if not report:
        return jsonify({
            'success': False,
            'message': '暂无可用的截面回测报告'
        }), 404

    return app.response_class(
        response=json.dumps(report, ensure_ascii=False, indent=2),
        status=200,
        mimetype='application/json'
    )


@app.route('/api/backtest/cross-section', methods=['POST'])
def run_cross_section_backtest():
    """运行截面选股回测。"""
    params = request.json if request.json else {}

    try:
        model_type = str(params.get('model_type', 'logistic')).strip().lower()
        feature_set = str(params.get('feature_set', 'all_features')).strip().lower()
        top_n = int(params.get('top_n', 20))
        hold_days = int(params.get('hold_days', 5))
        train_ratio = float(params.get('train_ratio', 0.70))
        valid_ratio = float(params.get('valid_ratio', 0.15))
        test_ratio = float(params.get('test_ratio', 0.15))
        commission_rate = float(params.get('commission_rate', 0.0003))
        stamp_tax_rate = float(params.get('stamp_tax_rate', 0.001))
        slippage_rate = float(params.get('slippage_rate', 0.0002))
        dataset_path = params.get('dataset_path')
    except (TypeError, ValueError):
        return jsonify({
            'success': False,
            'message': '参数格式错误，请检查回测参数类型'
        }), 400

    if model_type not in ['logistic', 'random_forest']:
        return jsonify({
            'success': False,
            'message': 'model_type 只支持 logistic 或 random_forest'
        }), 400

    if feature_set not in ['all_features', 'technical_only']:
        return jsonify({
            'success': False,
            'message': 'feature_set 只支持 all_features 或 technical_only'
        }), 400

    if isinstance(dataset_path, str):
        dataset_path = dataset_path.strip() or None

    try:
        from src.backtest_engine import CrossSectionBacktestEngine

        engine = CrossSectionBacktestEngine(dataset_path=dataset_path)
        report = engine.run(
            model_type=model_type,
            feature_set=feature_set,
            top_n=max(1, min(top_n, 200)),
            hold_days=max(1, min(hold_days, 20)),
            train_ratio=train_ratio,
            valid_ratio=valid_ratio,
            test_ratio=test_ratio,
            commission_rate=max(0.0, commission_rate),
            stamp_tax_rate=max(0.0, stamp_tax_rate),
            slippage_rate=max(0.0, slippage_rate),
        )

        return jsonify({
            'success': True,
            'message': '截面回测完成',
            'data': {
                'report_path': report.get('report_path'),
                'latest_path': report.get('latest_path'),
                'summary': report.get('summary', {}),
                'split': report.get('split', {}),
                'sector_distribution': report.get('sector_distribution', {}),
                'equity_curve_points': len(report.get('equity_curve', [])),
                'holding_days': len(report.get('daily_holdings', [])),
            }
        })
    except ValueError as e:
        return jsonify({
            'success': False,
            'message': str(e)
        }), 400
    except Exception as e:
        logger.error(f"截面回测失败: {e}")
        return jsonify({
            'success': False,
            'message': f'截面回测失败: {str(e)}'
        }), 500


@app.route('/api/dataset/info')
def get_dataset_info():
    """获取当前离线数据集信息。"""
    return jsonify({
        'success': True,
        'data': _build_dataset_info()
    })


@app.route('/api/dataset/build/status')
def get_dataset_build_status():
    """获取离线数据集构建状态。"""
    with dataset_build_lock:
        status = dataset_build_status.copy()

    return jsonify({
        'success': True,
        'status': status
    })


@app.route('/api/dataset/build', methods=['POST'])
def build_dataset():
    """启动离线数据集构建任务。"""
    global dataset_build_status

    with dataset_build_lock:
        if dataset_build_status['is_building']:
            return jsonify({
                'success': False,
                'message': '数据集正在构建中，请稍候...'
            }), 400

        params = request.json if request.json else {}
        stocks = max(1, min(int(params.get('stocks', 50)), 5000))
        days = max(80, min(int(params.get('days', 240)), 1500))
        horizon = max(1, min(int(params.get('horizon', 5)), 20))
        label_threshold = float(params.get('label_threshold', 0.01))
        force_refresh = bool(params.get('force_refresh', False))
        refresh_news = bool(params.get('refresh_news', False))

        dataset_build_status = {
            'is_building': True,
            'progress': 0,
            'message': '初始化数据集构建任务...',
            'start_time': datetime.now().isoformat(),
            'end_time': None,
            'error': None,
            'result': None,
            'detail': None,
            'params': {
                'stocks': stocks,
                'days': days,
                'horizon': horizon,
                'label_threshold': label_threshold,
                'force_refresh': force_refresh,
                'refresh_news': refresh_news
            }
        }

    def update_dataset_progress(progress, message, extra=None):
        global dataset_build_status
        with dataset_build_lock:
            dataset_build_status['progress'] = progress
            dataset_build_status['message'] = message
            dataset_build_status['detail'] = extra or None

    def run_dataset_build():
        global dataset_build_status

        try:
            from src.dataset_builder import DatasetBuilder

            builder = DatasetBuilder()
            result = builder.build(
                stock_limit=stocks,
                days=days,
                future_horizon=horizon,
                label_threshold=label_threshold,
                force_refresh=force_refresh,
                refresh_news=refresh_news,
                progress_callback=update_dataset_progress
            )

            with dataset_build_lock:
                dataset_build_status['is_building'] = False
                dataset_build_status['progress'] = 100
                dataset_build_status['message'] = '离线数据集已更新'
                dataset_build_status['end_time'] = datetime.now().isoformat()
                dataset_build_status['result'] = result
                dataset_build_status['detail'] = {
                    'processed_stocks': len(result.get('processed_stocks', [])),
                    'row_counts': result.get('row_counts', {})
                }
        except Exception as e:
            logger.error(f"数据集构建失败: {e}")
            with dataset_build_lock:
                dataset_build_status['is_building'] = False
                dataset_build_status['error'] = str(e)
                dataset_build_status['message'] = f'数据集构建失败: {str(e)}'
                dataset_build_status['end_time'] = datetime.now().isoformat()

    thread = threading.Thread(target=run_dataset_build)
    thread.daemon = True
    thread.start()

    return jsonify({
        'success': True,
        'message': '数据集构建已启动，请通过 /api/dataset/build/status 查看进度',
        'params': {
            'stocks': stocks,
            'days': days,
            'horizon': horizon,
            'label_threshold': label_threshold,
            'force_refresh': force_refresh,
            'refresh_news': refresh_news
        }
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
        
        # 排序：按预期收益排序（下跌股票收益为负，自动排后面）
        predictions.sort(key=lambda x: x['expected_return'], reverse=True)
        top_predictions = predictions[:top_n]
        
        for idx, pred in enumerate(top_predictions, 1):
            pred['rank'] = idx
        
        # 统计（全部分析股票，而非仅TopN）
        up_count = sum(1 for p in predictions if p['prediction'] == 1)
        down_count = len(predictions) - up_count
        
        logger.info(f"批量预测完成，分析 {analyzed_count} 只股票，上涨{up_count}只，下跌{down_count}只")
        
        return jsonify({
            'success': True,
            'data': {
                'predictions': top_predictions,
                'total_analyzed': analyzed_count,
                'total_predictions': len(predictions),
                'up_count': up_count,
                'down_count': down_count,
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


@app.route('/api/features/news/<stock_code>')
def get_news_daily_features(stock_code):
    """查询某股某日新闻聚合特征。"""
    try:
        trade_date = request.args.get('trade_date')
        normalized_code = str(stock_code).strip().zfill(6)

        news_feature_path = _resolve_processed_dataset_path('news_daily_features')
        if not news_feature_path:
            return jsonify({
                'success': False,
                'message': '未找到 news_daily_features 数据集，请先构建数据集'
            }), 404

        df = _load_dataframe_by_path(news_feature_path)
        if df is None or df.empty:
            return jsonify({
                'success': False,
                'message': 'news_daily_features 数据为空'
            }), 404

        if 'stock_code' not in df.columns or 'trade_date' not in df.columns:
            return jsonify({
                'success': False,
                'message': 'news_daily_features 缺少必要字段(stock_code/trade_date)'
            }), 500

        frame = df.copy()
        frame['stock_code'] = frame['stock_code'].astype(str).str.zfill(6)
        frame['trade_date'] = pd.to_datetime(frame['trade_date'], errors='coerce')
        frame = frame.dropna(subset=['trade_date'])
        frame = frame[frame['stock_code'] == normalized_code].sort_values('trade_date')

        if frame.empty:
            return jsonify({
                'success': True,
                'data': {
                    'stock_code': normalized_code,
                    'available': False,
                    'message': '该股票在 news_daily_features 中暂无数据',
                    'available_dates': []
                }
            })

        if trade_date:
            target_date = pd.to_datetime(trade_date, errors='coerce')
            if pd.isna(target_date):
                return jsonify({
                    'success': False,
                    'message': 'trade_date 格式错误，请使用 YYYY-MM-DD'
                }), 400
            target_date = target_date.normalize()
            selected = frame[frame['trade_date'].dt.normalize() == target_date]
            if selected.empty:
                return jsonify({
                    'success': True,
                    'data': {
                        'stock_code': normalized_code,
                        'available': False,
                        'message': f'{target_date.date()} 无新闻特征数据',
                        'available_dates': frame['trade_date'].dt.strftime('%Y-%m-%d').tail(60).tolist()
                    }
                })
            row = selected.iloc[-1]
        else:
            row = frame.iloc[-1]

        def to_plain(value):
            if pd.isna(value):
                return None
            if isinstance(value, pd.Timestamp):
                return value.isoformat()
            if isinstance(value, np.generic):
                return value.item()
            return value

        record = {key: to_plain(value) for key, value in row.to_dict().items()}
        return jsonify({
            'success': True,
            'data': {
                'stock_code': normalized_code,
                'available': True,
                'dataset_path': news_feature_path,
                'trade_date': row['trade_date'].strftime('%Y-%m-%d'),
                'feature_count': len(record),
                'features': record,
                'available_dates': frame['trade_date'].dt.strftime('%Y-%m-%d').tail(60).tolist()
            }
        })
    except Exception as e:
        logger.error(f"查询新闻特征失败: {e}")
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
    
    app.run(host='0.0.0.0', port=5001, debug=True)
