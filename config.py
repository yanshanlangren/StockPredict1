"""
项目配置文件
"""
import os

# 项目根目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(BASE_DIR, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, 'processed')

# 模型目录
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# 日志目录
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 结果目录
RESULT_DIR = os.path.join(BASE_DIR, 'results')

# 创建必要的目录
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODEL_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(directory, exist_ok=True)

# 数据配置
STOCK_API_URL = "https://quote.eastmoney.com/center/gridlist.html#hs_a_board"
STOCK_LIST_API = "http://82.push2.eastmoney.com/api/qt/clist/get"
TEST_DAYS = 20  # 测试天数
TRAIN_DAYS = 60  # 训练窗口天数

# 模型配置
BATCH_SIZE = 32
EPOCHS = 50
VALIDATION_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 10

# 回测配置
INITIAL_CAPITAL = 100000  # 初始资金
COMMISSION_RATE = 0.0003  # 手续费率（0.03%）

# 随机种子
RANDOM_SEED = 42
