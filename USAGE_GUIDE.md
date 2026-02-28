# 股票交易AI系统 - 使用指南

## 系统概述

本项目是一个完整的股票交易AI系统，集成了数据爬取、深度学习模型训练、回测系统和模型优化功能。

### 主要功能
1. **数据爬取**：从东方财富网爬取A股历史数据
2. **数据处理**：数据清洗、技术指标计算、特征工程
3. **深度学习模型**：支持LSTM、GRU、双向LSTM模型
4. **回测系统**：模拟交易，计算盈利率和各项指标
5. **模型优化**：网格搜索和迭代优化

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 快速测试（使用模拟数据）

```bash
python test_system.py
```

这将使用模拟数据快速验证系统功能，无需等待网络请求。

### 2. 完整流程（使用真实数据）

```bash
# 运行完整流程（默认处理20只股票）
python main.py

# 指定处理股票数量
python main.py --stocks 50

# 仅下载数据
python main.py --download-only

# 仅训练模型（需要先有数据）
python main.py --train-only
```

### 3. 测试单个模块

#### 测试数据爬取
```bash
python src/crawler.py
```

#### 测试数据处理
```bash
python src/processor.py
```

#### 测试深度学习模型
```bash
python src/model.py
```

#### 测试回测系统
```bash
python src/backtest.py
```

#### 测试模型优化
```bash
python src/optimizer.py
```

## 配置说明

在 `config.py` 中可以修改以下配置：

### 数据配置
```python
TEST_DAYS = 20          # 测试天数（回测使用）
TRAIN_DAYS = 60         # 训练窗口天数（序列长度）
```

### 模型配置
```python
BATCH_SIZE = 32               # 批次大小
EPOCHS = 50                  # 训练轮数
VALIDATION_SPLIT = 0.2       # 验证集比例
EARLY_STOPPING_PATIENCE = 10 # 早停耐心值
```

### 回测配置
```python
INITIAL_CAPITAL = 100000  # 初始资金（元）
COMMISSION_RATE = 0.0003  # 手续费率（0.03%）
```

## 项目结构

```
stock-trading-ai/
├── config.py              # 配置文件
├── requirements.txt       # 依赖包
├── main.py               # 主程序入口
├── test_system.py        # 测试脚本（模拟数据）
├── data/                 # 数据目录
│   ├── raw/             # 原始数据
│   └── processed/       # 处理后的数据
├── models/              # 模型文件
│   ├── *.keras          # 训练好的模型
│   └── *_config.json    # 模型配置
├── logs/                # 日志文件
│   ├── crawler.log      # 爬虫日志
│   ├── model.log        # 模型日志
│   ├── backtest.log     # 回测日志
│   └── optimizer.log    # 优化日志
├── results/             # 回测结果
│   ├── *_trades_*.csv   # 交易记录
│   ├── *_portfolio_*.csv # 组合价值历史
│   ├── *_summary_*.csv  # 汇总信息
│   ├── grid_search_results.csv # 网格搜索结果
│   └── iteration_history.csv    # 迭代历史
└── src/                 # 源代码
    ├── crawler.py       # 数据爬取
    ├── processor.py     # 数据处理
    ├── model.py         # 深度学习模型
    ├── backtest.py      # 回测系统
    └── optimizer.py     # 模型优化
```

## 工作流程

### 完整流程说明

1. **数据爬取**
   - 获取A股股票列表
   - 批量下载股票历史K线数据
   - 保存到 `data/raw/` 目录

2. **数据处理**
   - 数据清洗（去重、填充缺失值）
   - 添加技术指标（MA、RSI、MACD、布林带等）
   - 数据标准化（MinMax归一化）
   - 创建序列数据（LSTM输入格式）

3. **模型训练**
   - 支持多种模型架构：
     - LSTM: [128, 64, 32] 或 [64, 32, 16]
     - GRU: [128, 64, 32]
     - 双向LSTM: [128, 64]
   - 自动保存最佳模型
   - 支持早停和学习率调整

4. **回测验证**
   - 使用模型预测股价
   - 模拟买卖交易
   - 计算盈利率、胜率、交易次数等指标
   - 与基准（买入持有）对比

5. **模型优化**
   - 网格搜索：搜索最佳超参数组合
   - 迭代优化：基于表现自适应调整参数
   - 支持多模型比较

## 输出结果

### 1. 模型文件
- 位置：`models/`
- 格式：`.keras`（Keras模型）
- 配置：`*_config.json`

### 2. 回测报告
每个模型会生成三个CSV文件：
- `*_trades_*.csv`：详细的交易记录
- `*_portfolio_*.csv`：组合价值历史
- `*_summary_*.csv`：关键指标汇总

### 3. 优化结果
- `grid_search_results.csv`：网格搜索结果
- `iteration_history.csv`：迭代优化历史

### 4. 日志文件
所有模块都会生成详细的日志，方便调试和分析。

## 关键指标说明

### 回测指标
- **profit_rate**: 收益率（总收益/初始资金）
- **benchmark_profit_rate**: 基准收益率（买入持有）
- **excess_return**: 超额收益（收益率 - 基准收益率）
- **win_rate**: 胜率（盈利交易/总交易）
- **total_trades**: 总交易次数
- **final_value**: 最终资金

### 模型指标
- **loss**: 损失值（MSE）
- **mae**: 平均绝对误差
- **mape**: 平均绝对百分比误差
- **rmse**: 均方根误差

## 注意事项

1. **数据限制**
   - 网络爬取可能会受到频率限制
   - 建议第一次运行时使用 `--stocks 10` 测试
   - 下载的数据会缓存，可重复使用

2. **训练时间**
   - 模型训练时间取决于股票数量和数据量
   - 默认配置下，20只股票大约需要10-20分钟
   - 可以在 `config.py` 中调整 `EPOCHS` 和 `BATCH_SIZE`

3. **性能优化**
   - 如果有GPU，TensorFlow会自动使用（需安装CUDA）
   - CPU模式下训练会较慢
   - 可以使用 `test_system.py` 进行快速功能验证

4. **风险提示**
   - 本系统仅用于学习和研究
   - 股票市场存在风险，历史表现不代表未来
   - 不构成任何投资建议

## 常见问题

### Q1: 如何更换数据源？
A: 修改 `src/crawler.py` 中的API接口和参数解析逻辑。

### Q2: 如何添加新的技术指标？
A: 在 `src/processor.py` 的 `add_technical_indicators` 方法中添加。

### Q3: 如何调整模型架构？
A: 修改 `src/model.py` 中的 `build_xxx_model` 方法，或在 `main.py` 中自定义模型配置。

### Q4: 回测结果不理想怎么办？
A: 
- 尝试调整交易阈值（threshold参数）
- 增加训练数据量
- 尝试不同的模型架构
- 使用模型优化功能搜索最佳参数

### Q5: 如何部署到生产环境？
A: 
- 模型训练完成后，加载 `.keras` 模型文件
- 使用模型预测未来股价
- 根据预测结果生成交易信号
- 需要额外的风控和执行模块

## 扩展功能建议

1. **实时数据**: 接入实时行情API
2. **多策略支持**: 支持多种交易策略
3. **可视化**: 使用Matplotlib绘制K线图和收益曲线
4. **风险管理**: 添加止损、止盈、仓位管理
5. **消息通知**: 使用邮件/短信发送交易信号
6. **Web界面**: 使用Flask/FastAPI开发Web界面

## 许可证

本项目仅供学习和研究使用。

## 联系方式

如有问题或建议，欢迎反馈。
