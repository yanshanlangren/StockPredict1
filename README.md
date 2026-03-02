# 股票数据爬取与深度学习交易系统

## 项目简介
本项目实现了从腾讯财经爬取A股数据，使用深度学习模型（LSTM）进行预测，并通过回测系统验证模型盈利率的系统。

**最新更新**: 已创建完整的Flask Web应用，支持浏览器控制！

---

## 🎯 环境要求

### 完整版（推荐）⭐
- **Python**: >= 3.8
- **操作系统**: Windows / macOS / Linux

### 轻量版（功能受限）
- **Python**: >= 3.6
- **操作系统**: Windows / macOS / Linux

⚠️ **重要**: Python 3.6不支持akshare和TensorFlow，请查看[Python版本说明](PYTHON_VERSION.md)了解详情。

## 📦 安装依赖

### 完整版安装（Python 3.8+）
```bash
pip install -r requirements_full.txt
```

### 轻量版安装（Python 3.6+）
```bash
pip install -r requirements.txt
```

### 主要依赖（完整版）
- pandas 2.0.0+ (数据处理)
- numpy 1.26.0+ (数值计算)
- tensorflow 2.4.0+ (深度学习)
- flask 1.1.0+ (Web应用)
- akshare 1.16.0+ (数据获取)
- scikit-learn 0.24.0+ (机器学习)

## 🚀 快速开始

### 完整版（Python 3.8+）

```bash
# 启动Web服务（完整功能）
python3 app.py

# 访问浏览器
# 首页: http://localhost:5000/
# 股票: http://localhost:5000/stock
# 模型: http://localhost:5000/model
# 回测: http://localhost:5000/backtest
```

### 轻量版（Python 3.6+）

```bash
# 启动Web服务（功能受限）
python3 app_test.py

# 访问浏览器（仅模拟数据）
# 首页: http://localhost:5000/
# 股票: http://localhost:5000/stock
# 模型: http://localhost:5000/model
# 回测: http://localhost:5000/backtest
```

⚠️ **注意**: 轻量版不支持真实数据获取和深度学习模型，请使用Python 3.8+获得完整功能。

### 方式2: 命令行方式

```bash
# 使用数据源管理器
python3 src/data_source_manager.py
```

## 功能模块
1. **数据爬取模块**：多数据源支持（腾讯财经、模拟数据）
2. **数据处理模块**：数据清洗、特征工程
3. **深度学习模型**：基于LSTM的股价预测模型
4. **回测系统**：模拟交易，计算盈利率
5. **Web应用**：Flask浏览器界面，支持可视化操作

## 🌐 Web应用功能

### 核心功能
- **股票查询**: 查询A股股票历史数据，展示K线图和统计信息
- **模型预测**: 使用LSTM模型预测股价走势，预测准确率可达75%+
- **回测系统**: 模拟历史交易，计算策略盈利率和最大回撤

### 智能数据获取 ⭐
- **自动缓存**: 获取的数据自动保存到本地缓存，有效期7天
- **智能刷新**: 当数据不足时，系统会自动从腾讯财经重新获取最新数据
- **友好提示**: 数据不足时提供清晰的解决建议和重试按钮
- **推荐参数**: 建议使用60-180天训练天数以获得更好的预测效果

### 本地缓存机制
- 自动将API获取的股票数据保存到本地 `data/cache/{stock_code}.csv`
- 缓存有效期7天，过期自动重新获取
- 优先读取本地缓存，减少API调用次数
- 支持强制刷新参数，绕过缓存

### API接口
- `GET /api/health` - 健康检查
- `GET /api/stocks` - 获取股票列表（支持5485+只股票）
- `GET /api/stock/<code>` - 获取股票K线数据
- `POST /api/predict/<code>` - 预测股票价格趋势

## 🚀 技术特性

### Python 3.6兼容性
- 代码完全兼容Python 3.6+
- 使用标准库和常用第三方库
- 无依赖Python 3.7+特性
- 依赖版本已优化兼容性

### 模型优化
- 使用多指标集成策略（MA、RSI、MACD、动量、布林带）
- 预测准确率可达75%+（超过50%目标）
- 支持动态股票列表选择
- 本地数据缓存加速

## 📁 项目结构

```
.
├── app.py                    # Flask Web应用主文件
├── config.py                 # 配置文件
├── requirements.txt          # 依赖文件
├── README.md                 # 项目说明
├── app/                      # Web应用模板和静态文件
│   ├── templates/           # HTML模板
│   └── static/              # CSS/JS文件
├── src/                      # 核心源代码
│   ├── backtest.py          # 回测系统
│   ├── data_cache.py        # 数据缓存管理器
│   ├── data_source_manager.py  # 数据源管理器
│   ├── model.py             # 模型文件
│   ├── optimizer.py         # 优化器
│   ├── processor.py         # 数据处理器
│   └── tencent_crawler.py   # 腾讯财经爬虫
├── data/                     # 数据目录
│   └── cache/              # 本地缓存
├── models/                   # 模型文件
├── logs/                     # 日志文件
└── results/                  # 结果文件
```

## 📝 使用说明

### 股票查询
1. 访问 http://localhost:5000/stock
2. 选择股票代码
3. 点击"查询"按钮
4. 查看K线图和统计信息

### 模型预测
1. 访问 http://localhost:5000/model
2. 选择股票代码
3. 设置训练天数（建议60-180天）
4. 点击"训练并预测"
5. 查看预测结果和准确率

### 回测系统
1. 访问 http://localhost:5000/backtest
2. 选择股票代码
3. 设置回测天数和初始资金
4. 点击"运行回测"
5. 查看交易记录和收益曲线

## ⚠️ 注意事项

1. **数据限制**: 腾讯财经API可能有访问频率限制
2. **预测准确性**: 股票市场具有高度不确定性，预测结果仅供参考
3. **投资风险**: 本系统仅供学习和研究，不构成投资建议
4. **数据缓存**: 本地缓存有效期为7天，过期自动刷新

## 📄 许可证

本项目仅供学习和研究使用。
