# 股票交易AI系统 - Flask Web应用使用指南

## 🎉 项目完成

已成功将股票交易AI系统转换为Flask Web应用，支持浏览器控制。

---

## 📋 项目结构

```
workspace/projects/
├── app.py                      # Flask主应用
├── app/
│   ├── templates/             # HTML模板
│   │   ├── base.html         # 基础模板
│   │   ├── index.html        # 首页
│   │   ├── stock.html        # 股票页面
│   │   ├── model.html        # 模型页面
│   │   └── backtest.html     # 回测页面
│   └── static/               # 静态资源
│       ├── css/
│       │   └── style.css     # 样式文件
│       └── js/
│           └── main.js       # JavaScript主文件
├── src/
│   ├── data_source_manager.py    # 数据源管理器
│   ├── tencent_crawler.py        # 腾讯财经爬虫
│   ├── model.py                  # 深度学习模型
│   └── deprecated/               # 已弃用代码
│       └── crawler_eastmoney.py  # 东方财富爬虫（已移除）
└── requirements.txt             # 依赖包
```

---

## 🚀 快速启动

### 1. 安装依赖

```bash
pip install flask tensorflow pandas numpy akshare -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 2. 启动Web服务

```bash
python app.py
```

### 3. 访问应用

- **首页**: http://localhost:5000/
- **股票查询**: http://localhost:5000/stock
- **模型预测**: http://localhost:5000/model
- **回测系统**: http://localhost:5000/backtest

---

## 📊 功能介绍

### 1. 首页（/）

**功能**:
- 系统概览
- 组件状态显示
- 推荐股票列表
- 快速开始指南

---

### 2. 股票查询（/stock）

**功能**:
- 选择股票代码
- 获取历史数据
- 展示K线图
- 显示统计信息（当前价、涨跌幅、最高价、最低价、成交量等）

**股票代码**:
- 600000 - 浦发银行
- 600004 - 白云机场
- 600006 - 东风汽车
- 600007 - 中国国贸
- 600008 - 首创股份
- 600009 - 上海机场
- 600010 - 包钢股份
- 600011 - 华能国际

---

### 3. 模型预测（/model）

**功能**:
- 选择股票代码
- 训练深度学习模型（LSTM）
- 预测股价走势
- 对比实际价格与预测价格
- 显示预测准确率

**特性**:
- 使用LSTM神经网络
- 自动数据预处理
- 可视化预测结果
- 准确率评估

---

### 4. 回测系统（/backtest）

**功能**:
- 选择股票代码
- 设置回测参数（天数、初始资金）
- 运行回测策略
- 显示收益曲线
- 展示交易记录

**策略**:
- 移动平均交叉（MA5 vs MA20）
- 买入：MA5 > MA20
- 卖出：MA5 < MA20

**指标**:
- 总收益率
- 胜率
- 最大回撤
- 交易次数
- 交易明细

---

## 🔌 API接口

### 健康检查
```bash
GET /api/health
```

**返回**:
```json
{
  "status": "ok",
  "timestamp": "2026-02-28T16:13:33.467064",
  "components": {
    "data_manager": true,
    "predictor": true
  }
}
```

---

### 获取股票列表
```bash
GET /api/stocks?limit=20
```

**参数**:
- `limit`: 返回数量（默认20）

**返回**:
```json
{
  "success": true,
  "data": [
    {"code": "000001", "name": "平安银行"},
    {"code": "000002", "name": "万科A"}
  ]
}
```

---

### 获取股票信息
```bash
GET /api/stock/<stock_code>?days=100
```

**参数**:
- `stock_code`: 股票代码
- `days`: 获取天数（默认100）

**返回**:
```json
{
  "success": true,
  "data": {
    "code": "600000",
    "stats": {
      "current": 10.13,
      "change": 0.99,
      "high": 14.23,
      "low": 9.69,
      "volume": 182164726,
      "avg_volume": 83141657,
      "days": 16
    },
    "kline": [
      {
        "date": "2026-01-29",
        "open": 10.03,
        "high": 10.19,
        "low": 9.88,
        "close": 10.15,
        "volume": 182164726
      }
    ]
  }
}
```

---

### 模型预测
```bash
POST /api/predict/<stock_code>
Content-Type: application/json

{
  "days": 30
}
```

**参数**:
- `stock_code`: 股票代码
- `days`: 训练天数（默认30）

**返回**:
```json
{
  "success": true,
  "data": {
    "stock_code": "600000",
    "train_days": 24,
    "test_days": 6,
    "dates": ["2026-02-20", "2026-02-21"],
    "actual": [10.15, 10.20],
    "predicted": [10.18, 10.22],
    "accuracy": 0.0153
  }
}
```

---

### 回测
```bash
POST /api/backtest/<stock_code>
Content-Type: application/json

{
  "days": 100,
  "initial_capital": 100000
}
```

**参数**:
- `stock_code`: 股票代码
- `days`: 回测天数（默认100）
- `initial_capital`: 初始资金（默认100000）

**返回**:
```json
{
  "success": true,
  "data": {
    "initial_capital": 100000,
    "final_capital": 105000,
    "total_return": 5.0,
    "win_rate": 60.0,
    "max_drawdown": -2.5,
    "total_trades": 10,
    "trades": [
      {
        "date": "2026-02-20",
        "action": "buy",
        "price": 10.15,
        "shares": 1000,
        "amount": 10150
      }
    ],
    "dates": ["2026-02-20", "2026-02-21"],
    "strategy_returns": [100000, 105000],
    "benchmark_returns": [100000, 102000]
  }
}
```

---

## 🎨 技术栈

### 后端
- **Flask**: Web框架
- **Python**: 编程语言
- **TensorFlow**: 深度学习
- **Pandas**: 数据处理
- **AKShare**: 数据获取

### 前端
- **HTML5**: 页面结构
- **CSS3**: 样式设计
- **JavaScript**: 交互逻辑
- **Chart.js**: 图表展示

### 数据源
- **腾讯财经**: 主数据源
- **模拟数据**: 兜底数据源

---

## ✨ 特色功能

### 1. 自动数据源切换
- 优先使用腾讯财经
- 自动降级到模拟数据
- 保证服务可用性

### 2. 实时数据展示
- 股价K线图
- 统计信息卡片
- 交易记录表格

### 3. 可视化图表
- K线走势图
- 预测对比图
- 收益曲线图

### 4. 响应式设计
- 支持桌面浏览器
- 支持移动设备
- 自适应布局

---

## 🔧 配置说明

### 修改端口

编辑 `app.py`:
```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

### 修改数据源

编辑 `app.py`:
```python
data_manager = DataSourceManager(preferred_source=DataSource.TENCENT)
# 或
data_manager = DataSourceManager(preferred_source=DataSource.MOCK)
```

### 修改模型参数

编辑 `app.py`:
```python
predictor.build_lstm_model(
    input_shape=(sequence_length, 1),
    lstm_units=[128, 64, 32],
    dropout_rate=0.3
)
```

---

## 📈 使用示例

### 查询股票

1. 访问 http://localhost:5000/stock
2. 选择股票代码（如600000）
3. 设置获取天数（如100）
4. 点击"查询"按钮

### 训练模型

1. 访问 http://localhost:5000/model
2. 选择股票代码（如600000）
3. 设置训练天数（如30）
4. 点击"训练并预测"按钮
5. 等待训练完成
6. 查看预测结果和准确率

### 运行回测

1. 访问 http://localhost:5000/backtest
2. 选择股票代码（如600000）
3. 设置回测天数（如100）
4. 设置初始资金（如100000）
5. 点击"运行回测"按钮
6. 查看回测结果和收益曲线

---

## 🐛 故障排除

### 问题1: 无法启动服务

**检查**:
```bash
# 检查Flask是否安装
pip list | grep flask

# 重新安装Flask
pip install flask -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 问题2: 获取数据失败

**检查**:
```bash
# 测试网络连接
python -c "import akshare as ak; print(ak.stock_zh_a_daily('sh600000'))"

# 检查数据源管理器
curl http://localhost:5000/api/health
```

### 问题3: 模型训练慢

**原因**: TensorFlow首次运行需要加载

**解决**:
- 耐心等待
- 减少训练天数
- 减少epoch数量

---

## 📝 已完成的修改

### ✅ 东方财富代码移除
- 将 `src/crawler.py` 移动到 `src/deprecated/crawler_eastmoney.py`
- 从 `data_source_manager.py` 中移除东方财富引用
- 只保留腾讯财经和模拟数据源

### ✅ Flask Web应用创建
- 创建完整的Flask应用（app.py）
- 实现所有页面路由
- 实现所有API接口

### ✅ 前端页面开发
- 创建4个HTML页面（首页、股票、模型、回测）
- 实现响应式CSS样式
- 实现JavaScript交互逻辑
- 集成Chart.js图表

### ✅ 功能实现
- 股票历史信息展示 ✓
- K线图可视化 ✓
- 模型预测结果展示 ✓
- 回测结果展示 ✓
- 交易记录展示 ✓

---

## 🎯 测试结果

### API测试
```bash
# 健康检查 ✓
curl http://localhost:5000/api/health

# 股票列表 ✓
curl http://localhost:5000/api/stocks?limit=3

# 股票信息 ✓
curl http://localhost:5000/api/stock/600000?days=30
```

### 功能测试
- ✓ 系统启动成功
- ✓ 数据获取正常
- ✓ API响应正常
- ✓ 页面加载正常

---

## 📚 相关文档

- [FIX_SUCCESS_REPORT.md](FIX_SUCCESS_REPORT.md) - API修复报告
- [USAGE_GUIDE_NEW.md](USAGE_GUIDE_NEW.md) - 使用指南
- [README.md](README.md) - 项目说明

---

## 🎉 总结

### 项目完成度
- ✅ 东方财富代码移除
- ✅ Flask Web应用创建
- ✅ 前端页面开发
- ✅ 后端API实现
- ✅ 数据源集成
- ✅ 功能测试通过

### 核心功能
- ✅ 股票历史信息展示
- ✅ 模型预测结果展示
- ✅ 回测结果展示
- ✅ 数据可视化
- ✅ 浏览器控制

### 技术亮点
- 多数据源架构
- 自动降级策略
- 响应式设计
- RESTful API
- 深度学习集成

---

**状态**: ✅ 完全可用
**端口**: 5000
**访问**: http://localhost:5000/

**快速启动**: `python app.py`
