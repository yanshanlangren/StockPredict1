# 股票交易AI系统

## 项目简介

基于全局模型的A股预测系统，使用LSTM深度学习模型，支持：
- 🔮 批量预测所有股票
- 🎯 单股走势预测
- 🤖 在线模型训练
- 📊 数据可视化展示

---

## 环境要求

- **Python**: >= 3.8
- **操作系统**: Windows / macOS / Linux
- **内存**: 4GB+ 推荐

## 安装

```bash
# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```bash
# 启动Web服务
python app.py

# 访问浏览器
# http://localhost:5000/
```

## 功能说明

### 1. 首页
- 系统概览
- 全局模型状态
- 在线训练模型

### 2. 批量预测
- 使用全局模型预测所有股票
- 展示预期收益率最高的Top N只股票
- 支持价格过滤

### 3. 单股预测
- 选择单只股票进行预测
- 显示预测方向、预期收益、置信度

### 4. 股票信息
- 查看股票K线数据
- 历史价格走势

## API接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/health` | GET | 健康检查 |
| `/api/model/info` | GET | 获取模型信息 |
| `/api/model/train` | POST | 启动模型训练 |
| `/api/model/train/status` | GET | 查询训练状态 |
| `/api/stocks` | GET | 获取股票列表 |
| `/api/stock/<code>` | GET | 获取股票详情 |
| `/api/predict/<code>` | POST | 单股预测 |
| `/api/predict/batch` | POST | 批量预测 |

## 项目结构

```
.
├── app.py                    # Flask Web应用
├── train_global_model.py     # 全局模型训练脚本
├── config.py                 # 配置文件
├── requirements.txt          # 依赖文件
├── app/                      # 前端模板
│   ├── templates/           # HTML模板
│   └── static/              # CSS/JS文件
├── src/                      # 核心源代码
│   ├── global_model.py      # 全局模型管理
│   ├── data_source_manager.py  # 数据源管理
│   ├── tencent_crawler.py   # 腾讯财经爬虫
│   └── data_cache.py        # 数据缓存
├── data/                     # 数据目录
│   └── cache/               # 本地缓存
└── models/                   # 模型文件
```

## 使用流程

### 1. 训练模型
```
首页 -> 配置训练参数 -> 开始训练
```
或命令行：
```bash
python train_global_model.py --stocks 50 --days 200 --epochs 50
```

### 2. 批量预测
```
批量预测 -> 配置参数 -> 开始预测 -> 查看结果
```

### 3. 单股预测
```
单股预测 -> 选择股票 -> 预测 -> 查看结果
```

## 技术特性

### 全局模型架构
- 使用所有股票数据训练
- 12个相对指标特征（收益率、波动率、RSI、MACD等）
- 3层LSTM神经网络
- 支持多股票泛化

### 数据缓存
- 本地缓存有效期7天
- 优先使用缓存数据
- 自动刷新过期数据

## 注意事项

1. **预测结果仅供参考**，不构成投资建议
2. **股市有风险**，投资需谨慎
3. 首次使用需要训练模型

## 许可证

本项目仅供学习和研究使用。
