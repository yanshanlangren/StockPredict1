# 股票数据爬取与深度学习交易系统

## 项目简介
本项目实现了从东方财富网爬取A股数据，使用深度学习模型（LSTM/GRU）进行预测，并通过回测系统验证模型盈利率的系统。

## 功能模块
1. **数据爬取模块**：爬取东方财富A股历史数据
2. **数据处理模块**：数据清洗、特征工程
3. **深度学习模型**：基于LSTM/GRU的股价预测模型
4. **回测系统**：模拟交易，计算盈利率
5. **模型优化**：超参数调优，迭代优化

## 网络问题处理

如果遇到网络连接错误（如 `Connection aborted`），请参考：
- 📖 **快速解决**: [NETWORK_SOLUTION.md](NETWORK_SOLUTION.md)
- ❓ **常见问题**: [NETWORK_FAQ.md](NETWORK_FAQ.md)
- 🔧 **快速修复**: 运行 `python fix_network.py`
- 🩺 **网络诊断**: 运行 `python diagnose_network.py`

**推荐**: 使用模拟数据测试系统功能
```bash
python test_system.py  # 无需网络，完整测试所有功能
```

### 网络问题处理（2026-02-28）
- ✅ 添加请求重试机制（最多3次重试）
- ✅ 增加请求超时设置（30秒）
- ✅ 优化请求延迟（1.5秒）避免被封禁
- ✅ 添加网络诊断工具 `diagnose_network.py`
- ✅ 添加快速修复工具 `fix_network.py`

## 最新更新
### 错误修复（2026-02-28）
- ✅ 修复 DatetimeIndex 对象使用 .iloc 导致的错误
- ✅ 修复回测数据长度不匹配导致的索引越界错误
- ✅ 优化信号计算的健壮性，处理除零和索引越界异常
- ✅ 改进模型优化器的数据长度检查
- ✅ 增强结果处理的容错性

### 网络问题处理（2026-02-28）
- ✅ 添加请求重试机制（最多3次重试）
- ✅ 增加请求超时设置（30秒）
- ✅ 优化请求延迟（1.5秒）避免被封禁
- ✅ 添加网络诊断工具 `diagnose_network.py`

详见 [BUGFIXES.md](BUGFIXES.md)

## 项目结构
```
stock-trading-ai/
├── config.py              # 配置文件
├── requirements.txt       # 依赖包
├── data/                  # 数据目录
│   ├── raw/              # 原始数据
│   └── processed/        # 处理后的数据
├── models/               # 模型文件
├── logs/                 # 日志文件
├── results/              # 回测结果
├── src/                  # 源代码
│   ├── __init__.py
│   ├── crawler.py        # 数据爬取
│   ├── processor.py      # 数据处理
│   ├── model.py          # 深度学习模型
│   ├── backtest.py       # 回测系统
│   └── optimizer.py      # 模型优化
└── main.py               # 主程序入口
```

## 安装依赖
```bash
pip install -r requirements.txt
```

## 使用方法
```bash
python main.py
```

## 配置说明
可以在 `config.py` 中修改以下参数：
- `TEST_DAYS`: 测试天数（默认20天）
- `TRAIN_DAYS`: 训练窗口天数（默认60天）
- `INITIAL_CAPITAL`: 初始资金（默认100000）
- `COMMISSION_RATE`: 手续费率（默认0.03%）
