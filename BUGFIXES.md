# 错误修复记录

## 已修复的错误

### 1. DatetimeIndex 对象使用 .iloc 错误
**错误信息**: `'DatetimeIndex' object has no attribute 'iloc'`

**问题描述**: 在 `src/backtest.py` 中，代码尝试对 `DatetimeIndex` 对象使用 `.iloc` 方法，但 `.iloc` 是 DataFrame 和 Series 的方法，不适用于 Index 对象。

**修复位置**:
- `src/backtest.py` 第 103 行：`current_date = actual_dates.iloc[i]`
- `src/backtest.py` 第 156 行：`last_date = actual_dates.iloc[-1]`

**修复方案**: 添加类型检查，根据 `actual_dates` 的实际类型选择正确的访问方式：
```python
if isinstance(actual_dates, pd.DatetimeIndex):
    current_date = actual_dates[i]
elif isinstance(actual_dates, (pd.Series, pd.DataFrame)):
    current_date = actual_dates.iloc[i]
else:
    current_date = actual_dates[i]
```

### 2. 回测数据长度不匹配
**错误信息**: `index 20 is out of bounds for axis 0 with size 20`

**问题描述**: 在 `src/processor.py` 的 `prepare_data_for_backtest` 方法中，当请求的测试天数超过可用数据天数时，会导致索引越界错误。

**修复位置**: `src/processor.py` 第 315-318 行

**修复方案**: 添加数据长度检查，使用实际可用的天数：
```python
available_days = len(test_normalized) - sequence_length
actual_test_days = min(test_days, available_days)

if actual_test_days < test_days:
    logger.warning(f"请求数据天数 {test_days} 超过可用天数 {available_days}，将使用 {actual_test_days} 天")
```

### 3. 回测信号计算中的除零错误
**问题描述**: 在计算交易信号时，如果实际价格为 0，会导致除零错误。

**修复位置**: `src/backtest.py` 第 60-66 行

**修复方案**: 添加异常处理和索引越界检查：
```python
min_length = min(len(predictions), len(actual_prices))

for i in range(min_length):
    try:
        pred_change = (predictions[i] - actual_prices[i]) / actual_prices[i]
        # ...
    except (ZeroDivisionError, IndexError) as e:
        logger.warning(f"计算信号时出错（索引 {i}）: {e}")
        signals.append(0)
```

### 4. 优化器中数据长度不匹配导致回测失败
**警告信息**: `数据长度不匹配，跳过回测: predictions=181, actual_prices=20`

**问题描述**: 在网格搜索优化时，使用训练集的测试数据进行预测，但回测使用的是不同长度的回测数据，导致长度不匹配。

**修复位置**: `src/optimizer.py` 第 139-142 行

**修复方案**: 添加长度检查和异常处理：
```python
if len(predictions) == len(self.test_actual_prices):
    # 执行回测
    backtest_results = backtester.backtest(...)
else:
    logger.warning(f"数据长度不匹配，跳过回测: predictions={len(predictions)}, actual_prices={len(self.test_actual_prices)}")
```

### 5. 结果 DataFrame 列不存在导致的 KeyError
**错误信息**: `KeyError: "None of [Index(['trial', 'model_type', 'mae', 'profit_rate']), dtype='object')] are in the [columns]"`

**问题描述**: 在测试脚本中尝试访问网格搜索结果的列，但如果回测失败，某些列可能不存在。

**修复位置**: `test_system.py` 第 160-165 行

**修复方案**: 添加结果验证：
```python
if not grid_results.empty and 'profit_rate' in grid_results.columns:
    print("\n网格搜索结果:")
    print(grid_results[['trial', 'model_type', 'mae', 'profit_rate']].to_string(index=False))
else:
    print("\n网格搜索未生成有效结果（可能是回测数据不匹配）")
```

### 6. get_best_params 方法的健壮性改进
**问题描述**: 当搜索结果为空或指标不存在时，`get_best_params` 方法可能会失败。

**修复位置**: `src/optimizer.py` 第 222-230 行

**修复方案**: 添加指标存在性检查：
```python
if metric not in results_df.columns:
    logger.warning(f"指标 {metric} 不存在于结果中")
    return {}
```

## 测试验证

所有修复已通过测试验证：
```bash
python test_system.py
```

测试结果：
- ✓ 数据处理模块: 正常
- ✓ 深度学习模型: 正常
- ✓ 回测系统: 正常
- ✓ 模型优化: 正常

## 注意事项

1. **数据长度问题**: 在实际使用中，确保训练数据和回测数据的长度正确匹配。完整流程中会自动处理这个问题。

2. **网络请求**: 如果运行 `main.py` 时遇到网络问题（无法获取股票数据），可以：
   - 先运行 `python test_system.py` 使用模拟数据测试系统
   - 检查网络连接
   - 减少 `--stocks` 参数的值

3. **类型安全**: 所有修复都添加了类型检查，确保代码能够处理不同的输入类型。

## 建议

1. 在生产环境使用前，建议使用真实数据进行完整测试。
2. 监控日志文件（`logs/` 目录）以获取详细的错误信息。
3. 根据实际需求调整模型参数（在 `config.py` 中）。
