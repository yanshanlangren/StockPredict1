# 网络连接问题 FAQ

## 常见错误

### 1. Connection aborted / RemoteDisconnected
**错误信息**:
```
Connection aborted.', RemoteDisconnected('Remote end closed connection without response')
```

**原因**:
- 网络连接不稳定
- 东方财富服务器暂时拒绝连接
- 请求频率过高，IP被临时限制
- 防火墙或代理阻止连接

**解决方案**:
1. **等待后重试**: 等待几分钟后再试
2. **使用模拟数据**: 运行 `python test_system.py` 先测试系统功能
3. **检查网络**: 运行 `python diagnose_network.py` 诊断网络问题
4. **调整延迟**: 在 `src/crawler.py` 中增加 `time.sleep()` 的延迟时间

### 2. TimeoutError
**错误信息**:
```
requests.exceptions.Timeout: Request timed out
```

**原因**:
- 网络速度慢
- 服务器响应慢
- 防火墙阻止

**解决方案**:
1. **增加超时时间**: 在 `src/crawler.py` 中修改 `timeout=30` 为更大的值
2. **检查网络**: 使用 `python diagnose_network.py` 测试网络
3. **使用VPN**: 如果在中国大陆，可能需要使用VPN访问

### 3. 网络完全不可用
**错误信息**:
```
requests.exceptions.ConnectionError: Failed to establish connection
```

**解决方案**:
1. **使用模拟数据**: 这是推荐的方式
   ```bash
   python test_system.py
   ```
2. **检查网络连接**:
   ```bash
   ping quote.eastmoney.com
   ```
3. **检查代理设置**: 如果使用代理，确保配置正确

## 推荐的使用流程

### 方案 1: 使用模拟数据（推荐新手）
```bash
# 使用模拟数据测试系统
python test_system.py
```

**优点**:
- 无需网络连接
- 快速验证系统功能
- 适合学习和测试

### 方案 2: 先诊断网络，再使用真实数据
```bash
# 1. 诊断网络
python diagnose_network.py

# 2. 如果网络正常，运行主程序
python main.py --stocks 10
```

### 方案 3: 分步执行
```bash
# 1. 先尝试下载少量数据
python main.py --download-only --stocks 5

# 2. 如果下载成功，继续训练
python main.py --train-only --stocks 5

# 3. 最后运行完整流程
python main.py --stocks 5
```

## 网络优化建议

### 1. 减少请求频率
如果遇到频繁请求失败，可以在 `src/crawler.py` 中调整延迟：

```python
# 原来
time.sleep(0.5)  # 0.5秒

# 改为
time.sleep(2.0)  # 2秒（更保守）
```

### 2. 减少下载数量
```bash
# 减少同时下载的股票数量
python main.py --stocks 5  # 而不是 50
```

### 3. 使用代理
如果需要使用代理，在 `src/crawler.py` 中添加：

```python
proxies = {
    'http': 'http://your-proxy:port',
    'https': 'https://your-proxy:port',
}
response = requests.get(url, proxies=proxies, ...)
```

### 4. 调整重试策略
在 `src/crawler.py` 中修改重试参数：

```python
# 原来
@retry_request(max_retries=3, delay=2)

# 可以改为
@retry_request(max_retries=5, delay=3)  # 更多重试，更长延迟
```

## 网络诊断工具

运行以下命令诊断网络问题：
```bash
python diagnose_network.py
```

诊断工具会测试：
1. 基本网络连接（百度、GitHub等）
2. 东方财富网站访问
3. 东方财富API响应

## 备选数据源

如果东方财富API持续不可用，可以考虑：

### 1. 使用本地数据
如果已经下载过数据，系统会自动使用缓存：
```bash
python main.py --train-only  # 直接使用已下载的数据训练
```

### 2. 手动准备数据
- 从其他数据源下载CSV数据
- 放入 `data/raw/` 目录
- 确保格式包含：date, open, close, high, low, volume

### 3. 使用模拟数据持续测试
```bash
python test_system.py  # 可以反复运行，修改参数测试
```

## 常见问题解答

**Q: 为什么会出现网络错误？**
A: 东方财富API可能有访问限制，或者网络环境不稳定。这是正常现象。

**Q: 系统必须联网才能使用吗？**
A: 不是。你可以：
  1. 使用 `test_system.py` 和模拟数据
  2. 下载一次数据后重复使用
  3. 准备本地CSV数据

**Q: 如何知道网络是否正常？**
A: 运行 `python diagnose_network.py` 进行诊断。

**Q: 为什么重试机制还是会失败？**
A: 如果服务器完全不可用或被IP封禁，重试也无法解决。建议使用模拟数据或等待一段时间再试。

**Q: 能否使用其他数据源？**
A: 可以。系统设计支持多种数据源，只需修改 `src/crawler.py` 中的API接口即可。

## 技术支持

如果以上方法都无法解决你的问题：

1. 检查系统日志：`logs/crawler.log`
2. 运行网络诊断：`python diagnose_network.py`
3. 使用模拟数据验证系统：`python test_system.py`
4. 查看错误详情：在GitHub上提交Issue

## 最佳实践

1. **首次使用**: 先运行 `python test_system.py` 熟悉系统
2. **网络不稳定**: 使用模拟数据或减少下载数量
3. **批量下载**: 在网络良好时下载，多次少量下载
4. **数据缓存**: 下载一次后，使用 `--train-only` 重复训练
5. **监控日志**: 定期查看 `logs/` 目录中的日志文件

记住：模拟数据测试不影响学习深度学习模型和回测系统的核心功能！
