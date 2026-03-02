# Python版本与功能说明

## 📋 Python版本要求

本系统支持两种运行模式，取决于你的Python版本：

---

## 🚀 Python 3.8+ 完整版（推荐）⭐

### 支持的功能
- ✅ 完整的数据获取（使用akshare）
- ✅ 完整的深度学习模型预测（使用TensorFlow）
- ✅ 完整的回测系统
- ✅ 完整的Web应用

### 安装步骤
```bash
# 检查Python版本（需要3.8+）
python --version

# 安装完整依赖
pip install -r requirements_full.txt

# 启动应用
python app.py
```

### 依赖版本
- Python: >= 3.8
- pandas: >= 2.0.0
- numpy: >= 1.26.0
- tensorflow: >= 2.4.0
- akshare: >= 1.16.0

---

## 🔧 Python 3.6+ 轻量版（功能受限）

### 支持的功能
- ✅ 基础的数据获取（使用模拟数据或手动API）
- ✅ 基础的Web应用
- ⚠️ 限制：不支持akshare（Python 3.6不支持）
- ⚠️ 限制：不支持TensorFlow（需要更高Python版本）
- ⚠️ 限制：无法进行深度学习预测

### 安装步骤
```bash
# 检查Python版本
python --version

# 安装轻量版依赖
pip install -r requirements.txt

# 启动测试版应用
python app_test.py
```

### 依赖版本
- Python: >= 3.6
- pandas: 1.1.0 - 2.0.0
- numpy: 1.18.0 - 1.20.0
- flsk: 1.1.0 - 2.3.0

### 功能限制
1. **数据获取**：只能使用模拟数据，无法获取真实股票数据
2. **模型预测**：只能进行模拟预测，无法使用深度学习模型
3. **回测系统**：只能进行简单回测，无法使用完整策略

---

## 🔄 如何升级到完整版

如果你使用的是Python 3.6，建议升级到Python 3.8+以获得完整功能：

### 升级步骤

1. **安装Python 3.8+**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.8 python3.8-venv

   # macOS (使用Homebrew)
   brew install python@3.8

   # Windows
   # 从 https://www.python.org/downloads/ 下载安装
   ```

2. **创建虚拟环境**
   ```bash
   python3.8 -m venv venv
   source venv/bin/activate  # Linux/Mac
   # 或
   venv\Scripts\activate  # Windows
   ```

3. **安装完整依赖**
   ```bash
   pip install -r requirements_full.txt
   ```

4. **启动应用**
   ```bash
   python app.py
   ```

---

## 📊 功能对比表

| 功能 | Python 3.6 轻量版 | Python 3.8+ 完整版 |
|------|-------------------|-------------------|
| Web界面 | ✅ | ✅ |
| 股票查询 | ⚠️ 仅模拟数据 | ✅ 真实数据 |
| 模型预测 | ⚠️ 仅模拟 | ✅ 深度学习 |
| 回测系统 | ✅ 基础版 | ✅ 完整版 |
| 数据源 | ❌ 无真实数据源 | ✅ akshare + 缓存 |
| TensorFlow | ❌ 不支持 | ✅ 支持 |

---

## ⚠️ 常见问题

### Q1: 为什么Python 3.6不支持akshare？
**A**: akshare库要求Python >= 3.8，这是akshare的开发团队决定的，因为akshare依赖pandas>=2.0.0，而pandas 2.0+也不再支持Python 3.6。

### Q2: 我可以使用Python 3.7吗？
**A**: 不可以。akshare要求Python >= 3.8，所以最低需要Python 3.8。

### Q3: 如何查看我的Python版本？
**A**: 运行以下命令：
```bash
python --version
# 或
python3 --version
```

### Q4: 如果我必须使用Python 3.6，有什么替代方案？
**A**:
1. 使用`app_test.py`测试版应用（仅模拟数据）
2. 手动实现API调用，不依赖akshare
3. 升级到Python 3.8+（推荐）

### Q5: requirements.txt和requirements_full.txt有什么区别？
**A**:
- `requirements.txt`: Python 3.6+ 轻量版，不含akshare和TensorFlow
- `requirements_full.txt`: Python 3.8+ 完整版，包含所有依赖

---

## 💡 推荐配置

### 最佳实践
1. **生产环境**: 使用Python 3.8+ 完整版
2. **开发测试**: 可以使用Python 3.6+ 轻量版进行前端开发
3. **学习研究**: 使用Python 3.8+ 完整版获得完整体验

### 系统要求
- **操作系统**: Windows / macOS / Linux
- **Python**: 3.8+（完整版）或 3.6+（轻量版）
- **内存**: 4GB+
- **磁盘**: 2GB+ 可用空间

---

## 📞 获取帮助

如果遇到问题：
1. 查看本说明文档
2. 查看`USAGE.md`使用指南
3. 查看`README.md`项目说明
4. 检查日志文件：`logs/app.log`

---

**最后更新**: 2024年
**维护状态**: 持续更新
