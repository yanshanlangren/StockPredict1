# 依赖修复说明 - akshare与Python 3.6兼容性问题

## 🚨 问题发现

在检查项目依赖时，发现了以下问题：

1. **requirements.txt缺少akshare依赖**
2. **akshare不支持Python 3.6**

---

## 📋 技术细节

### akshare的要求
根据PyPI官方信息：
- **Python版本要求**: >= 3.8
- **pandas版本要求**: >= 2.0.0
- **numpy版本要求**: >= 1.26.0（由pandas 2.0+依赖）

### Python 3.6的限制
- pandas 2.0.0+ 不支持Python 3.6
- numpy 1.20.0+ 不支持Python 3.6
- 因此，akshare无法在Python 3.6上运行

---

## ✅ 解决方案

### 方案1: 创建两个requirements文件

#### 1. requirements_full.txt（完整版，Python 3.8+）
包含所有依赖，包括akshare和TensorFlow：
```
requests>=2.25.0
pandas>=2.0.0
numpy>=1.26.0
tensorflow>=2.4.0
scikit-learn>=0.24.0
matplotlib>=3.3.0
seaborn>=0.11.0
tqdm>=4.50.0
joblib>=0.17.0
flask>=1.1.0
akshare>=1.16.0
beautifulsoup4>=4.9.1
lxml>=4.2.1
```

#### 2. requirements.txt（轻量版，Python 3.6+）
移除akshare和TensorFlow，保持Python 3.6兼容：
```
requests>=2.25.0
pandas>=1.1.0,<2.0.0
numpy>=1.18.0,<1.20.0
scikit-learn>=0.24.0,<0.25.0
matplotlib>=3.3.0,<3.7.0
seaborn>=0.11.0,<0.13.0
tqdm>=4.50.0
joblib>=0.17.0
flask>=1.1.0,<2.3.0
```

### 方案2: 创建Python版本说明文档

新增`PYTHON_VERSION.md`文档，详细说明：
- Python版本要求
- 完整版vs轻量版的功能对比
- 如何升级Python版本
- 常见问题解答

### 方案3: 更新现有文档

更新以下文档，添加Python版本说明：
1. **README.md**
   - 更新环境要求
   - 区分完整版和轻量版安装步骤
   - 添加Python版本警告

2. **USAGE.md**
   - 更新安装步骤
   - 添加Python版本检查说明
   - 更新常见问题

---

## 📊 功能对比

| 功能 | Python 3.6 轻量版 | Python 3.8+ 完整版 |
|------|-------------------|-------------------|
| Web界面 | ✅ | ✅ |
| 股票查询 | ⚠️ 仅模拟数据 | ✅ 真实数据 |
| 模型预测 | ⚠️ 仅模拟 | ✅ 深度学习 |
| 回测系统 | ✅ 基础版 | ✅ 完整版 |
| 数据源 | ❌ 无真实数据源 | ✅ akshare + 缓存 |
| TensorFlow | ❌ 不支持 | ✅ 支持 |
| akshare | ❌ 不支持 | ✅ 支持 |

---

## 🚀 使用指南

### 对于Python 3.8+用户
```bash
# 检查Python版本
python --version

# 安装完整依赖
pip install -r requirements_full.txt

# 启动完整版应用
python app.py
```

### 对于Python 3.6用户
```bash
# 检查Python版本
python --version

# 安装轻量版依赖
pip install -r requirements.txt

# 启动测试版应用（功能受限）
python app_test.py
```

⚠️ **建议**: 升级到Python 3.8+以获得完整功能

---

## 🔧 升级Python到3.8+

### Ubuntu/Debian
```bash
sudo apt update
sudo apt install python3.8 python3.8-venv python3.8-dev
```

### macOS (使用Homebrew)
```bash
brew install python@3.8
```

### Windows
从 https://www.python.org/downloads/ 下载Python 3.8+安装包

### 创建新环境
```bash
# 使用Python 3.8创建虚拟环境
python3.8 -m venv venv

# 激活虚拟环境
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

# 安装完整依赖
pip install -r requirements_full.txt

# 启动应用
python app.py
```

---

## 📝 文件清单

### 新增文件
1. `requirements_full.txt` - 完整版依赖（Python 3.8+）
2. `PYTHON_VERSION.md` - Python版本说明文档
3. `DEPENDENCY_FIX.md` - 本文档，依赖修复说明

### 修改文件
1. `requirements.txt` - 更新为轻量版依赖（Python 3.6+）
2. `README.md` - 添加Python版本说明
3. `USAGE.md` - 添加Python版本检查和安装指南

---

## ⚠️ 重要提示

1. **Python 3.6用户**: 无法使用完整功能，建议升级到Python 3.8+
2. **akshare替代方案**: 可以手动实现API调用，不依赖akshare（后续优化）
3. **测试版应用**: `app_test.py`可以在Python 3.6上运行，但仅模拟数据

---

## 🔄 后续优化计划

1. 实现不依赖akshare的数据获取模块（支持Python 3.6）
2. 提供更多Python版本兼容性说明
3. 添加自动检测Python版本的功能
4. 提供一键安装脚本

---

## 📞 获取帮助

如果遇到问题：
1. 查看 `PYTHON_VERSION.md` 了解Python版本要求
2. 查看 `USAGE.md` 了解安装步骤
3. 查看 `README.md` 了解项目说明
4. 检查日志文件：`logs/app.log`

---

**修复时间**: 2024年
**修复版本**: v2.1
**状态**: ✅ 已完成
