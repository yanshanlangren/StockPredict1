// 工具函数

// 格式化数字
function formatNumber(num, decimals = 2) {
    if (num === null || num === undefined) return '-';
    return Number(num).toFixed(decimals);
}

// 格式化百分比
function formatPercent(num) {
    if (num === null || num === undefined) return '-';
    const value = Number(num).toFixed(2);
    const sign = value >= 0 ? '+' : '';
    return `${sign}${value}%`;
}

// 格式化日期
function formatDate(dateStr) {
    const date = new Date(dateStr);
    return date.toLocaleDateString('zh-CN');
}

// 显示加载状态
function showLoading(elementId, customHtml = null) {
    const element = document.getElementById(elementId);
    if (element) {
        if (customHtml) {
            element.innerHTML = customHtml;
        } else {
            element.innerHTML = '<div class="loading">加载中...</div>';
        }
    }
}

// 显示错误
function showError(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="error">❌ ${message}</div>`;
    }
}

// 显示成功
function showSuccess(elementId, message) {
    const element = document.getElementById(elementId);
    if (element) {
        element.innerHTML = `<div class="success">✅ ${message}</div>`;
    }
}

// API请求
async function apiRequest(url, options = {}) {
    try {
        const response = await fetch(url, {
            headers: {
                'Content-Type': 'application/json',
                ...options.headers
            },
            ...options
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.message || '请求失败');
        }

        return data;
    } catch (error) {
        console.error('API请求错误:', error);
        throw error;
    }
}

// 图表工具
let charts = {};

// 创建K线图表
function createKLineChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // 销毁现有图表
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map(d => d.date),
            datasets: [
                {
                    label: '收盘价',
                    data: data.map(d => d.close),
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '股价走势'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// 创建预测对比图表
function createPredictionChart(canvasId, dates, actual, predicted) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // 销毁现有图表
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: '实际价格',
                    data: actual,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: '预测价格',
                    data: predicted,
                    borderColor: '#28a745',
                    backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '模型预测对比'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// 创建回测收益曲线图表
function createBacktestChart(canvasId, data) {
    const ctx = document.getElementById(canvasId).getContext('2d');

    // 销毁现有图表
    if (charts[canvasId]) {
        charts[canvasId].destroy();
    }

    charts[canvasId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.dates,
            datasets: [
                {
                    label: '策略收益',
                    data: data.strategy_returns,
                    borderColor: '#667eea',
                    backgroundColor: 'rgba(102, 126, 234, 0.1)',
                    fill: true,
                    tension: 0.4
                },
                {
                    label: '基准收益',
                    data: data.benchmark_returns,
                    borderColor: '#dc3545',
                    backgroundColor: 'rgba(220, 53, 69, 0.1)',
                    fill: true,
                    tension: 0.4
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                title: {
                    display: true,
                    text: '回测收益曲线'
                },
                tooltip: {
                    mode: 'index',
                    intersect: false
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 10
                    }
                },
                y: {
                    beginAtZero: false
                }
            }
        }
    });
}

// 健康检查
async function checkHealth() {
    try {
        const response = await apiRequest('/api/health');
        console.log('系统状态:', response.data);
        return response.data;
    } catch (error) {
        console.error('健康检查失败:', error);
        return null;
    }
}

// ==================== 全局参数说明悬浮框 ====================
const PARAM_HELP_BY_ID = {
    'stock-code-input': '股票代码输入框：支持 6 位 A 股代码（如 600519）。可输入代码或简称关键字，系统会自动筛选。',
    'stock-code': '股票代码：用于定位查询或预测的目标股票，通常为 6 位数字编码。',
    'days': '获取天数：控制单股页面拉取的历史行情长度，影响K线图与统计区间。',
    'predict-days': '历史K线天数：模型预测时使用的历史窗口长度，窗口越长历史信息越充分。',
    'use-news': '使用新闻：启用后将新闻情感、重要性等特征并入单股预测。',
    'use-relevance': '使用相关性：启用后将关联图谱特征并入预测，增强横截面联动信息。',
    'top-n': '推荐数量（Top N）：按模型预期收益排序，返回前 N 只股票。',
    'kline-days': '历史K线天数：股票推荐中每只股票使用的历史行情窗口。',
    'news-limit': '每股新闻条数上限：每只股票最多关联的新闻条数，影响新闻特征覆盖度。',
    'news-age-hours': '新闻时效窗口（小时）：仅使用最近 N 小时新闻参与股票推荐。',
    'min-price': '最低价格：过滤低于该阈值的股票，减少低价噪声样本。',
    'max-price': '最高价格：过滤高于该阈值的股票，控制候选池价格区间。',
    'batch-use-news': '融合新闻情感与领域影响：启用后推荐时将加入新闻特征。',
    'batch-use-relevance': '融合相关性矩阵：启用后推荐时将加入股票关联图谱特征。',
    'news-source': '新闻源：选择查询的新闻来源渠道（东方财富/新浪/腾讯/全部）。',
    'news-max-age-hours': '新闻时效（小时）：新闻中心仅展示最近 N 小时的新闻记录。',
    'predict-news-age-hours': '新闻时效（小时）：单股预测仅使用最近 N 小时新闻，降低陈旧信息干扰。',
    'feature-stock-code': '新闻特征股票代码：按股票维度查询新闻日级聚合特征。',
    'feature-trade-date': '交易日（可选）：为空则查询最新交易日特征，填写则查询指定日期。',
    'joint-stocks': '股票数量（构建+训练共用）：参与一体化数据构建与训练的股票上限。',
    'joint-days': '历史天数（构建+训练共用）：构建样本与训练模型共同使用的历史窗口。',
    'joint-epochs': '训练轮数：多模态模型训练总轮次（epoch），轮数越高训练时间越长。',
    'joint-horizon': '预测窗口：用于标签构建的未来观察天数（如 5 表示未来 5 个交易日）。',
    'joint-threshold': '标签阈值：未来收益超过该阈值记为上涨标签，决定分类边界。',
    'joint-force-refresh': '强制刷新行情缓存：忽略旧缓存并重新拉取行情数据。',
    'joint-refresh-news': '缓存缺失时抓取新闻：当新闻缓存不足时自动补抓新闻。',
    'baseline-model-type': '基线模型类型：选择结构化基线算法（逻辑回归/随机森林/梯度提升树）。',
    'baseline-topk': '前 N 数量：基线评估时用于 Top-N 指标统计的口径。',
    'baseline-valid-ratio': '验证集比例：用于模型调参与过程验证的数据占比。',
    'baseline-test-ratio': '测试集比例：用于最终离线评估的数据占比。',
    'baseline-dataset-path': '数据集路径（可选）：自定义基线训练输入文件，留空使用默认 model_dataset。',
    'eval-model-type': '离线评估模型类型：选择离线评估阶段使用的模型。',
    'eval-topk': '前 N 数量：离线评估中用于命中率与收益统计的 Top-N 口径。',
    'eval-train-days': '滚动训练天数：每个时间窗口用于训练的历史交易日数量。',
    'eval-valid-days': '滚动验证天数：每个时间窗口用于验证的交易日数量。',
    'eval-test-days': '滚动测试天数：每个时间窗口用于测试的交易日数量。',
    'eval-step-days': '滚动步长：窗口每次向前推进的交易日数量。',
    'eval-rolling-windows': '滚动窗口数：离线评估重复滚动的窗口次数。',
    'eval-dataset-path': '评估数据集路径（可选）：自定义离线评估输入文件路径。',
    'backtest-model-type': '回测模型类型：截面回测时使用的模型算法。',
    'backtest-feature-set': '特征集合：选择仅技术面或全特征（技术+新闻+静态）进行回测。',
    'backtest-topn': '前 N 数量：每个交易日选入组合的股票数量。',
    'backtest-hold-days': '持有天数：策略单次入选股票的持有周期。',
    'backtest-commission': '手续费率：交易手续费比例，会直接从收益中扣减。',
    'backtest-stamp-tax': '印花税率（卖出）：卖出侧税费比例。',
    'backtest-slippage': '滑点率：模拟成交价偏离引起的交易损耗。',
    'backtest-dataset-path': '回测数据集路径（可选）：自定义回测输入文件路径。',
    'sample-replay-stock-code': '样本回放股票代码：查询指定股票在某交易日的样本构成。',
    'sample-replay-trade-date': '样本回放交易日（可选）：留空默认最新，填写则回放指定日期。',
    'sample-replay-news-limit': '新闻回放条数：样本回放中展示新闻明细的最大条数。'
};

const PARAM_HELP_BY_TEXT = {
    '数据集状态': '数据集状态：离线样本是否已成功构建并可用于训练。',
    '处理股票数': '处理股票数：本次离线构建覆盖的股票数量。',
    '训练样本数': '训练样本数：最终可用于模型训练的样本条数。',
    '最近构建时间': '最近构建时间：离线数据集上一次成功生成的时间戳。',
    '基线模型状态': '基线模型状态：结构化基线模型当前是否可用。',
    '最新测试集曲线面积': '最新测试集曲线面积：最近基线报告的测试集 AUC 指标。',
    '股票代码': '股票代码：证券唯一标识，用于查询、训练、预测和回放定位。',
    '股票代码可选': '股票代码（可选）：不填表示全市场范围，填写后只聚焦该股票。',
    '新闻源': '新闻源：决定新闻数据来自哪个渠道。',
    '新闻数量': '新闻数量：控制一次查询返回的新闻条数上限。',
    '模型类型': '模型类型：指定训练或评估时采用的算法类型。',
    '前n数量': '前 N 数量：按模型评分选前 N 只股票进行统计或回测。',
    '验证集比例': '验证集比例：训练过程中用于调参与监控的数据比例。',
    '测试集比例': '测试集比例：训练完成后用于泛化检验的数据比例。',
    '数据集路径可选': '数据集路径（可选）：自定义输入文件路径；留空走系统默认数据集。',
    '滚动训练天数': '滚动训练天数：滚动评估中训练窗口长度。',
    '滚动验证天数': '滚动验证天数：滚动评估中验证窗口长度。',
    '滚动测试天数': '滚动测试天数：滚动评估中测试窗口长度。',
    '滚动步长': '滚动步长：每次滚动向前推进的交易日长度。',
    '滚动窗口数': '滚动窗口数：执行滚动评估的次数。',
    '特征集合': '特征集合：决定回测使用技术面还是全量多模态特征。',
    '持有天数': '持有天数：回测策略中单笔持仓周期。',
    '手续费率': '手续费率：交易手续费成本比例。',
    '印花税率卖出': '印花税率（卖出）：卖出侧税费比例。',
    '滑点率': '滑点率：成交价偏离导致的额外成本比例。',
    '交易日': '交易日：市场开盘并可形成有效行情数据的日期。',
    '入选数量': '入选数量：策略当日选中的股票数量。',
    '平均概率': '平均概率：入选股票预测概率的均值。',
    '原始未来5日收益': '原始未来 5 日收益：不含交易成本的未来收益表现。',
    '交易成本': '交易成本：手续费、印花税和滑点等成本总和。',
    '净收益': '净收益：扣除交易成本后的收益。',
    '换手率': '换手率：组合仓位变动强度，反映交易频率。',
    '平均标签命中率': '平均标签命中率：入选样本上涨标签的命中比例。',
    '持仓股票代码': '持仓股票代码：当日组合包含的股票代码集合。',
    '字段': '字段：样本或特征中的列名。',
    '值': '值：字段对应的实际取值。',
    '发布时间': '发布时间：新闻原始发布时间，用于时序对齐与泄漏校验。',
    '来源': '来源：新闻来源平台。',
    '标题': '标题：新闻标题文本。',
    '情绪': '情绪：新闻情感分值，正值偏利好、负值偏利空。',
    '重要性': '重要性：新闻影响强度评分。',
    '排名': '排名：结果在当前排序规则下的位置。',
    '股票名称': '股票名称：证券简称，便于人工识别。',
    '当前价格': '当前价格：最新可用行情价格。',
    '预测方向': '预测方向：模型对未来窗口上涨/下跌的判断。',
    '新闻条数': '新闻条数：参与该次预测的新闻记录数量。',
    '预期收益': '预期收益：模型估计的未来窗口收益率。',
    '置信度': '置信度：模型对当前预测方向的确定性。',
    '预测价格': '预测价格：根据当前价格和预期收益估算的目标价。',
    '数据源状态': '数据源状态：行情与新闻外部数据源可用性。',
    '可用股票': '可用股票：当前系统可分析股票池规模。',
    '行情更新任务': '行情更新任务：后台全量行情刷新任务执行状态。',
    '新闻同步任务': '新闻同步任务：后台全源新闻同步任务执行状态。',
    '系统状态': '系统状态：服务整体健康状态。'
};

let paramHelpObserver = null;
let paramHelpDebounceTimer = null;
let activeParamHelpTarget = null;

function normalizeParamKey(text) {
    return String(text || '')
        .replace(/\s+/g, '')
        .replace(/[：:（）()【】\[\]\/、，,。.·\-—_]/g, '')
        .toLowerCase();
}

function buildHeuristicHelp(target, fallbackName) {
    const name = String(fallbackName || target.textContent || '').trim();
    if (!name) {
        return '';
    }

    const inputEl = target.tagName === 'LABEL' && target.getAttribute('for')
        ? document.getElementById(target.getAttribute('for'))
        : null;

    const hints = [];
    if (inputEl) {
        const tag = (inputEl.tagName || '').toLowerCase();
        const type = (inputEl.type || '').toLowerCase();
        if (tag === 'select') {
            hints.push('该参数为选项型输入，请根据业务口径选择。');
        } else if (type === 'checkbox') {
            hints.push('该参数为开关项，勾选表示启用，不勾选表示关闭。');
        } else if (type === 'number') {
            const min = inputEl.min !== '' ? inputEl.min : '-';
            const max = inputEl.max !== '' ? inputEl.max : '-';
            hints.push(`建议输入范围：${min} ~ ${max}。`);
            hints.push('该参数会影响任务规模、筛选强度或统计口径。');
        } else if (type === 'date') {
            hints.push('该参数用于指定日期切片；留空通常表示最新日期。');
        } else {
            hints.push('该参数会作为查询、训练或评估任务的输入条件。');
        }
    } else if (target.tagName === 'TH' || target.classList.contains('label')) {
        hints.push('该字段为输出指标名称，用于解释结果含义。');
    }

    if (name.includes('比例') || name.includes('率')) {
        hints.push('该值通常为比例指标，建议结合其他指标综合解读。');
    }
    if (name.includes('天数') || name.includes('窗口')) {
        hints.push('该值控制时间跨度，跨度越大覆盖更广但时效性可能下降。');
    }
    if (name.includes('数量') || name.includes('条数') || name.toLowerCase().includes('top')) {
        hints.push('该值控制样本或输出规模，越大通常耗时越高。');
    }
    if (!hints.length) {
        hints.push('该参数用于控制当前页面任务的输入条件或输出解释。');
    }

    return `${name}：${hints.join(' ')}`;
}

function resolveParamHelpText(target) {
    const targetId = target.id || '';
    if (targetId && PARAM_HELP_BY_ID[targetId]) {
        return PARAM_HELP_BY_ID[targetId];
    }

    if (target.tagName === 'LABEL') {
        const forId = target.getAttribute('for');
        if (forId && PARAM_HELP_BY_ID[forId]) {
            return PARAM_HELP_BY_ID[forId];
        }
    }

    const text = String(target.textContent || '').trim();
    const normalized = normalizeParamKey(text);
    if (normalized && PARAM_HELP_BY_TEXT[normalized]) {
        return PARAM_HELP_BY_TEXT[normalized];
    }
    return buildHeuristicHelp(target, text);
}

function ensureParamHelpTooltip() {
    let tooltip = document.getElementById('param-help-tooltip');
    if (tooltip) {
        return tooltip;
    }

    tooltip = document.createElement('div');
    tooltip.id = 'param-help-tooltip';
    tooltip.className = 'param-help-tooltip';
    tooltip.style.display = 'none';
    document.body.appendChild(tooltip);
    return tooltip;
}

function positionParamHelpTooltip(target, tooltip) {
    if (!target || !tooltip) {
        return;
    }

    const rect = target.getBoundingClientRect();
    const viewportWidth = window.innerWidth || document.documentElement.clientWidth;
    const viewportHeight = window.innerHeight || document.documentElement.clientHeight;
    const offset = 10;

    tooltip.style.left = '0px';
    tooltip.style.top = '0px';
    tooltip.style.maxWidth = `${Math.min(460, viewportWidth - 24)}px`;

    const tipRect = tooltip.getBoundingClientRect();
    let left = rect.left + window.scrollX;
    let top = rect.bottom + window.scrollY + offset;

    if (left + tipRect.width > window.scrollX + viewportWidth - 12) {
        left = window.scrollX + viewportWidth - tipRect.width - 12;
    }
    if (left < window.scrollX + 12) {
        left = window.scrollX + 12;
    }
    if (top + tipRect.height > window.scrollY + viewportHeight - 12) {
        top = rect.top + window.scrollY - tipRect.height - offset;
    }
    if (top < window.scrollY + 12) {
        top = window.scrollY + 12;
    }

    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
}

function showParamHelpTooltip(target) {
    const detail = target.getAttribute('data-param-help') || resolveParamHelpText(target);
    if (!detail) {
        return;
    }

    const tooltip = ensureParamHelpTooltip();
    tooltip.textContent = detail;
    tooltip.style.display = 'block';
    requestAnimationFrame(() => {
        tooltip.classList.add('show');
        positionParamHelpTooltip(target, tooltip);
    });
    activeParamHelpTarget = target;
}

function hideParamHelpTooltip() {
    const tooltip = document.getElementById('param-help-tooltip');
    if (!tooltip) {
        return;
    }
    tooltip.classList.remove('show');
    tooltip.style.display = 'none';
    activeParamHelpTarget = null;
}

function attachParamHelp(target) {
    if (!target || target.dataset.paramHelpBound === '1') {
        return;
    }

    const detail = resolveParamHelpText(target);
    if (!detail) {
        return;
    }

    target.dataset.paramHelpBound = '1';
    target.dataset.paramHelp = detail;
    target.setAttribute('title', detail);
    target.classList.add('param-help-target');
    target.addEventListener('mouseenter', () => showParamHelpTooltip(target));
    target.addEventListener('mouseleave', hideParamHelpTooltip);
    target.addEventListener('focus', () => showParamHelpTooltip(target));
    target.addEventListener('blur', hideParamHelpTooltip);
}

function applyGlobalParamHelp(root = document) {
    const selectors = [
        'label',
        'th',
        '.label',
        '.result-kv-item > span',
        '.baseline-metric-name',
        '.dataset-output-name',
        '.dataset-output-state',
        '.backtest-chart-title',
        '#result-stats > div > div:first-child',
        '#impact-content > div > div > div:first-child',
        '#news-feature-result summary'
    ];

    const nodes = root.querySelectorAll(selectors.join(','));
    nodes.forEach(node => {
        if (!node.textContent || !node.textContent.trim()) {
            return;
        }
        attachParamHelp(node);
    });

    // 二次扫描：为页面中“短文本叶子节点”自动补齐说明（覆盖动态渲染输出标签）
    const leafCandidates = root.querySelectorAll('div,span,strong,small,p,summary');
    leafCandidates.forEach(node => {
        if (!node || node.dataset.paramHelpBound === '1') {
            return;
        }
        if (node.children && node.children.length > 0) {
            return;
        }
        const text = (node.textContent || '').trim();
        if (!text || text.length > 24) {
            return;
        }
        const normalized = normalizeParamKey(text);
        if (!normalized || !PARAM_HELP_BY_TEXT[normalized]) {
            return;
        }
        attachParamHelp(node);
    });
}

function initGlobalParamHelpSystem() {
    applyGlobalParamHelp(document);

    if (!paramHelpObserver) {
        paramHelpObserver = new MutationObserver(() => {
            if (paramHelpDebounceTimer) {
                clearTimeout(paramHelpDebounceTimer);
            }
            paramHelpDebounceTimer = setTimeout(() => {
                applyGlobalParamHelp(document);
                if (activeParamHelpTarget) {
                    const tip = document.getElementById('param-help-tooltip');
                    if (tip && tip.style.display !== 'none') {
                        positionParamHelpTooltip(activeParamHelpTarget, tip);
                    }
                }
            }, 80);
        });
        paramHelpObserver.observe(document.body, { childList: true, subtree: true });
    }

    window.addEventListener('scroll', () => {
        const tip = document.getElementById('param-help-tooltip');
        if (!tip || tip.style.display === 'none' || !activeParamHelpTarget) {
            return;
        }
        positionParamHelpTooltip(activeParamHelpTarget, tip);
    }, true);

    window.addEventListener('resize', () => {
        const tip = document.getElementById('param-help-tooltip');
        if (!tip || tip.style.display === 'none' || !activeParamHelpTarget) {
            return;
        }
        positionParamHelpTooltip(activeParamHelpTarget, tip);
    });
}

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    console.log('页面加载完成');
    initGlobalParamHelpSystem();
    checkHealth();
});
