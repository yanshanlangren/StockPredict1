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

// 页面加载完成后执行
document.addEventListener('DOMContentLoaded', function() {
    console.log('页面加载完成');
    checkHealth();
});
