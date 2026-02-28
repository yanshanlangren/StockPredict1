"""
回测系统模块 - 模拟交易并计算盈利率
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RESULT_DIR, LOG_DIR, INITIAL_CAPITAL, COMMISSION_RATE

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'backtest.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class Backtester:
    """回测系统"""

    def __init__(self, initial_capital: float = INITIAL_CAPITAL, 
                 commission_rate: float = COMMISSION_RATE):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.reset()

    def reset(self):
        """重置回测状态"""
        self.cash = self.initial_capital
        self.position = 0  # 持仓数量
        self.trades = []  # 交易记录
        self.portfolio_value = []  # 组合价值历史
        self.buy_signals = []
        self.sell_signals = []

    def calculate_signals(self, predictions: np.ndarray, actual_prices: np.ndarray, 
                          threshold: float = 0.02) -> List[int]:
        """
        计算买卖信号
        
        Args:
            predictions: 预测价格
            actual_prices: 实际价格
            threshold: 阈值，预测上涨超过threshold时买入
            
        Returns:
            信号列表，1=买入，0=持有，-1=卖出
        """
        signals = []
        
        for i in range(len(predictions)):
            pred_change = (predictions[i] - actual_prices[i]) / actual_prices[i]
            
            if pred_change > threshold:
                signals.append(1)  # 买入
            elif pred_change < -threshold:
                signals.append(-1)  # 卖出
            else:
                signals.append(0)  # 持有
        
        return signals

    def backtest(self, predictions: np.ndarray, actual_prices: np.ndarray, 
                 actual_dates: pd.Series, threshold: float = 0.02) -> Dict:
        """
        执行回测
        
        Args:
            predictions: 预测价格
            actual_prices: 实际价格
            actual_dates: 实际日期
            threshold: 交易阈值
            
        Returns:
            回测结果字典
        """
        self.reset()
        
        signals = self.calculate_signals(predictions, actual_prices, threshold)
        
        logger.info(f"开始回测，共 {len(signals)} 个交易日")
        logger.info(f"初始资金: {self.initial_capital:.2f}")
        
        for i, signal in enumerate(signals):
            current_price = actual_prices[i]
            current_date = actual_dates.iloc[i]
            
            # 记录组合价值
            portfolio_val = self.cash + self.position * current_price
            self.portfolio_value.append({
                'date': current_date,
                'price': current_price,
                'cash': self.cash,
                'position': self.position,
                'portfolio_value': portfolio_val,
                'signal': signal,
                'prediction': predictions[i]
            })
            
            # 执行交易
            if signal == 1 and self.position == 0:  # 买入信号且空仓
                shares = int(self.cash / (current_price * (1 + self.commission_rate)))
                if shares > 0:
                    cost = shares * current_price * (1 + self.commission_rate)
                    self.cash -= cost
                    self.position = shares
                    self.trades.append({
                        'date': current_date,
                        'action': 'BUY',
                        'price': current_price,
                        'shares': shares,
                        'cost': cost
                    })
                    self.buy_signals.append(i)
                    logger.info(f"买入: {current_date}, 价格={current_price:.2f}, 数量={shares}")
            
            elif signal == -1 and self.position > 0:  # 卖出信号且持仓
                revenue = self.position * current_price * (1 - self.commission_rate)
                self.cash += revenue
                self.trades.append({
                    'date': current_date,
                    'action': 'SELL',
                    'price': current_price,
                    'shares': self.position,
                    'revenue': revenue
                })
                self.position = 0
                self.sell_signals.append(i)
                logger.info(f"卖出: {current_date}, 价格={current_price:.2f}, 收入={revenue:.2f}")
        
        # 最后一天强制平仓
        if self.position > 0:
            last_price = actual_prices[-1]
            last_date = actual_dates.iloc[-1]
            revenue = self.position * last_price * (1 - self.commission_rate)
            self.cash += revenue
            self.trades.append({
                'date': last_date,
                'action': 'SELL',
                'price': last_price,
                'shares': self.position,
                'revenue': revenue,
                'forced': True
            })
            logger.info(f"强制平仓: {last_date}, 价格={last_price:.2f}")
        
        # 计算最终收益
        final_value = self.cash
        profit = final_value - self.initial_capital
        profit_rate = profit / self.initial_capital * 100
        
        # 计算基准收益（买入持有）
        first_price = actual_prices[0]
        last_price = actual_prices[-1]
        benchmark_profit_rate = (last_price - first_price) / first_price * 100
        
        # 计算交易统计
        total_trades = len([t for t in self.trades if t['action'] == 'BUY'])
        win_trades = 0
        total_profit = 0
        
        for i in range(0, len(self.trades)-1, 2):
            if i+1 < len(self.trades):
                buy_price = self.trades[i]['price']
                sell_price = self.trades[i+1]['price']
                if sell_price > buy_price:
                    win_trades += 1
                total_profit += (sell_price - buy_price) / buy_price
        
        win_rate = (win_trades / total_trades * 100) if total_trades > 0 else 0
        
        results = {
            'initial_capital': self.initial_capital,
            'final_value': final_value,
            'profit': profit,
            'profit_rate': profit_rate,
            'benchmark_profit_rate': benchmark_profit_rate,
            'excess_return': profit_rate - benchmark_profit_rate,
            'total_trades': total_trades,
            'win_trades': win_trades,
            'win_rate': win_rate,
            'trades': self.trades,
            'portfolio_history': pd.DataFrame(self.portfolio_value),
            'buy_signals': self.buy_signals,
            'sell_signals': self.sell_signals
        }
        
        logger.info(f"回测完成!")
        logger.info(f"最终资金: {final_value:.2f}")
        logger.info(f"总收益: {profit:.2f}")
        logger.info(f"收益率: {profit_rate:.2f}%")
        logger.info(f"基准收益率: {benchmark_profit_rate:.2f}%")
        logger.info(f"超额收益: {profit_rate - benchmark_profit_rate:.2f}%")
        logger.info(f"交易次数: {total_trades}")
        logger.info(f"胜率: {win_rate:.2f}%")
        
        return results

    def save_results(self, results: Dict, model_name: str = 'model'):
        """
        保存回测结果
        
        Args:
            results: 回测结果
            model_name: 模型名称
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 保存交易记录
        trades_df = pd.DataFrame(results['trades'])
        trades_path = os.path.join(RESULT_DIR, f'{model_name}_trades_{timestamp}.csv')
        trades_df.to_csv(trades_path, index=False, encoding='utf-8-sig')
        
        # 保存组合价值历史
        portfolio_path = os.path.join(RESULT_DIR, f'{model_name}_portfolio_{timestamp}.csv')
        results['portfolio_history'].to_csv(portfolio_path, index=False, encoding='utf-8-sig')
        
        # 保存汇总信息
        summary = {
            'model_name': model_name,
            'timestamp': timestamp,
            'initial_capital': results['initial_capital'],
            'final_value': results['final_value'],
            'profit': results['profit'],
            'profit_rate': results['profit_rate'],
            'benchmark_profit_rate': results['benchmark_profit_rate'],
            'excess_return': results['excess_return'],
            'total_trades': results['total_trades'],
            'win_trades': results['win_trades'],
            'win_rate': results['win_rate']
        }
        
        summary_df = pd.DataFrame([summary])
        summary_path = os.path.join(RESULT_DIR, f'{model_name}_summary_{timestamp}.csv')
        summary_df.to_csv(summary_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"回测结果已保存:")
        logger.info(f"  交易记录: {trades_path}")
        logger.info(f"  组合历史: {portfolio_path}")
        logger.info(f"  汇总信息: {summary_path}")
        
        return summary


class ModelComparison:
    """模型比较器"""

    def __init__(self):
        self.results = {}

    def add_result(self, model_name: str, result: Dict):
        """
        添加模型回测结果
        
        Args:
            model_name: 模型名称
            result: 回测结果
        """
        self.results[model_name] = result
        logger.info(f"已添加模型 {model_name} 的回测结果")

    def compare(self) -> pd.DataFrame:
        """
        比较各模型性能
        
        Returns:
            比较结果DataFrame
        """
        comparison_data = []
        
        for model_name, result in self.results.items():
            comparison_data.append({
                'model': model_name,
                'profit_rate': result['profit_rate'],
                'benchmark_profit_rate': result['benchmark_profit_rate'],
                'excess_return': result['excess_return'],
                'total_trades': result['total_trades'],
                'win_rate': result['win_rate'],
                'final_value': result['final_value']
            })
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('profit_rate', ascending=False)
        
        logger.info("\n模型性能比较:")
        logger.info(df.to_string(index=False))
        
        return df

    def get_best_model(self) -> Tuple[str, Dict]:
        """
        获取表现最好的模型
        
        Returns:
            (模型名称, 回测结果)
        """
        best_model = max(self.results.items(), key=lambda x: x[1]['profit_rate'])
        logger.info(f"\n最佳模型: {best_model[0]}")
        logger.info(f"收益率: {best_model[1]['profit_rate']:.2f}%")
        
        return best_model


if __name__ == "__main__":
    # 测试代码
    logger.info("测试回测系统模块...")
    
    # 创建模拟数据
    np.random.seed(42)
    n_days = 20
    
    # 模拟价格（随机游走）
    prices = [100]
    for _ in range(n_days):
        change = np.random.normal(0, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    actual_prices = np.array(prices[1:])
    predictions = actual_prices * (1 + np.random.normal(0, 0.01, n_days))  # 预测略偏离
    dates = pd.date_range(start='2024-01-01', periods=n_days)
    
    # 执行回测
    backtester = Backtester(initial_capital=100000)
    results = backtester.backtest(predictions, actual_prices, dates, threshold=0.015)
    
    # 保存结果
    backtester.save_results(results, 'test_model')
    
    # 模型比较
    comparator = ModelComparison()
    comparator.add_result('LSTM', results)
    
    # 添加第二个模型结果
    results2 = backtester.backtest(
        actual_prices * 1.02,  # 假设另一个模型预测
        actual_prices,
        dates,
        threshold=0.015
    )
    comparator.add_result('GRU', results2)
    
    # 比较模型
    comparison_df = comparator.compare()
    best_model = comparator.get_best_model()
