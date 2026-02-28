"""
数据爬取模块 - 从东方财富网爬取A股数据
"""
import requests
import pandas as pd
import time
from datetime import datetime, timedelta
from typing import List, Dict
import logging
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import RAW_DATA_DIR, LOG_DIR

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, 'crawler.log'), encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def retry_request(max_retries=3, delay=2):
    """
    请求重试装饰器
    
    Args:
        max_retries: 最大重试次数
        delay: 重试延迟（秒）
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except (requests.exceptions.ConnectionError, 
                       requests.exceptions.Timeout,
                       requests.exceptions.RemoteDisconnected) as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"请求失败，第 {attempt + 1}/{max_retries} 次重试... 错误: {e}")
                        time.sleep(delay * (attempt + 1))
                    else:
                        logger.error(f"请求失败，已达到最大重试次数 {max_retries}")
                        raise
            return None
        return wrapper
    return decorator


class EastMoneyCrawler:
    """东方财富数据爬虫"""

    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Referer': 'https://quote.eastmoney.com/center/gridlist.html#hs_a_board'
        }
        self.stock_list_url = "http://82.push2.eastmoney.com/api/qt/clist/get"
        self.kline_url = "http://push2his.eastmoney.com/api/qt/stock/kline/get"

    @retry_request(max_retries=3, delay=2)
    def get_stock_list(self, market='0,1') -> pd.DataFrame:
        """
        获取A股股票列表
        
        Args:
            market: 市场类型，0=沪A，1=深A，默认全部
            
        Returns:
            股票列表DataFrame
        """
        logger.info("开始获取股票列表...")

        params = {
            'pn': 1,
            'pz': 5000,  # 获取5000只股票，足够覆盖所有A股
            'po': 1,
            'np': 1,
            'ut': 'bd1d9ddb04089700cf9c27f6f7426281',
            'fltt': 2,
            'invt': 2,
            'fid': 'f3',
            'fs': f'm:{market}+t:6',  # 6表示A股
            'fields': 'f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f12,f13,f14,f15,f16,f17,f18,f20,f21,f23,f24,f25,f22,f11,f62,f128,f136,f115,f152',
        }

        try:
            response = requests.get(self.stock_list_url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get('data') and data['data'].get('diff'):
                stocks = data['data']['diff']
                df = pd.DataFrame(stocks)
                
                # 重命名列
                column_mapping = {
                    'f12': 'code',
                    'f14': 'name',
                    'f2': 'current_price',
                    'f3': 'change_percent',
                    'f5': 'volume',
                    'f6': 'turnover',
                    'f15': 'high',
                    'f16': 'low',
                    'f17': 'open',
                    'f18': 'close',
                    'f20': 'market_cap',
                }
                
                df = df.rename(columns=column_mapping)
                df = df[list(column_mapping.values())]
                
                logger.info(f"成功获取 {len(df)} 只股票")
                return df
            else:
                logger.warning("未获取到股票数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"获取股票列表失败: {e}")
            return pd.DataFrame()

    @retry_request(max_retries=3, delay=2)
    def get_stock_kline(self, stock_code: str, klt: int = 101, start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """
        获取个股K线数据
        
        Args:
            stock_code: 股票代码（如：600000）
            klt: K线类型，101=日K，102=周K，103=月K
            start_date: 开始日期，格式 YYYY-MM-DD
            end_date: 结束日期，格式 YYYY-MM-DD
            
        Returns:
            K线数据DataFrame
        """
        if not start_date:
            start_date = (datetime.now() - timedelta(days=365*3)).strftime('%Y%m%d')  # 默认3年数据
        
        if not end_date:
            end_date = datetime.now().strftime('%Y%m%d')

        # 判断市场
        market = '1' if stock_code.startswith('6') else '0'
        secid = f"{market}.{stock_code}"

        params = {
            'secid': secid,
            'ut': 'fa5fd1943c7b386f172d6893dbfba10b',
            'fields1': 'f1,f2,f3,f4,f5,f6',
            'fields2': 'f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61',
            'klt': klt,
            'fqt': '1',  # 1=前复权
            'beg': start_date.replace('-', ''),
            'end': end_date.replace('-', ''),
            '_': str(int(time.time() * 1000))
        }

        try:
            response = requests.get(self.kline_url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get('data') and data['data'].get('klines'):
                klines = data['data']['klines']
                
                # 解析K线数据
                df_data = []
                for line in klines:
                    items = line.split(',')
                    df_data.append({
                        'date': items[0],
                        'open': float(items[1]),
                        'close': float(items[2]),
                        'high': float(items[3]),
                        'low': float(items[4]),
                        'volume': float(items[5]),
                        'turnover': float(items[6]),
                        'amplitude': float(items[7]),
                        'change_percent': float(items[8]),
                        'change_amount': float(items[9]),
                        'turnover_rate': float(items[10]),
                    })
                
                df = pd.DataFrame(df_data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.sort_values('date').reset_index(drop=True)
                
                return df
            else:
                logger.warning(f"未获取到股票 {stock_code} 的K线数据")
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"获取股票 {stock_code} K线数据失败: {e}")
            return pd.DataFrame()

    def batch_download_stocks(self, stock_list: List[str], max_stocks: int = 50) -> Dict[str, pd.DataFrame]:
        """
        批量下载多只股票数据
        
        Args:
            stock_list: 股票代码列表
            max_stocks: 最大下载数量
            
        Returns:
            股票数据字典 {stock_code: DataFrame}
        """
        logger.info(f"开始批量下载 {len(stock_list)} 只股票的数据...")
        
        stocks_data = {}
        failed_stocks = []
        
        for i, stock_code in enumerate(stock_list[:max_stocks]):
            try:
                logger.info(f"[{i+1}/{min(len(stock_list), max_stocks)}] 下载 {stock_code}...")
                df = self.get_stock_kline(stock_code)
                
                if not df.empty and len(df) > 100:  # 至少需要100天数据
                    stocks_data[stock_code] = df
                    logger.info(f"  成功获取 {len(df)} 天数据")
                else:
                    failed_stocks.append(stock_code)
                    logger.warning(f"  数据不足，跳过")
                
                time.sleep(1.5)  # 增加延迟，避免请求过快被封禁
                
            except Exception as e:
                failed_stocks.append(stock_code)
                logger.error(f"  下载失败: {e}")
        
        logger.info(f"批量下载完成！成功: {len(stocks_data)}, 失败: {len(failed_stocks)}")
        
        return stocks_data

    def save_data(self, stocks_data: Dict[str, pd.DataFrame], filename: str = 'stock_data.csv'):
        """
        保存数据到文件
        
        Args:
            stocks_data: 股票数据字典
            filename: 文件名
        """
        filepath = os.path.join(RAW_DATA_DIR, filename)
        
        # 合并所有股票数据
        all_data = []
        for stock_code, df in stocks_data.items():
            df_copy = df.copy()
            df_copy['stock_code'] = stock_code
            all_data.append(df_copy)
        
        if all_data:
            merged_df = pd.concat(all_data, ignore_index=True)
            merged_df.to_csv(filepath, index=False, encoding='utf-8-sig')
            logger.info(f"数据已保存到 {filepath}")
        else:
            logger.warning("没有数据可保存")


if __name__ == "__main__":
    # 测试代码
    crawler = EastMoneyCrawler()
    
    # 获取股票列表
    stock_df = crawler.get_stock_list()
    print(f"\n股票列表（前10只）:")
    print(stock_df.head(10))
    
    if not stock_df.empty:
        # 获取单只股票数据
        test_code = stock_df.iloc[0]['code']
        print(f"\n测试下载 {test_code} 的K线数据...")
        kline_df = crawler.get_stock_kline(test_code)
        print(f"\n{test_code} K线数据（前5天）:")
        print(kline_df.head())
        
        # 批量下载
        stock_codes = stock_df['code'].head(10).tolist()
        stocks_data = crawler.batch_download_stocks(stock_codes)
        crawler.save_data(stocks_data, 'test_stock_data.csv')
