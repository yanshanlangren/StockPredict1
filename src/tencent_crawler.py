"""
腾讯财经数据爬虫 - 使用AKShare的腾讯财经数据源
"""
import pandas as pd
import time
import logging
from typing import Optional, List
from datetime import datetime, timedelta

# 尝试导入akshare
try:
    import akshare as ak
    AKSHARE_AVAILABLE = True
    logger = logging.getLogger(__name__)
    logger.info("✓ akshare库已加载")
except ImportError:
    AKSHARE_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.error("✗ akshare库未安装，数据获取功能不可用")
    logger.error("请运行: pip install akshare")


class TencentFinanceCrawler:
    """腾讯财经数据爬虫"""

    def __init__(self, delay: float = 0.5):
        """
        初始化爬虫

        Args:
            delay: 请求延迟（秒），避免请求过快
        """
        self.delay = delay
        self.last_request_time = 0

    def _rate_limit(self):
        """实现请求速率限制"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time

        if elapsed < self.delay:
            sleep_time = self.delay - elapsed
            time.sleep(sleep_time)

        self.last_request_time = time.time()

    def get_stock_kline(self, stock_code: str, days: int = 300) -> pd.DataFrame:
        """
        获取股票K线数据

        Args:
            stock_code: 股票代码（格式：sh600000 或 sz000001）
            days: 获取天数

        Returns:
            DataFrame: 包含股票K线数据的DataFrame
        """
        logger.info(f"开始获取股票 {stock_code} K线数据...")

        # 检查akshare是否可用
        if not AKSHARE_AVAILABLE:
            logger.error("akshare库不可用，无法获取数据")
            return pd.DataFrame()

        try:
            self._rate_limit()

            # 计算日期范围
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            start_date_str = start_date.strftime("%Y%m%d")
            end_date_str = end_date.strftime("%Y%m%d")

            # 使用AKShare的腾讯财经数据源
            df = ak.stock_zh_a_daily(
                symbol=stock_code,
                start_date=start_date_str,
                end_date=end_date_str,
                adjust="qfq"  # 前复权
            )

            if df.empty:
                logger.warning(f"股票 {stock_code} 未返回数据")
                return pd.DataFrame()

            # 标准化数据格式
            df = self._standardize_data(df, stock_code)

            logger.info(f"成功获取股票 {stock_code} 数据，共 {len(df)} 天")
            return df

        except Exception as e:
            logger.error(f"获取股票 {stock_code} K线数据时出错: {e}")
            return pd.DataFrame()

    def _standardize_data(self, df: pd.DataFrame, stock_code: str) -> pd.DataFrame:
        """
        标准化数据格式

        Args:
            df: 原始数据
            stock_code: 股票代码

        Returns:
            标准化后的DataFrame
        """
        # 重置索引以处理date列
        df = df.reset_index()

        # 确保date列存在
        if 'date' not in df.columns:
            logger.warning(f"股票 {stock_code} 数据缺少date列")
            return pd.DataFrame()

        # 转换日期格式
        df['date'] = pd.to_datetime(df['date'])

        # 设置日期为索引
        df = df.set_index('date')

        # 重命名列以匹配系统格式
        column_mapping = {
            'open': 'open',
            'high': 'high',
            'low': 'low',
            'close': 'close',
            'volume': 'volume',
        }

        # 只保留需要的列
        available_columns = {k: v for k, v in column_mapping.items() if k in df.columns}

        if not available_columns:
            logger.warning(f"股票 {stock_code} 数据缺少必要的列")
            return pd.DataFrame()

        df = df[list(available_columns.keys())].rename(columns=available_columns)

        # 计算涨跌幅
        if 'close' in df.columns:
            df['change_pct'] = df['close'].pct_change() * 100

        # 按日期排序
        df = df.sort_index()

        return df

    def get_stock_list(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        获取股票列表

        Args:
            limit: 返回数量限制

        Returns:
            DataFrame: 包含股票列表的DataFrame
        """
        logger.info("开始获取股票列表...")

        # 检查akshare是否可用
        if not AKSHARE_AVAILABLE:
            logger.error("akshare库不可用，使用预定义的股票列表")
            return self._get_predefined_stocks(limit)

        try:
            self._rate_limit()

            # 使用AKShare获取A股列表
            # 由于腾讯财经接口限制，我们使用东方财富的列表接口（如果可用）
            # 或者返回预定义的股票列表

            # 尝试使用东方财富获取列表
            try:
                stock_list = ak.stock_info_a_code_name()
                # 筛选A股
                stock_list = stock_list[stock_list['code'].str.match(r'^[0-9]{6}$')]

                # 重命名列
                stock_list.columns = ['code', 'name']

                if limit:
                    stock_list = stock_list.head(limit)

                logger.info(f"成功获取股票列表，共 {len(stock_list)} 只")
                return stock_list

            except Exception as e:
                logger.warning(f"使用东方财富接口获取列表失败: {e}")
                logger.info("使用预定义的股票列表")
                return self._get_predefined_stocks(limit)

        except Exception as e:
            logger.error(f"获取股票列表时出错: {e}")
            return self._get_predefined_stocks(limit)

    def _get_predefined_stocks(self, limit: Optional[int] = None) -> pd.DataFrame:
        """
        获取预定义的股票列表

        Args:
            limit: 返回数量限制

        Returns:
            DataFrame: 预定义股票列表
        """
        # 使用预定义的活跃股票列表
        predefined_stocks = [
            ('600000', '浦发银行'),
            ('600004', '白云机场'),
            ('600006', '东风汽车'),
            ('600007', '中国国贸'),
            ('600008', '首创股份'),
            ('600009', '上海机场'),
            ('600010', '包钢股份'),
            ('600011', '华能国际'),
            ('600015', '华夏银行'),
            ('600016', '民生银行'),
            ('600019', '宝钢股份'),
            ('600028', '中国石化'),
            ('600030', '中信证券'),
            ('600036', '招商银行'),
            ('600048', '保利发展'),
            ('600050', '中国联通'),
            ('600104', '上汽集团'),
            ('600519', '贵州茅台'),
            ('600585', '海螺水泥'),
            ('600690', '海尔智家'),
            ('000001', '平安银行'),
            ('000002', '万科A'),
            ('000725', '京东方A'),
            ('000858', '五粮液'),
        ]

        stock_list = pd.DataFrame(predefined_stocks, columns=['code', 'name'])

        if limit:
            stock_list = stock_list.head(limit)

        logger.info(f"使用预定义股票列表，共 {len(stock_list)} 只")
        return stock_list

    def get_batch_kline(self, stock_codes: List[str], days: int = 300) -> dict:
        """
        批量获取股票K线数据

        Args:
            stock_codes: 股票代码列表
            days: 获取天数

        Returns:
            dict: {股票代码: DataFrame}
        """
        logger.info(f"开始批量获取 {len(stock_codes)} 只股票的数据...")

        results = {}
        success_count = 0
        fail_count = 0

        for i, stock_code in enumerate(stock_codes, 1):
            logger.info(f"[{i}/{len(stock_codes)}] 正在获取 {stock_code}...")

            df = self.get_stock_kline(stock_code, days)

            if not df.empty:
                results[stock_code] = df
                success_count += 1
            else:
                fail_count += 1

        logger.info(f"批量获取完成: 成功 {success_count}, 失败 {fail_count}")

        return results

    def save_to_csv(self, df: pd.DataFrame, stock_code: str, output_dir: str = 'data/raw'):
        """
        保存数据到CSV文件

        Args:
            df: 要保存的数据
            stock_code: 股票代码
            output_dir: 输出目录
        """
        if df.empty:
            logger.warning(f"股票 {stock_code} 数据为空，不保存")
            return

        import os
        os.makedirs(output_dir, exist_ok=True)

        filename = f"tencent_{stock_code}.csv"
        filepath = os.path.join(output_dir, filename)

        df.to_csv(filepath)
        logger.info(f"已保存到: {filepath}")


# 测试代码
if __name__ == "__main__":
    print("腾讯财经爬虫测试")
    print("="*60)

    crawler = TencentFinanceCrawler(delay=0.5)

    # 测试1: 获取单只股票
    print("\n测试1: 获取单只股票数据")
    print("-"*60)
    df = crawler.get_stock_kline('sh600000', days=30)

    if not df.empty:
        print(f"✓ 成功获取 {len(df)} 天数据")
        print(df.head())
        print(f"\n数据形状: {df.shape}")
        print(f"列名: {list(df.columns)}")
    else:
        print("✗ 获取失败")

    # 测试2: 获取股票列表
    print("\n测试2: 获取股票列表")
    print("-"*60)
    stock_list = crawler.get_stock_list(limit=10)

    if not stock_list.empty:
        print(f"✓ 成功获取 {len(stock_list)} 只股票")
        print(stock_list.head())
    else:
        print("✗ 获取失败")

    # 测试3: 批量获取
    print("\n测试3: 批量获取数据")
    print("-"*60)
    test_codes = ['sh600004', 'sh600006', 'sh600007']
    results = crawler.get_batch_kline(test_codes, days=30)

    for code, data in results.items():
        if not data.empty:
            print(f"✓ {code}: {len(data)} 天")
        else:
            print(f"✗ {code}: 失败")
