"""
数据源管理器 - 支持多数据源切换
"""
import pandas as pd
import logging
from typing import Optional, List
from enum import Enum

from crawler import EastMoneyCrawler
from tencent_crawler import TencentFinanceCrawler
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """数据源枚举"""
    TENCENT = "tencent"  # 腾讯财经（推荐）
    EASTMONEY = "eastmoney"  # 东方财富（备用）
    MOCK = "mock"  # 模拟数据（兜底）


class DataSourceManager:
    """数据源管理器 - 自动切换数据源"""

    def __init__(self, preferred_source: DataSource = DataSource.TENCENT):
        """
        初始化数据源管理器

        Args:
            preferred_source: 首选数据源
        """
        self.preferred_source = preferred_source
        self.tencent_crawler = None
        self.eastmoney_crawler = None
        self.mock_data_path = "data/raw/mock_stock_data.csv"

        # 延迟初始化爬虫
        self._init_crawlers()

    def _init_crawlers(self):
        """初始化爬虫实例"""
        try:
            self.tencent_crawler = TencentFinanceCrawler(delay=0.5)
            logger.info("腾讯财经爬虫初始化成功")
        except Exception as e:
            logger.warning(f"腾讯财经爬虫初始化失败: {e}")

        try:
            self.eastmoney_crawler = EastMoneyCrawler()
            logger.info("东方财富爬虫初始化成功")
        except Exception as e:
            logger.warning(f"东方财富爬虫初始化失败: {e}")

    def get_stock_kline(self, stock_code: str, days: int = 300, source: Optional[DataSource] = None) -> pd.DataFrame:
        """
        获取股票K线数据（自动切换数据源）

        Args:
            stock_code: 股票代码
            days: 获取天数
            source: 指定数据源，None表示自动选择

        Returns:
            DataFrame: 股票K线数据
        """
        if source is None:
            source = self.preferred_source

        # 尝试顺序
        sources_to_try = self._get_source_priority(source)

        last_error = None

        for ds in sources_to_try:
            logger.info(f"尝试使用数据源: {ds.value}")

            try:
                if ds == DataSource.TENCENT and self.tencent_crawler:
                    # 转换股票代码格式
                    tencent_code = self._convert_to_tencent_format(stock_code)
                    df = self.tencent_crawler.get_stock_kline(tencent_code, days)
                    if not df.empty:
                        logger.info(f"✓ 使用腾讯财经成功获取数据")
                        return df

                elif ds == DataSource.EASTMONEY and self.eastmoney_crawler:
                    df = self.eastmoney_crawler.get_stock_kline(stock_code, days)
                    if not df.empty:
                        logger.info(f"✓ 使用东方财富成功获取数据")
                        return df

                elif ds == DataSource.MOCK:
                    df = self._get_mock_data(stock_code, days)
                    if not df.empty:
                        logger.info(f"✓ 使用模拟数据成功")
                        return df

            except Exception as e:
                logger.warning(f"数据源 {ds.value} 失败: {e}")
                last_error = e
                continue

        # 所有数据源都失败
        logger.error(f"所有数据源都失败，最后错误: {last_error}")
        return pd.DataFrame()

    def _convert_to_tencent_format(self, stock_code: str) -> str:
        """
        转换股票代码格式为腾讯财经格式

        Args:
            stock_code: 原始代码（如600000）

        Returns:
            腾讯财经格式（如sh600000）
        """
        if stock_code.startswith('6'):
            return f"sh{stock_code}"
        else:
            return f"sz{stock_code}"

    def _get_source_priority(self, preferred: DataSource) -> List[DataSource]:
        """
        获取数据源优先级

        Args:
            preferred: 首选数据源

        Returns:
            按优先级排序的数据源列表
        """
        # 默认优先级：腾讯财经 > 东方财富 > 模拟数据
        all_sources = [DataSource.TENCENT, DataSource.EASTMONEY, DataSource.MOCK]

        # 将首选源移到最前面
        all_sources.remove(preferred)
        all_sources.insert(0, preferred)

        return all_sources

    def _get_mock_data(self, stock_code: str, days: int) -> pd.DataFrame:
        """
        从模拟数据文件中获取数据

        Args:
            stock_code: 股票代码
            days: 获取天数

        Returns:
            DataFrame: 模拟数据
        """
        try:
            import os

            if not os.path.exists(self.mock_data_path):
                logger.warning(f"模拟数据文件不存在: {self.mock_data_path}")
                return pd.DataFrame()

            # 读取模拟数据
            df = pd.read_csv(self.mock_data_path, index_col=0, parse_dates=True)

            # 筛选指定股票
            if 'code' in df.columns:
                stock_df = df[df['code'] == stock_code].copy()
            else:
                logger.warning("模拟数据缺少code列")
                return pd.DataFrame()

            if stock_df.empty:
                logger.warning(f"模拟数据中未找到股票 {stock_code}")
                return pd.DataFrame()

            # 只保留需要的列
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in required_columns if col in stock_df.columns]

            if not available_columns:
                logger.warning("模拟数据缺少必要的列")
                return pd.DataFrame()

            stock_df = stock_df[available_columns].copy()

            # 限制天数
            if len(stock_df) > days:
                stock_df = stock_df.tail(days)

            return stock_df

        except Exception as e:
            logger.error(f"读取模拟数据失败: {e}")
            return pd.DataFrame()

    def get_stock_list(self, limit: Optional[int] = None, source: Optional[DataSource] = None) -> pd.DataFrame:
        """
        获取股票列表（自动切换数据源）

        Args:
            limit: 返回数量限制
            source: 指定数据源，None表示自动选择

        Returns:
            DataFrame: 股票列表
        """
        if source is None:
            source = self.preferred_source

        # 尝试顺序
        sources_to_try = self._get_source_priority(source)

        for ds in sources_to_try:
            logger.info(f"尝试使用数据源获取股票列表: {ds.value}")

            try:
                if ds == DataSource.TENCENT and self.tencent_crawler:
                    df = self.tencent_crawler.get_stock_list(limit=limit)
                    if not df.empty:
                        logger.info(f"✓ 使用腾讯财经成功获取股票列表")
                        return df

                elif ds == DataSource.EASTMONEY and self.eastmoney_crawler:
                    df = self.eastmoney_crawler.get_stock_list()
                    if not df.empty:
                        if limit:
                            df = df.head(limit)
                        logger.info(f"✓ 使用东方财富成功获取股票列表")
                        return df

                elif ds == DataSource.MOCK:
                    # 从模拟数据中提取股票列表
                    df = self._get_mock_stock_list(limit)
                    if not df.empty:
                        logger.info(f"✓ 使用模拟数据成功获取股票列表")
                        return df

            except Exception as e:
                logger.warning(f"数据源 {ds.value} 获取股票列表失败: {e}")
                continue

        logger.error("所有数据源都失败")
        return pd.DataFrame()

    def _get_mock_stock_list(self, limit: Optional[int] = None) -> pd.DataFrame:
        """从模拟数据中提取股票列表"""
        try:
            import os

            if not os.path.exists(self.mock_data_path):
                return pd.DataFrame()

            df = pd.read_csv(self.mock_data_path, index_col=0)

            if 'code' in df.columns and 'name' in df.columns:
                # 提取唯一股票代码
                stock_list = df[['code', 'name']].drop_duplicates()

                if limit:
                    stock_list = stock_list.head(limit)

                return stock_list
            else:
                return pd.DataFrame()

        except Exception as e:
            logger.error(f"提取模拟股票列表失败: {e}")
            return pd.DataFrame()

    def get_batch_kline(self, stock_codes: List[str], days: int = 300, source: Optional[DataSource] = None) -> dict:
        """
        批量获取股票K线数据

        Args:
            stock_codes: 股票代码列表
            days: 获取天数
            source: 指定数据源，None表示自动选择

        Returns:
            dict: {股票代码: DataFrame}
        """
        logger.info(f"开始批量获取 {len(stock_codes)} 只股票的数据...")

        results = {}
        success_count = 0
        fail_count = 0

        for stock_code in stock_codes:
            df = self.get_stock_kline(stock_code, days, source)

            if not df.empty:
                results[stock_code] = df
                success_count += 1
            else:
                fail_count += 1

        logger.info(f"批量获取完成: 成功 {success_count}, 失败 {fail_count}")

        return results

    def set_preferred_source(self, source: DataSource):
        """设置首选数据源"""
        self.preferred_source = source
        logger.info(f"首选数据源已设置为: {source.value}")


# 测试代码
if __name__ == "__main__":
    print("数据源管理器测试")
    print("="*60)

    manager = DataSourceManager(preferred_source=DataSource.TENCENT)

    # 测试1: 获取单只股票
    print("\n测试1: 获取单只股票数据")
    print("-"*60)
    df = manager.get_stock_kline('600000', days=30)

    if not df.empty:
        print(f"✓ 成功获取 {len(df)} 天数据")
        print(f"  数据形状: {df.shape}")
        print(f"  前5行:")
        print(df.head())
    else:
        print("✗ 获取失败")

    # 测试2: 获取股票列表
    print("\n测试2: 获取股票列表")
    print("-"*60)
    stock_list = manager.get_stock_list(limit=10)

    if not stock_list.empty:
        print(f"✓ 成功获取 {len(stock_list)} 只股票")
        print(stock_list.head())
    else:
        print("✗ 获取失败")

    # 测试3: 批量获取
    print("\n测试3: 批量获取数据")
    print("-"*60)
    test_codes = ['600004', '600006', '600007']
    results = manager.get_batch_kline(test_codes, days=30)

    for code, data in results.items():
        if not data.empty:
            print(f"✓ {code}: {len(data)} 天")
        else:
            print(f"✗ {code}: 失败")

    print("\n" + "="*60)
    print("测试完成")
