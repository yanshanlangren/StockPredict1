"""
数据源管理器 - 支持多数据源切换，优先使用本地缓存
"""
import pandas as pd
import logging
from typing import Optional, List
from enum import Enum

from tencent_crawler import TencentFinanceCrawler
from data_cache import cache
import sys
import os

# 添加src目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataSource(Enum):
    """数据源枚举"""
    CACHE = "cache"      # 本地缓存（优先）
    TENCENT = "tencent"  # 腾讯财经


class DataSourceManager:
    """数据源管理器 - 优先使用本地缓存，其次API"""

    def __init__(self, preferred_source: DataSource = DataSource.CACHE):
        """
        初始化数据源管理器

        Args:
            preferred_source: 首选数据源
        """
        self.preferred_source = preferred_source
        self.tencent_crawler = None

        # 延迟初始化爬虫
        self._init_crawlers()

    def _init_crawlers(self):
        """初始化爬虫实例"""
        try:
            self.tencent_crawler = TencentFinanceCrawler(delay=0.5)
            logger.info("腾讯财经爬虫初始化成功")
        except Exception as e:
            logger.warning(f"腾讯财经爬虫初始化失败: {e}")

        except Exception as e:
            logger.warning(f"腾讯财经爬虫初始化失败: {e}")

    def get_stock_kline(self, stock_code: str, days: int = 300, source: Optional[DataSource] = None, force_refresh: bool = False) -> pd.DataFrame:
        """
        获取股票K线数据（优先使用本地缓存）

        Args:
            stock_code: 股票代码
            days: 获取天数
            source: 指定数据源，None表示自动选择
            force_refresh: 强制刷新（不使用缓存）

        Returns:
            DataFrame: 股票K线数据
        """
        # 1. 优先检查本地缓存
        if not force_refresh:
            cached_df = cache.get_cached_data(stock_code)
            if cached_df is not None and len(cached_df) >= days * 0.5:  # 至少需要一半的数据
                logger.info(f"✓ 从本地缓存加载股票 {stock_code} 数据，共 {len(cached_df)} 天")
                return cached_df.tail(days)  # 只返回需要的天数

        # 2. 缓存无效或强制刷新，使用API获取
        if source is None:
            source = self.preferred_source

        # 尝试顺序：缓存（已跳过）-> 腾讯财经
        sources_to_try = self._get_source_priority(source)

        # 跳过CACHE，因为已经检查过了
        sources_to_try = [ds for ds in sources_to_try if ds != DataSource.CACHE]

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
                        # 保存到本地缓存
                        cache.save_to_cache(stock_code, df)
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
        # 默认优先级：缓存 > 腾讯财经
        all_sources = [DataSource.CACHE, DataSource.TENCENT]

        # 将首选源移到最前面
        if preferred in all_sources:
            all_sources.remove(preferred)
        all_sources.insert(0, preferred)

        return all_sources

    def get_stock_list(self, limit: Optional[int] = None, source: Optional[DataSource] = None, force_refresh: bool = False) -> pd.DataFrame:
        """
        获取股票列表（自动切换数据源，支持缓存）

        Args:
            limit: 返回数量限制
            source: 指定数据源，None表示自动选择
            force_refresh: 强制刷新，不使用缓存

        Returns:
            DataFrame: 股票列表
        """
        # 1. 尝试从缓存读取（如果不强制刷新）
        if not force_refresh:
            cached_list = cache.get_cached_stock_list()
            if cached_list is not None and not cached_list.empty:
                logger.info(f"✓ 从缓存加载股票列表，共 {len(cached_list)} 只股票")
                if limit:
                    cached_list = cached_list.head(limit)
                return cached_list

        # 2. 缓存未命中或强制刷新，从数据源获取
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
                        # 保存到缓存
                        cache.save_stock_list_cache(df)
                        return df

            except Exception as e:
                logger.warning(f"数据源 {ds.value} 获取股票列表失败: {e}")
                continue

        logger.error("所有数据源都失败")
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
