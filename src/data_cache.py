"""
本地数据缓存管理器 - 优先使用本地CSV文件
"""
import pandas as pd
import os
from datetime import datetime, timedelta
import logging
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCache:
    """本地数据缓存管理器"""

    def __init__(self, cache_dir: str = "data/cache"):
        """
        初始化缓存管理器

        Args:
            cache_dir: 缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def get_cached_data(self, stock_code: str) -> Optional[pd.DataFrame]:
        """
        从缓存获取数据

        Args:
            stock_code: 股票代码

        Returns:
            DataFrame或None
        """
        cache_file = self._get_cache_file(stock_code)

        if not os.path.exists(cache_file):
            logger.debug(f"缓存文件不存在: {cache_file}")
            return None

        try:
            # 读取缓存数据
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)

            if df.empty:
                logger.debug(f"缓存文件为空: {cache_file}")
                return None

            # 检查数据是否过期（超过7天）
            latest_date = df.index[-1]
            days_old = (datetime.now() - latest_date).days

            if days_old > 7:
                logger.info(f"缓存数据已过期（{days_old}天）: {stock_code}")
                return None

            logger.info(f"从缓存加载股票 {stock_code} 数据，共 {len(df)} 天")
            return df

        except Exception as e:
            logger.error(f"读取缓存失败 {stock_code}: {e}")
            return None

    def save_to_cache(self, stock_code: str, df: pd.DataFrame) -> bool:
        """
        保存数据到缓存

        Args:
            stock_code: 股票代码
            df: 要保存的数据

        Returns:
            是否成功
        """
        if df.empty:
            logger.warning(f"数据为空，不保存缓存: {stock_code}")
            return False

        cache_file = self._get_cache_file(stock_code)

        try:
            # 保存到CSV
            df.to_csv(cache_file)
            logger.info(f"已缓存股票 {stock_code} 数据，共 {len(df)} 天")
            return True
        except Exception as e:
            logger.error(f"保存缓存失败 {stock_code}: {e}")
            return False

    def _get_cache_file(self, stock_code: str) -> str:
        """
        获取缓存文件路径

        Args:
            stock_code: 股票代码

        Returns:
            文件路径
        """
        return os.path.join(self.cache_dir, f"{stock_code}.csv")

    def has_today_data(self, stock_code: str) -> bool:
        """
        检查缓存是否包含今天的数据

        Args:
            stock_code: 股票代码

        Returns:
            是否包含今天数据
        """
        df = self.get_cached_data(stock_code)

        if df is None:
            return False

        latest_date = df.index[-1]
        today = datetime.now().date()

        # 检查最新数据是否是今天或昨天（考虑周末）
        days_diff = (today - latest_date.date()).days

        return days_diff <= 1

    def clear_cache(self, stock_code: Optional[str] = None):
        """
        清除缓存

        Args:
            stock_code: 股票代码，None表示清除所有
        """
        if stock_code:
            cache_file = self._get_cache_file(stock_code)
            if os.path.exists(cache_file):
                os.remove(cache_file)
                logger.info(f"已清除缓存: {stock_code}")
        else:
            # 清除所有缓存
            for filename in os.listdir(self.cache_dir):
                if filename.endswith('.csv'):
                    file_path = os.path.join(self.cache_dir, filename)
                    os.remove(file_path)
                    logger.info(f"已清除缓存: {filename}")

    def get_cache_info(self) -> dict:
        """
        获取缓存信息

        Returns:
            缓存信息字典
        """
        cache_files = [f for f in os.listdir(self.cache_dir) if f.endswith('.csv')]

        total_files = len(cache_files)
        total_size = sum(
            os.path.getsize(os.path.join(self.cache_dir, f))
            for f in cache_files
        )

        info = {
            'total_files': total_files,
            'total_size': total_size,
            'cache_dir': self.cache_dir
        }

        # 获取每个文件的详细信息
        file_details = []
        for filename in cache_files[:10]:  # 只返回前10个
            file_path = os.path.join(self.cache_dir, filename)
            stock_code = filename.replace('.csv', '')
            file_size = os.path.getsize(file_path)

            try:
                df = pd.read_csv(file_path, index_col=0, parse_dates=True)
                latest_date = df.index[-1] if not df.empty else None
                data_count = len(df)
            except:
                latest_date = None
                data_count = 0

            file_details.append({
                'stock_code': stock_code,
                'size': file_size,
                'data_count': data_count,
                'latest_date': latest_date.strftime('%Y-%m-%d') if latest_date else None
            })

        info['files'] = file_details

        return info


# 全局缓存实例
cache = DataCache()


if __name__ == "__main__":
    print("测试数据缓存管理器")
    print("="*60)

    # 创建测试数据
    test_data = pd.DataFrame({
        'open': [10.0, 10.1, 10.2],
        'high': [10.5, 10.6, 10.7],
        'low': [9.8, 9.9, 10.0],
        'close': [10.2, 10.3, 10.4],
        'volume': [1000000, 1100000, 1200000]
    }, index=pd.date_range('2026-02-01', periods=3))

    # 保存到缓存
    print("\n1. 保存测试数据到缓存")
    cache.save_to_cache('600000', test_data)

    # 从缓存读取
    print("\n2. 从缓存读取数据")
    cached_data = cache.get_cached_data('600000')
    if cached_data is not None:
        print(f"✓ 成功读取缓存数据，共 {len(cached_data)} 天")
        print(cached_data)

    # 检查是否包含今天数据
    print("\n3. 检查是否包含今天数据")
    has_today = cache.has_today_data('600000')
    print(f"包含今天数据: {has_today}")

    # 获取缓存信息
    print("\n4. 获取缓存信息")
    info = cache.get_cache_info()
    print(f"总文件数: {info['total_files']}")
    print(f"总大小: {info['total_size']} 字节")
    print(f"文件详情:")
    for file_info in info['files']:
        print(f"  {file_info['stock_code']}: {file_info['data_count']} 天, {file_info['latest_date']}")

    print("\n" + "="*60)
    print("测试完成")
