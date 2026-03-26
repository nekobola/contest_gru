"""
数据获取与特征工程模块
使用akshare获取沪深300成分股数据，构造基础特征和标签
"""

import os
import warnings
from datetime import datetime
from pathlib import Path

import akshare as ak
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')


class DataIngestion:
    """数据获取与特征工程类"""

    def __init__(self, start_date: str = '20230101', end_date: str = None):
        """
        初始化数据获取器

        Args:
            start_date: 起始日期，格式YYYYMMDD
            end_date: 结束日期，格式YYYYMMDD，默认为今天
        """
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y%m%d')
        self.raw_data = None
        self.df_features = None

    def get_hs300_members(self) -> list:
        """
        获取沪深300成分股列表

        Returns:
            list: 沪深300成分股代码列表
        """
        hs300_df = ak.index_stock_cons_weight_csindex(symbol="000300")
        tickers = hs300_df['成分券代码'].tolist()
        return tickers

    def fetch_stock_data(self, ticker: str) -> pd.DataFrame:
        """
        获取单只股票的日频前复权OHLCV数据

        Args:
            ticker: 股票代码

        Returns:
            pd.DataFrame: 包含OHLCV数据的DataFrame
        """
        try:
            df = ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date=self.start_date,
                end_date=self.end_date,
                adjust="qfq"  # 前复权
            )
            if df is None or df.empty:
                return pd.DataFrame()

            df['ticker'] = ticker
            df.rename(columns={
                '日期': 'date',
                '开盘': 'open',
                '收盘': 'close',
                '最高': 'high',
                '最低': 'low',
                '成交量': 'volume',
                '成交额': 'amount'
            }, inplace=True)

            df['date'] = pd.to_datetime(df['date'])
            df = df[['date', 'ticker', 'open', 'close', 'high', 'low', 'volume', 'amount']]
            return df
        except Exception as e:
            print(f"获取{ticker}数据失败: {e}")
            return pd.DataFrame()

    def fetch_all_data(self, tickers: list) -> pd.DataFrame:
        """
        批量获取所有股票数据

        Args:
            tickers: 股票代码列表

        Returns:
            pd.DataFrame: 合并后的数据
        """
        all_data = []
        for i, ticker in enumerate(tickers):
            df = self.fetch_stock_data(ticker)
            if not df.empty:
                all_data.append(df)
            if (i + 1) % 50 == 0:
                print(f"已获取 {i + 1}/{len(tickers)} 只股票数据")

        if not all_data:
            raise ValueError("未能获取任何股票数据")

        combined_df = pd.concat(all_data, ignore_index=True)
        combined_df.sort_values(['ticker', 'date'], inplace=True)
        combined_df.reset_index(drop=True, inplace=True)
        return combined_df

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算特征指标

        Args:
            df: 原始OHLCV数据

        Returns:
            pd.DataFrame: 包含特征的DataFrame
        """
        df = df.copy()
        df = df.sort_values(['ticker', 'date'])
        df.reset_index(drop=True, inplace=True)

        grouped = df.groupby('ticker')

        # 1. MA5_Gap: 收盘价与5日均线偏离度 = Close / MA5 - 1
        df['MA5'] = grouped['close'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['MA5_Gap'] = df['close'] / df['MA5'] - 1

        # 2. Mom_20: 20日动量 = Close / Close.shift(20) - 1
        df['close_shift20'] = grouped['close'].shift(20)
        df['Mom_20'] = df['close'] / df['close_shift20'] - 1

        # 3. Vol_20: 20日年化波动率 = std(returns, 20) * sqrt(252)
        df['returns'] = grouped['close'].pct_change()
        df['Vol_20'] = grouped['returns'].transform(
            lambda x: x.rolling(20, min_periods=5).std() * np.sqrt(252)
        )

        # 4. Turnover: 换手率近似 = Volume / MA_Volume_gap
        df['volume_MA5'] = grouped['volume'].transform(lambda x: x.rolling(5, min_periods=1).mean())
        df['Turnover'] = df['volume'] / df['volume_MA5'] - 1

        # 5. Corr_PV: 10日价量相关系数
        def rolling_corr(group):
            return group['close'].rolling(10, min_periods=5).corr(group['volume'])

        df['Corr_PV'] = grouped.apply(rolling_corr).reset_index(level=0, drop=True)

        # 6. Price_Range: (High - Low) / Close - 日内波动率
        df['Price_Range'] = (df['high'] - df['low']) / df['close']

        # 7. Close_Open: Close / Open - 1 - 日内收益率
        df['Close_Open'] = df['close'] / df['open'] - 1

        # 8. Volume_MA10: 成交量10日均线偏离度 Volume / MA10_Volume - 1
        df['volume_MA10'] = grouped['volume'].transform(lambda x: x.rolling(10, min_periods=1).mean())
        df['Volume_MA10'] = df['volume'] / df['volume_MA10'] - 1

        # 9. Price_MA10: 收盘价10日均线偏离度 Close / MA10 - 1
        df['price_MA10'] = grouped['close'].transform(lambda x: x.rolling(10, min_periods=1).mean())
        df['Price_MA10'] = df['close'] / df['price_MA10'] - 1

        # 10. RSI_14: 14日RSI指标
        df['delta'] = grouped['close'].diff()
        df['gain'] = df['delta'].clip(lower=0)
        df['loss'] = (-df['delta']).clip(lower=0)
        df['avg_gain'] = grouped['gain'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        df['avg_loss'] = grouped['loss'].transform(lambda x: x.rolling(14, min_periods=1).mean())
        df['rs'] = df['avg_gain'] / df['avg_loss']
        df['RSI_14'] = 100 - (100 / (1 + df['rs']))
        df['RSI_14'] = df['RSI_14'].fillna(50)  # 处理初始缺失值，设为中性值50

        return df

    def calculate_label(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算目标标签: 未来5日累计收益率 (T+1开盘到T+6开盘)
        formula: Open.shift(-5) / Open.shift(-1) - 1

        Args:
            df: 包含特征的DataFrame

        Returns:
            pd.DataFrame: 包含标签的DataFrame
        """
        df = df.copy()
        grouped = df.groupby('ticker')

        # T+6开盘 / T+1开盘 - 1
        df['open_shift_neg1'] = grouped['open'].shift(-1)
        df['open_shift_neg5'] = grouped['open'].shift(-5)
        df['target_return'] = df['open_shift_neg5'] / df['open_shift_neg1'] - 1

        return df

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        处理缺失值: ffill填充，无法填充的设为0

        Args:
            df: 包含缺失值的DataFrame

        Returns:
            pd.DataFrame: 处理后的DataFrame
        """
        feature_cols = ['MA5_Gap', 'Mom_20', 'Vol_20', 'Turnover', 'Corr_PV',
                        'Price_Range', 'Close_Open', 'Volume_MA10', 'Price_MA10', 'RSI_14']

        # 按股票分组进行ffill
        for col in feature_cols:
            df[col] = df.groupby('ticker')[col].transform(lambda x: x.ffill())
            # 无法填充的设为0
            df[col] = df[col].fillna(0)

        return df

    def run(self) -> pd.DataFrame:
        """
        执行完整的数据获取与特征工程流程

        Returns:
            pd.DataFrame: 最终的特征DataFrame
        """
        print("=" * 60)
        print("开始数据获取与特征工程流程")
        print("=" * 60)

        # 1. 获取沪深300成分股
        print("\n[1/5] 获取沪深300成分股列表...")
        tickers = self.get_hs300_members()
        print(f"共获取 {len(tickers)} 只成分股")

        # 2. 获取原始数据
        print(f"\n[2/5] 获取日频数据 (从 {self.start_date} 到 {self.end_date})...")
        raw_df = self.fetch_all_data(tickers)
        print(f"原始数据行数: {len(raw_df)}")

        # 3. 计算特征
        print("\n[3/5] 计算基础特征...")
        df_with_features = self.calculate_features(raw_df)
        print("特征计算完成: MA5_Gap, Mom_20, Vol_20, Turnover, Corr_PV, Price_Range, Close_Open, Volume_MA10, Price_MA10, RSI_14")

        # 4. 计算标签
        print("\n[4/5] 计算目标收益率标签...")
        df_with_label = self.calculate_label(df_with_features)
        print("标签计算完成: target_return")

        # 5. 处理缺失值
        print("\n[5/5] 处理缺失值...")
        df_clean = self.handle_missing_values(df_with_label)
        print("缺失值处理完成 (ffill + fillna(0))")

        # 整理输出格式
        output_cols = ['date', 'ticker', 'MA5_Gap', 'Mom_20', 'Vol_20',
                       'Turnover', 'Corr_PV', 'Price_Range', 'Close_Open',
                       'Volume_MA10', 'Price_MA10', 'RSI_14', 'target_return']
        self.df_features = df_clean[output_cols].sort_values(['date', 'ticker']).reset_index(drop=True)

        # 删除标签为空的行（因shift(-5)导致的尾部缺失）
        self.df_features = self.df_features[self.df_features['target_return'].notna()].reset_index(drop=True)

        print("\n" + "=" * 60)
        print("流程完成!")
        print(f"最终数据形状: {self.df_features.shape}")
        print(f"数据时间范围: {self.df_features['date'].min()} 至 {self.df_features['date'].max()}")
        print("=" * 60)

        return self.df_features

    def get_summary(self) -> dict:
        """
        获取数据统计摘要

        Returns:
            dict: 统计信息字典
        """
        if self.df_features is None:
            return {}

        summary = {
            'total_rows': len(self.df_features),
            'unique_tickers': self.df_features['ticker'].nunique(),
            'date_range': {
                'start': self.df_features['date'].min().strftime('%Y-%m-%d'),
                'end': self.df_features['date'].max().strftime('%Y-%m-%d')
            },
            'feature_stats': self.df_features[['MA5_Gap', 'Mom_20', 'Vol_20',
                                                'Turnover', 'Corr_PV', 'target_return']].describe().to_dict()
        }
        return summary


def main():
    """
    主函数：执行数据获取与特征工程
    生成 df_features DataFrame
    """
    ingestion = DataIngestion(
        start_date='20230101',
        end_date=datetime.now().strftime('%Y%m%d')
    )

    # 执行完整流程
    df_features = ingestion.run()

    # 打印摘要
    summary = ingestion.get_summary()
    print("\n数据摘要:")
    print(f"  总行数: {summary['total_rows']}")
    print(f"  股票数: {summary['unique_tickers']}")
    print(f"  日期范围: {summary['date_range']['start']} ~ {summary['date_range']['end']}")

    # 显示前几行
    print("\ndf_features 前5行:")
    print(df_features.head())

    # 保存到CSV（可选）
    output_path = Path(__file__).parent.parent.parent / "data" / "features.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_features.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n数据已保存至: {output_path}")

    return df_features


if __name__ == '__main__':
    df_features = main()
