"""
Simple portfolio backtesting module.
Evaluates portfolio performance based on price data.
"""

import pandas as pd
import numpy as np


def backtest_portfolio(portfolio_df, price_data, period=5):
    """
    对投资组合进行简单的回测，计算收益表现。
    
    Parameters:
    -----------
    portfolio_df : pd.DataFrame
        投资组合，包含 stock_id 和 weight 列
    price_data : pd.DataFrame
        未来价格数据，列名为 stock_id，行为时间序列
    period : int
        回测周期（天数），默认5天
        
    Returns:
    --------
    dict
        {
            'cumulative_return': 累计收益率,
            'daily_returns': 日收益率序列,
            'final_value': 最终 portfolio value
        }
    """
    # 获取投资组合中的股票
    stocks = portfolio_df['stock_id'].tolist()
    weights = portfolio_df.set_index('stock_id')['weight'].to_dict()
    
    # 初始化 portfolio value 序列
    portfolio_values = []
    
    # 计算每日 portfolio value
    for day in range(period + 1):
        if day >= len(price_data):
            break
        
        daily_value = 0.0
        for stock in stocks:
            if stock in price_data.columns:
                weight = weights.get(stock, 0)
                price = price_data.iloc[day][stock]
                # 假设初始投资为 1，按权重分配
                daily_value += weight * price
        
        portfolio_values.append(daily_value)
    
    # 计算日收益率序列
    daily_returns = []
    for i in range(1, len(portfolio_values)):
        ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1] \
              if portfolio_values[i-1] != 0 else 0
        daily_returns.append(ret)
    
    # 计算累计收益率
    if len(portfolio_values) > 0 and portfolio_values[0] != 0:
        cumulative_return = (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0]
    else:
        cumulative_return = 0.0
    
    return {
        'cumulative_return': cumulative_return,
        'daily_returns': daily_returns,
        'final_value': portfolio_values[-1] if portfolio_values else 0.0
    }
