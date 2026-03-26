#!/usr/bin/env python3
"""
Contest GRU - 主执行文件
整合完整流程：数据获取 -> 特征工程 -> 模型训练 -> 推理输出

输出格式：
- result.csv: stock_id,weight (UTF-8编码, 不超过5行, 权重累加<=1)
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch

# 添加src到路径
sys.path.insert(0, str(Path(__file__).parent))

from src.data.data_ingestion import DataIngestion
from src.data.tensor_builder import build_rolling_tensors
from src.data.quant_dataset import create_dataloaders
from src.model.quant_gru import QuantGRU, train_model
from src.inference.portfolio_generator import generate_portfolio

warnings.filterwarnings('ignore')

# 配置参数
CONFIG = {
    'start_date': '20230101',
    'end_date': datetime.now().strftime('%Y%m%d'),
    'window': 20,
    'feature_cols': ['MA5_Gap', 'Mom_20', 'Vol_20', 'Turnover', 'Corr_PV',
                     'Price_Range', 'Close_Open', 'Volume_MA10', 'Price_MA10', 'RSI_14'],
    'train_ratio': 0.8,
    'batch_size': 1,
    'epochs': 20,
    'lr': 1e-3,
    'hidden_size': 64,
    'temperature': 2.0,
    'top_k': 5,
    'output_file': 'result.csv',
}


def setup_device():
    """设置计算设备，优先使用MPS"""
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"[INFO] 使用MPS设备 (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print(f"[INFO] 使用CPU设备")
    return device


def main():
    """主执行流程"""
    print("=" * 70)
    print("Contest GRU - 量化投资模型全流程")
    print("=" * 70)
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # 步骤1: 数据获取与特征工程
    print("=" * 70)
    print("步骤1: 数据获取与特征工程")
    print("=" * 70)
    
    ingestion = DataIngestion(
        start_date=CONFIG['start_date'],
        end_date=CONFIG['end_date']
    )
    df_features = ingestion.run()
    print()

    # 步骤2: 张量构建 (截面标准化 + 滚动切片)
    print("=" * 70)
    print("步骤2: 张量构建 (截面标准化 + 滚动切片)")
    print("=" * 70)
    
    X, Y = build_rolling_tensors(
        df=df_features,
        window=CONFIG['window'],
        feature_cols=CONFIG['feature_cols']
    )
    print()

    # 转换为PyTorch张量
    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).float()
    print(f"[INFO] PyTorch张量已创建")
    print(f"  X_tensor: {X_tensor.shape}, dtype={X_tensor.dtype}")
    print(f"  Y_tensor: {Y_tensor.shape}, dtype={Y_tensor.dtype}")
    print()

    # 步骤3: 创建DataLoader
    print("=" * 70)
    print("步骤3: 创建DataLoader")
    print("=" * 70)
    
    train_loader, val_loader = create_dataloaders(
        X=X_tensor,
        Y=Y_tensor,
        train_ratio=CONFIG['train_ratio'],
        batch_size=CONFIG['batch_size']
    )
    print()

    # 步骤4: 模型训练
    print("=" * 70)
    print("步骤4: 模型训练 (QuantGRU)")
    print("=" * 70)
    
    device = setup_device()
    
    model = QuantGRU(
        input_size=len(CONFIG['feature_cols']),
        hidden_size=CONFIG['hidden_size'],
        batch_first=True
    )
    
    print(f"[INFO] 模型架构:")
    print(f"  - 输入维度: {len(CONFIG['feature_cols'])}")
    print(f"  - 隐藏层维度: {CONFIG['hidden_size']}")
    print(f"  - 训练轮数: {CONFIG['epochs']}")
    print(f"  - 学习率: {CONFIG['lr']}")
    print()
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=CONFIG['epochs'],
        lr=CONFIG['lr'],
        device=device
    )
    print()

    # 步骤5: 获取最新数据进行推理
    print("=" * 70)
    print("步骤5: 投资组合生成 (最新数据推理)")
    print("=" * 70)
    
    # 获取最新的截面数据 (最后一天的300只股票数据)
    X_latest = X_tensor[-1]  # shape: (300, 20, 10)
    print(f"[INFO] 最新截面数据形状: {X_latest.shape}")
    
    # 获取股票代码列表
    tickers = sorted(df_features['ticker'].unique())
    print(f"[INFO] 股票数量: {len(tickers)}")
    
    # 生成投资组合 (使用 portfolio_generator)
    print(f"[INFO] 生成投资组合 (Top-K={CONFIG['top_k']}, T={CONFIG['temperature']})")
    
    # 使用 portfolio_generator 生成投资组合
    output_path = Path(__file__).parent / CONFIG['output_file']
    
    portfolio_df = generate_portfolio(
        model=model,
        X_latest=X_latest,
        tickers=tickers,
        temperature=CONFIG['temperature'],
        top_k=CONFIG['top_k']
    )
    
    # 将 portfolio_df 列名转换为 result_df 格式
    result_df = portfolio_df.rename(columns={'Ticker': 'stock_id', 'Weight': 'weight'})
    
    # 保存结果 (使用正确的列名)
    result_df.to_csv(output_path, index=False, encoding='utf-8')
    
    print(f"[INFO] 投资组合生成完成")
    print()

    # 步骤6: 保存结果
    print("=" * 70)
    print("步骤6: 保存结果")
    print("=" * 70)
    
    # 文件已在上方保存
    print(f"[INFO] 结果已保存至: {output_path}")
    print(f"[INFO] 文件编码: UTF-8")
    print()
    
    # 验证输出格式
    print("=" * 70)
    print("输出验证")
    print("=" * 70)
    
    # 检查文件是否存在
    assert output_path.exists(), f"输出文件不存在: {output_path}"
    print(f"✓ 文件已创建")
    
    # 读取并验证
    saved_df = pd.read_csv(output_path, encoding='utf-8')
    
    # 验证表头
    assert list(saved_df.columns) == ['stock_id', 'weight'], f"表头错误: {list(saved_df.columns)}"
    print(f"✓ 表头正确: stock_id, weight")
    
    # 验证行数
    assert len(saved_df) <= 5, f"行数超过5: {len(saved_df)}"
    print(f"✓ 行数正确: {len(saved_df)}行 (<=5)")
    
    # 验证权重
    total_weight = saved_df['weight'].sum()
    assert total_weight <= 1.0 + 1e-5, f"权重总和超过1: {total_weight}"
    print(f"✓ 权重总和: {total_weight:.6f} (<=1.0)")
    
    # 验证逗号分隔 (CSV格式)
    with open(output_path, 'r', encoding='utf-8') as f:
        content = f.read()
        assert ',' in content, "不是逗号分隔格式"
    print(f"✓ 逗号分隔格式正确")
    
    print()
    print("=" * 70)
    print("最终结果")
    print("=" * 70)
    print(result_df.to_string(index=False))
    print()
    
    print("=" * 70)
    print("流程完成!")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    return result_df


if __name__ == '__main__':
    try:
        result = main()
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] 执行失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
