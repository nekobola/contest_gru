"""
Portfolio Generator Module
==========================
金融投资组合权重生成器 - 基于PyTorch模型推理和Temperature Softmax权重计算

Author: PyTorch Inference Engineer
Task: TASK-005
"""

import torch
import pandas as pd
import numpy as np
from typing import List, Optional
import os


def generate_portfolio(
    model: torch.nn.Module,
    X_latest: torch.Tensor,
    tickers: List[str],
    temperature: float = 2.0,
    top_k: int = 5,
    output_path: Optional[str] = None
) -> pd.DataFrame:
    """
    基于模型推理生成投资组合权重
    
    Args:
        model: PyTorch模型，输出形状为 (batch_size, 1)
        X_latest: 最新截面数据，形状 (300, 20, 10)
        tickers: 股票代码列表，长度300
        temperature: Softmax温度系数，默认2.0
        top_k: 选择的Top K股票数量，默认5
        output_path: CSV保存路径，可选
    
    Returns:
        pd.DataFrame: 包含 [Ticker, Weight] 两列的DataFrame
    
    Raises:
        ValueError: 输入形状不匹配或参数错误
    """
    # ========== 输入验证 ==========
    if X_latest.shape != (300, 20, 10):
        raise ValueError(f"X_latest 形状必须是 (300, 20, 10)，当前: {X_latest.shape}")
    
    if len(tickers) != 300:
        raise ValueError(f"tickers 长度必须是300，当前: {len(tickers)}")
    
    if temperature <= 0:
        raise ValueError(f"temperature 必须大于0，当前: {temperature}")
    
    if top_k <= 0 or top_k > 300:
        raise ValueError(f"top_k 必须在1-300之间，当前: {top_k}")
    
    # 确保模型在评估模式
    model.eval()
    
    # ========== 模型推理 ==========
    with torch.no_grad():
        # 添加batch维度: (1, 300, 20, 10) -> 模型处理
        # 假设模型接受 (batch, stocks, time, features) 或需要reshape
        # 根据任务描述，模型前向传播得到 scores (300, 1)
        
        # 如果X_latest没有batch维度，添加它
        if X_latest.dim() == 3:
            X_input = X_latest.unsqueeze(0)  # (1, 300, 20, 10)
        else:
            X_input = X_latest
        
        # 模型前向传播
        scores = model(X_input)  # 预期输出: (300, 1) 或 (1, 300, 1)
        
        # 确保scores形状正确
        if scores.dim() == 3:
            scores = scores.squeeze(0)  # (300, 1)
        if scores.dim() == 2 and scores.shape[1] == 1:
            scores = scores.squeeze(1)  # (300,)
        
        if scores.shape[0] != 300:
            raise ValueError(f"模型输出第一维必须是300，当前: {scores.shape}")
    
    # ========== Top-K 选择 ==========
    # 使用 torch.topk 选出得分最高的 top_k 只股票
    top_k_values, top_k_indices = torch.topk(scores, k=top_k, largest=True, sorted=True)
    
    # ========== Temperature Softmax 权重计算 ==========
    # w_i = exp(S_i / T) / sum_j(exp(S_j / T))
    # 其中 S_i 是股票i的得分，T是温度系数
    
    # 计算温度缩放后的分数
    scaled_scores = top_k_values / temperature  # (top_k,)
    
    # 计算softmax权重
    exp_scores = torch.exp(scaled_scores)
    weights = exp_scores / exp_scores.sum()  # 归一化，确保总和为1
    
    # 转换为numpy便于处理
    weights_np = weights.cpu().numpy()
    indices_np = top_k_indices.cpu().numpy()
    
    # ========== 构建输出DataFrame ==========
    selected_tickers = [tickers[i] for i in indices_np]
    
    portfolio_df = pd.DataFrame({
        'Ticker': selected_tickers,
        'Weight': weights_np
    })
    
    # 验证权重归一化
    weight_sum = portfolio_df['Weight'].sum()
    if not np.isclose(weight_sum, 1.0, rtol=1e-5):
        # 强制归一化以处理数值误差
        portfolio_df['Weight'] = portfolio_df['Weight'] / weight_sum
    
    # ========== 保存CSV ==========
    if output_path is not None:
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        portfolio_df.to_csv(output_path, index=False)
        print(f"[INFO] 投资组合已保存至: {output_path}")
    
    return portfolio_df


# =============================================================================
# 测试代码
# =============================================================================

def test_generate_portfolio():
    """
    测试 generate_portfolio 函数的完整功能
    """
    print("=" * 60)
    print("[TEST] Portfolio Generator 功能测试")
    print("=" * 60)
    
    # ========== 设置随机种子确保可复现 ==========
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ========== 创建模拟模型 ==========
    class MockGRUModel(torch.nn.Module):
        """模拟GRU模型用于测试"""
        def __init__(self):
            super().__init__()
            self.gru = torch.nn.GRU(input_size=10, hidden_size=32, batch_first=True)
            self.fc = torch.nn.Linear(32, 1)
        
        def forward(self, x):
            # x: (batch, 300, 20, 10)
            batch_size, n_stocks, seq_len, n_features = x.shape
            
            # 处理每只股票的时间序列
            scores = []
            for i in range(n_stocks):
                stock_data = x[:, i, :, :]  # (batch, 20, 10)
                out, _ = self.gru(stock_data)  # (batch, 20, 32)
                last_hidden = out[:, -1, :]  # (batch, 32)
                score = self.fc(last_hidden)  # (batch, 1)
                scores.append(score)
            
            # 拼接所有股票的分数
            scores = torch.cat(scores, dim=1)  # (batch, 300)
            return scores.squeeze(0) if batch_size == 1 else scores  # (300,) 或 (batch, 300)
    
    # ========== 准备测试数据 ==========
    model = MockGRUModel()
    X_latest = torch.randn(300, 20, 10)  # (300, 20, 10)
    tickers = [f"STOCK_{i:03d}" for i in range(300)]
    
    print(f"[INFO] 输入数据形状: X_latest = {X_latest.shape}")
    print(f"[INFO] 股票代码数量: {len(tickers)}")
    print(f"[INFO] 温度系数 T = 2.0")
    print(f"[INFO] Top-K = 5")
    print()
    
    # ========== 测试1: 基本功能测试 ==========
    print("[TEST-1] 基本功能测试...")
    try:
        output_path = "/tmp/test_portfolio.csv"
        result_df = generate_portfolio(
            model=model,
            X_latest=X_latest,
            tickers=tickers,
            temperature=2.0,
            top_k=5,
            output_path=output_path
        )
        
        # 验证输出
        assert isinstance(result_df, pd.DataFrame), "输出必须是DataFrame"
        assert list(result_df.columns) == ['Ticker', 'Weight'], "列名必须是 ['Ticker', 'Weight']"
        assert len(result_df) == 5, "必须选出5只股票"
        
        print(f"[PASS] DataFrame形状: {result_df.shape}")
        print(f"[PASS] 列名: {list(result_df.columns)}")
        print()
        
    except Exception as e:
        print(f"[FAIL] 基本功能测试失败: {e}")
        raise
    
    # ========== 测试2: 权重归一化验证 ==========
    print("[TEST-2] 权重归一化验证...")
    weight_sum = result_df['Weight'].sum()
    print(f"[INFO] 权重总和: {weight_sum:.10f}")
    
    assert np.isclose(weight_sum, 1.0, rtol=1e-5), f"权重总和必须接近1.0，当前: {weight_sum}"
    print(f"[PASS] 权重已正确归一化 (sum ≈ 1.0)")
    print()
    
    # ========== 测试3: Temperature Softmax 验证 ==========
    print("[TEST-3] Temperature Softmax 计算验证...")
    
    # 手动计算验证
    model.eval()
    with torch.no_grad():
        scores = model(X_latest.unsqueeze(0))
    
    # 获取Top 5
    top_values, top_indices = torch.topk(scores, k=5)
    
    # 手动计算temperature softmax
    T = 2.0
    manual_weights = torch.exp(top_values / T) / torch.sum(torch.exp(top_values / T))
    
    # 比较
    result_weights = torch.tensor(result_df['Weight'].values)
    weight_diff = torch.abs(manual_weights - result_weights).max().item()
    
    print(f"[INFO] 手动计算的权重: {manual_weights.numpy()}")
    print(f"[INFO] 函数输出的权重: {result_weights.numpy()}")
    print(f"[INFO] 最大差异: {weight_diff:.10f}")
    
    assert weight_diff < 1e-5, f"Softmax计算不匹配，差异: {weight_diff}"
    print(f"[PASS] Temperature Softmax 计算正确")
    print()
    
    # ========== 测试4: CSV文件保存验证 ==========
    print("[TEST-4] CSV文件保存验证...")
    assert os.path.exists(output_path), f"CSV文件未创建: {output_path}"
    
    loaded_df = pd.read_csv(output_path)
    assert len(loaded_df) == 5, "CSV文件行数不正确"
    assert list(loaded_df.columns) == ['Ticker', 'Weight'], "CSV列名不正确"
    
    print(f"[PASS] CSV文件已保存并验证: {output_path}")
    print(f"[INFO] CSV内容预览:")
    print(loaded_df.to_string(index=False))
    print()
    
    # ========== 测试5: 异常处理测试 ==========
    print("[TEST-5] 异常处理测试...")
    
    # 测试错误的输入形状
    try:
        X_wrong = torch.randn(100, 20, 10)  # 错误的股票数量
        generate_portfolio(model, X_wrong, tickers[:100])
        assert False, "应该抛出形状错误异常"
    except ValueError as e:
        print(f"[PASS] 正确捕获形状错误: {e}")
    
    # 测试错误的tickers长度
    try:
        wrong_tickers = tickers[:100]  # 长度不匹配
        generate_portfolio(model, X_latest, wrong_tickers)
        assert False, "应该抛出tickers长度错误异常"
    except ValueError as e:
        print(f"[PASS] 正确捕获tickers长度错误: {e}")
    
    # 测试错误的temperature
    try:
        generate_portfolio(model, X_latest, tickers, temperature=-1.0)
        assert False, "应该抛出temperature错误异常"
    except ValueError as e:
        print(f"[PASS] 正确捕获temperature错误: {e}")
    
    print()
    
    # ========== 测试6: 不同温度系数对比 ==========
    print("[TEST-6] 不同温度系数效果对比...")
    
    for T in [0.5, 1.0, 2.0, 5.0]:
        df = generate_portfolio(model, X_latest, tickers, temperature=T, top_k=5)
        weight_std = df['Weight'].std()
        print(f"[INFO] T={T}: 权重标准差={weight_std:.4f}, 范围=[{df['Weight'].min():.4f}, {df['Weight'].max():.4f}]")
    
    print("[PASS] 温度系数对比完成 (低温=激进，高温=分散)")
    print()
    
    # ========== 最终结果 ==========
    print("=" * 60)
    print("[RESULT] 所有测试通过!")
    print("=" * 60)
    print()
    print("最终投资组合示例 (T=2.0):")
    print(result_df.to_string(index=False))
    print()
    
    # 清理测试文件
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"[INFO] 测试文件已清理: {output_path}")
    
    return True


def demo_usage():
    """
    使用示例
    """
    print("\n" + "=" * 60)
    print("[DEMO] Portfolio Generator 使用示例")
    print("=" * 60)
    
    # 创建简单示例模型
    class SimpleModel(torch.nn.Module):
        def forward(self, x):
            # 简单的示例: 返回随机分数
            batch = x.shape[0] if x.dim() == 4 else 1
            return torch.randn(300) if batch == 1 else torch.randn(batch, 300)
    
    model = SimpleModel()
    X_latest = torch.randn(300, 20, 10)
    tickers = [f"AAPL", f"GOOGL", f"MSFT", f"AMZN", f"TSLA"] + [f"STOCK_{i}" for i in range(5, 300)]
    
    # 生成投资组合
    portfolio = generate_portfolio(
        model=model,
        X_latest=X_latest,
        tickers=tickers,
        temperature=2.0,
        output_path="/tmp/demo_portfolio.csv"
    )
    
    print("\n生成的投资组合:")
    print(portfolio.to_string(index=False))
    
    # 验证权重
    print(f"\n权重总和: {portfolio['Weight'].sum():.10f}")
    print(f"\n文件已保存至: /tmp/demo_portfolio.csv")


if __name__ == "__main__":
    # 运行完整测试
    test_generate_portfolio()
    
    # 运行使用示例
    demo_usage()
