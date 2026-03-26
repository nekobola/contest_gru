"""
QuantGRU: Cross-Sectional Scoring GRU Model for Quantitative Finance
任务: TASK-004
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional


class QuantGRU(nn.Module):
    """
    GRU模型用于金融截面打分
    
    架构:
        - 1层GRU: input_size=10, hidden_size=64, batch_first=True
        - 提取最后一个时间步的隐状态
        - 全连接层输出截面Score
    """
    
    def __init__(self, input_size: int = 10, hidden_size: int = 64, batch_first: bool = True):
        super(QuantGRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        
        # GRU层: 输入(batch, seq_len, input_size) -> 输出(batch, seq_len, hidden_size)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=batch_first
        )
        
        # 全连接层: hidden_size -> 1 (截面打分)
        self.fc = nn.Linear(hidden_size, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量, shape (batch_size, seq_len, input_size) 或 (batch_size, input_size)
               如果输入是2D，自动添加序列维度
        
        Returns:
            score: 截面打分, shape (batch_size, 1)
        """
        # 如果输入是2D，添加序列维度 (batch, input_size) -> (batch, 1, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # GRU输出: output (batch, seq_len, hidden_size), hidden (1, batch, hidden_size)
        output, hidden = self.gru(x)
        
        # 提取最后一个时间步的隐状态: (batch, hidden_size)
        if self.batch_first:
            last_hidden = output[:, -1, :]  # (batch, hidden_size)
        else:
            last_hidden = output[-1, :, :]  # (batch, hidden_size)
        
        # 全连接层输出截面打分
        score = self.fc(last_hidden)  # (batch, 1)
        
        return score


def ic_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """
    计算IC (Pearson相关系数) Loss
    
    Loss = 1 - IC
    其中 IC = Pearson(y_pred, y_true)
    
    Args:
        y_pred: 预测分数, shape (batch_size, 1) 或 (batch_size,)
        y_true: 真实收益, shape (batch_size, 1) 或 (batch_size,)
    
    Returns:
        loss: IC Loss, 标量张量
    """
    # 展平张量
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    
    # 确保维度一致
    assert y_pred.shape == y_true.shape, f"Shape mismatch: {y_pred.shape} vs {y_true.shape}"
    
    # 计算均值
    mean_pred = torch.mean(y_pred)
    mean_true = torch.mean(y_true)
    
    # 中心化
    pred_centered = y_pred - mean_pred
    true_centered = y_true - mean_true
    
    # 计算协方差
    covariance = torch.mean(pred_centered * true_centered)
    
    # 计算标准差
    std_pred = torch.sqrt(torch.mean(pred_centered ** 2) + 1e-8)
    std_true = torch.sqrt(torch.mean(true_centered ** 2) + 1e-8)
    
    # Pearson相关系数 IC = Cov(X,Y) / (Std(X) * Std(Y))
    ic = covariance / (std_pred * std_true + 1e-8)
    
    # Loss = 1 - IC (最小化Loss等价于最大化IC)
    loss = 1.0 - ic
    
    return loss


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int = 50,
    lr: float = 1e-3,
    device: Optional[torch.device] = None
) -> Dict[str, list]:
    """
    训练QuantGRU模型
    
    Args:
        model: QuantGRU模型实例
        train_loader: 训练数据加载器
        val_loader: 验证数据加载器
        epochs: 训练轮数
        lr: 学习率
        device: 计算设备，若为None则自动检测MPS
    
    Returns:
        history: 包含训练和验证指标的字典
    """
    # 显式检测MPS设备
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"[INFO] Using device: {device}")
    print(f"[INFO] MPS available: {torch.backends.mps.is_available()}")
    
    # 将模型搬运至device
    model = model.to(device)
    
    # 初始化Adam优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # 记录训练历史
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_ic': []
    }
    
    print(f"[INFO] Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        # ==================== Training Phase ====================
        model.train()
        train_losses = []
        
        for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
            # 将数据搬运至device
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 前向传播
            optimizer.zero_grad()
            y_pred = model(x_batch)
            
            # 计算IC Loss
            loss = ic_loss(y_pred, y_batch)
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
        
        avg_train_loss = np.mean(train_losses)
        history['train_loss'].append(avg_train_loss)
        
        # ==================== Validation Phase ====================
        model.eval()
        val_losses = []
        val_ic_values = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                # 将数据搬运至device
                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                
                # 前向传播
                y_pred = model(x_batch)
                
                # 计算IC Loss
                loss = ic_loss(y_pred, y_batch)
                val_losses.append(loss.item())
                
                # 计算IC值 (1 - loss)
                ic_value = 1.0 - loss.item()
                val_ic_values.append(ic_value)
        
        avg_val_loss = np.mean(val_losses)
        avg_val_ic = np.mean(val_ic_values)
        history['val_loss'].append(avg_val_loss)
        history['val_ic'].append(avg_val_ic)
        
        # 每个epoch后打印验证集IC
        print(f"Epoch [{epoch+1}/{epochs}] - "
              f"Train Loss: {avg_train_loss:.4f}, "
              f"Val Loss: {avg_val_loss:.4f}, "
              f"Val IC: {avg_val_ic:.4f}")
    
    print(f"[INFO] Training completed!")
    print(f"[INFO] Final Val IC: {avg_val_ic:.4f}")
    
    return history


def test_ic_loss_implementation():
    """测试IC Loss实现是否正确"""
    print("\n" + "="*60)
    print("[TEST] Testing IC Loss Implementation")
    print("="*60)
    
    # 测试数据: 完全正相关
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = torch.tensor([2.0, 4.0, 6.0, 8.0, 10.0])
    
    loss = ic_loss(y_pred, y_true)
    expected_ic = 1.0  # 完全正相关
    expected_loss = 0.0
    
    print(f"Test 1 - Perfect Positive Correlation:")
    print(f"  Predicted: {y_pred.numpy()}")
    print(f"  True:      {y_true.numpy()}")
    print(f"  IC Loss:   {loss.item():.6f} (expected ~0.0)")
    print(f"  IC Value:  {1-loss.item():.6f} (expected ~1.0)")
    assert abs(loss.item() - expected_loss) < 1e-5, "IC Loss test failed!"
    print("  ✓ PASSED")
    
    # 测试数据: 完全负相关
    y_pred = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    y_true = torch.tensor([10.0, 8.0, 6.0, 4.0, 2.0])
    
    loss = ic_loss(y_pred, y_true)
    expected_ic = -1.0
    expected_loss = 2.0
    
    print(f"\nTest 2 - Perfect Negative Correlation:")
    print(f"  Predicted: {y_pred.numpy()}")
    print(f"  True:      {y_true.numpy()}")
    print(f"  IC Loss:   {loss.item():.6f} (expected ~2.0)")
    print(f"  IC Value:  {1-loss.item():.6f} (expected ~-1.0)")
    assert abs(loss.item() - expected_loss) < 1e-5, "IC Loss test failed!"
    print("  ✓ PASSED")
    
    # 测试数据: 无相关
    torch.manual_seed(42)
    y_pred = torch.randn(100)
    y_true = torch.randn(100)
    
    loss = ic_loss(y_pred, y_true)
    ic = 1 - loss.item()
    
    print(f"\nTest 3 - Random (No Correlation):")
    print(f"  IC Loss:  {loss.item():.6f}")
    print(f"  IC Value: {ic:.6f} (expected near 0.0)")
    assert abs(ic) < 0.3, f"IC should be near 0 for random data, got {ic}"
    print("  ✓ PASSED")
    
    print("\n" + "="*60)
    print("[TEST] All IC Loss Tests PASSED!")
    print("="*60 + "\n")


def test_model_forward():
    """测试模型前向传播"""
    print("="*60)
    print("[TEST] Testing QuantGRU Forward Pass")
    print("="*60)
    
    # 测试3D输入 (batch, seq_len, input_size)
    batch_size = 16
    seq_len = 20
    input_size = 10
    
    model = QuantGRU(input_size=input_size, hidden_size=64, batch_first=True)
    x = torch.randn(batch_size, seq_len, input_size)
    
    output = model(x)
    
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Expected:     ({batch_size}, 1)")
    
    assert output.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output.shape}"
    print("  ✓ PASSED")
    
    # 测试2D输入 (batch, input_size)
    x_2d = torch.randn(batch_size, input_size)
    output_2d = model(x_2d)
    
    print(f"\n2D Input shape:  {x_2d.shape}")
    print(f"Output shape:    {output_2d.shape}")
    
    assert output_2d.shape == (batch_size, 1), f"Expected shape ({batch_size}, 1), got {output_2d.shape}"
    print("  ✓ PASSED")
    
    print("\n" + "="*60)
    print("[TEST] Model Forward Tests PASSED!")
    print("="*60 + "\n")


def test_training_loop():
    """测试完整训练循环"""
    print("="*60)
    print("[TEST] Testing Full Training Loop")
    print("="*60)
    
    # 创建合成数据
    batch_size = 32
    seq_len = 30
    input_size = 10
    n_samples = 256
    
    # 生成训练数据
    X_train = torch.randn(n_samples, seq_len, input_size)
    # 创建与输入有一定相关性的目标
    y_train = torch.randn(n_samples, 1) * 0.5 + X_train[:, -1, 0:1] * 0.3
    
    X_val = torch.randn(n_samples // 4, seq_len, input_size)
    y_val = torch.randn(n_samples // 4, 1) * 0.5 + X_val[:, -1, 0:1] * 0.3
    
    # 创建DataLoader
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size)
    
    # 初始化模型
    model = QuantGRU(input_size=input_size, hidden_size=64, batch_first=True)
    
    # 训练 (使用少量epoch进行测试)
    print(f"\nStarting test training with 5 epochs...")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=1e-3
    )
    
    # 验证训练历史
    assert len(history['train_loss']) == 5, "Train loss history length mismatch"
    assert len(history['val_loss']) == 5, "Val loss history length mismatch"
    assert len(history['val_ic']) == 5, "Val IC history length mismatch"
    
    print(f"\n✓ Training completed successfully!")
    print(f"✓ Final Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"✓ Final Val IC: {history['val_ic'][-1]:.4f}")
    
    print("\n" + "="*60)
    print("[TEST] Training Loop Tests PASSED!")
    print("="*60 + "\n")
    
    return history


if __name__ == "__main__":
    """
    测试代码验证模型可正常训练
    """
    print("\n" + "="*70)
    print("  QuantGRU Model Test Suite - TASK-004")
    print("="*70 + "\n")
    
    # 1. 测试IC Loss实现
    test_ic_loss_implementation()
    
    # 2. 测试模型前向传播
    test_model_forward()
    
    # 3. 测试完整训练循环
    test_training_loop()
    
    print("="*70)
    print("  ALL TESTS PASSED - TASK-004 COMPLETE!")
    print("="*70 + "\n")
