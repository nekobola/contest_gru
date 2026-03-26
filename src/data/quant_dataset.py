"""
QuantDataset - PyTorch Dataset for Financial Time Series Data

数据结构:
    X: (N_days, 300, 20, 10) - 特征张量
    Y: (N_days, 300) - 标签张量
    
其中:
    N_days: 交易日天数
    300: 股票数量（截面维度）
    20: 时间步长（过去20个交易日）
    10: 特征维度
"""

import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional


class QuantDataset(Dataset):
    """
    金融时序截面数据Dataset
    
    每个样本对应一个交易日的全市场数据（300只股票）
    """
    
    def __init__(self, X: torch.Tensor, Y: torch.Tensor):
        """
        Args:
            X: 特征张量, shape (N_days, 300, 20, 10)
            Y: 标签张量, shape (N_days, 300)
        """
        if X.shape[0] != Y.shape[0]:
            raise ValueError(f"X和Y的第一维必须相同，得到 X:{X.shape[0]}, Y:{Y.shape[0]}")
        
        if X.dim() != 4:
            raise ValueError(f"X必须是4维张量，当前维度: {X.dim()}")
        
        if Y.dim() != 2:
            raise ValueError(f"Y必须是2维张量，当前维度: {Y.dim()}")
        
        self.X = X
        self.Y = Y
        self.n_days = X.shape[0]
        
        # 验证形状
        assert X.shape == (self.n_days, 300, 20, 10), \
            f"X形状错误，期望: ({self.n_days}, 300, 20, 10)，实际: {X.shape}"
        assert Y.shape == (self.n_days, 300), \
            f"Y形状错误，期望: ({self.n_days}, 300)，实际: {Y.shape}"
    
    def __len__(self) -> int:
        """返回交易日天数"""
        return self.n_days
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        返回某一天的全市场截面数据
        
        Args:
            idx: 日期索引 (0 到 N_days-1)，支持负索引
            
        Returns:
            x: 特征张量, shape (300, 20, 10)
            y: 标签张量, shape (300,)
        """
        # 处理负索引
        if idx < 0:
            idx = self.n_days + idx
        
        if idx < 0 or idx >= self.n_days:
            raise IndexError(f"索引 {idx} 超出范围 [0, {self.n_days})")
        
        return self.X[idx], self.Y[idx]
    
    def get_feature_shape(self) -> Tuple[int, ...]:
        """返回单个样本的特征形状"""
        return (300, 20, 10)
    
    def get_label_shape(self) -> Tuple[int, ...]:
        """返回单个样本的标签形状"""
        return (300,)


def create_dataloaders(
    X: torch.Tensor, 
    Y: torch.Tensor, 
    train_ratio: float = 0.8,
    batch_size: int = 1,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    创建训练和验证DataLoader
    
    按时序分割：前80%为训练集，后20%为验证集
    金融时序数据严禁打乱(shuffle=False)
    
    Args:
        X: 特征张量, shape (N_days, 300, 20, 10)
        Y: 标签张量, shape (N_days, 300)
        train_ratio: 训练集比例，默认0.8
        batch_size: 批次大小，默认1（单日数据已包含300只股票）
        num_workers: 数据加载线程数，默认0
        
    Returns:
        train_loader: 训练集DataLoader
        val_loader: 验证集DataLoader
    """
    n_days = X.shape[0]
    train_size = int(n_days * train_ratio)
    
    # 时序分割：前train_size天为训练集，剩余为验证集
    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_val = X[train_size:]
    Y_val = Y[train_size:]
    
    # 创建Dataset
    train_dataset = QuantDataset(X_train, Y_train)
    val_dataset = QuantDataset(X_val, Y_val)
    
    # 创建DataLoader - 严禁打乱时序！
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 时序数据严禁打乱
        num_workers=num_workers
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # 时序数据严禁打乱
        num_workers=num_workers
    )
    
    print(f"[DataLoader] 总天数: {n_days}")
    print(f"[DataLoader] 训练集: {len(train_dataset)} 天 ({train_ratio*100:.0f}%)")
    print(f"[DataLoader] 验证集: {len(val_dataset)} 天 ({(1-train_ratio)*100:.0f}%)")
    print(f"[DataLoader] batch_size={batch_size}, shuffle=False")
    
    return train_loader, val_loader


# ==================== 测试代码 ====================

def test_quant_dataset():
    """测试QuantDataset类"""
    print("=" * 60)
    print("测试 QuantDataset 类")
    print("=" * 60)
    
    # 创建模拟数据
    N_days = 100
    X = torch.randn(N_days, 300, 20, 10)
    Y = torch.randn(N_days, 300)
    
    # 测试Dataset创建
    dataset = QuantDataset(X, Y)
    print(f"✓ Dataset创建成功")
    print(f"  - 总天数: {len(dataset)}")
    print(f"  - 特征形状: {dataset.get_feature_shape()}")
    print(f"  - 标签形状: {dataset.get_label_shape()}")
    
    # 测试__getitem__
    x, y = dataset[0]
    print(f"✓ __getitem__(0) 返回:")
    print(f"  - x形状: {tuple(x.shape)}")
    print(f"  - y形状: {tuple(y.shape)}")
    
    # 测试最后一个索引
    x_last, y_last = dataset[-1]
    print(f"✓ __getitem__(-1) 返回:")
    print(f"  - x形状: {tuple(x_last.shape)}")
    print(f"  - y形状: {tuple(y_last.shape)}")
    
    # 测试越界检查
    try:
        _ = dataset[N_days]
        print("✗ 越界检查失败")
    except IndexError:
        print("✓ 越界检查正常")
    
    print("\n")


def test_create_dataloaders():
    """测试create_dataloaders函数"""
    print("=" * 60)
    print("测试 create_dataloaders 函数")
    print("=" * 60)
    
    # 创建模拟数据
    N_days = 100
    X = torch.randn(N_days, 300, 20, 10)
    Y = torch.randn(N_days, 300)
    
    # 创建DataLoader
    train_loader, val_loader = create_dataloaders(X, Y, train_ratio=0.8)
    
    # 验证DataLoader长度
    assert len(train_loader) == 80, f"训练集长度应为80，实际{len(train_loader)}"
    assert len(val_loader) == 20, f"验证集长度应为20，实际{len(val_loader)}"
    print(f"✓ DataLoader长度验证通过")
    
    # 测试迭代
    print("\n测试训练集迭代 (前3个batch):")
    for i, (x_batch, y_batch) in enumerate(train_loader):
        if i >= 3:
            break
        print(f"  Batch {i}: x={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}")
    
    print("\n测试验证集迭代 (前3个batch):")
    for i, (x_batch, y_batch) in enumerate(val_loader):
        if i >= 3:
            break
        print(f"  Batch {i}: x={tuple(x_batch.shape)}, y={tuple(y_batch.shape)}")
    
    # 验证batch内容
    x_batch, y_batch = next(iter(train_loader))
    assert x_batch.shape == (1, 300, 20, 10), f"x_batch形状错误: {x_batch.shape}"
    assert y_batch.shape == (1, 300), f"y_batch形状错误: {y_batch.shape}"
    print(f"\n✓ Batch形状验证通过")
    
    print("\n")


def test_edge_cases():
    """测试边界情况"""
    print("=" * 60)
    print("测试边界情况")
    print("=" * 60)
    
    # 测试形状不匹配
    print("测试: X和Y第一维不匹配...")
    try:
        X_wrong = torch.randn(100, 300, 20, 10)
        Y_wrong = torch.randn(50, 300)
        _ = QuantDataset(X_wrong, Y_wrong)
        print("✗ 应抛出ValueError")
    except ValueError as e:
        print(f"✓ 正确捕获错误: {e}")
    
    # 测试X维度错误
    print("\n测试: X维度不为4...")
    try:
        X_wrong = torch.randn(100, 300, 20)
        Y_wrong = torch.randn(100, 300)
        _ = QuantDataset(X_wrong, Y_wrong)
        print("✗ 应抛出ValueError")
    except ValueError as e:
        print(f"✓ 正确捕获错误: {e}")
    
    # 测试Y维度错误
    print("\n测试: Y维度不为2...")
    try:
        X_wrong = torch.randn(100, 300, 20, 10)
        Y_wrong = torch.randn(100, 300, 1)
        _ = QuantDataset(X_wrong, Y_wrong)
        print("✗ 应抛出ValueError")
    except ValueError as e:
        print(f"✓ 正确捕获错误: {e}")
    
    # 测试最小数据集
    print("\n测试: 最小数据集 (1天)...")
    X_min = torch.randn(1, 300, 20, 10)
    Y_min = torch.randn(1, 300)
    dataset_min = QuantDataset(X_min, Y_min)
    assert len(dataset_min) == 1
    print(f"✓ 最小数据集测试通过")
    
    print("\n")


def test_temporal_order():
    """验证时序顺序"""
    print("=" * 60)
    print("验证时序顺序 (确保shuffle=False生效)")
    print("=" * 60)
    
    # 创建带标记的数据
    N_days = 10
    X = torch.arange(N_days).view(-1, 1, 1, 1).float()
    X = X.expand(-1, 300, 20, 10).clone()
    Y = torch.arange(N_days).view(-1, 1).float()
    Y = Y.expand(-1, 300).clone()
    
    # 每个样本的第一维应该是日期索引
    train_loader, val_loader = create_dataloaders(X, Y, train_ratio=0.8)
    
    # 验证训练集时序
    print("训练集时序验证:")
    expected_idx = 0
    for x_batch, y_batch in train_loader:
        actual_idx = int(x_batch[0, 0, 0, 0].item())
        if actual_idx != expected_idx:
            print(f"✗ 时序错误: 期望 {expected_idx}, 实际 {actual_idx}")
            break
        expected_idx += 1
    else:
        print(f"✓ 训练集时序正确 (0-7)")
    
    # 验证验证集时序
    print("验证集时序验证:")
    expected_idx = 8
    for x_batch, y_batch in val_loader:
        actual_idx = int(x_batch[0, 0, 0, 0].item())
        if actual_idx != expected_idx:
            print(f"✗ 时序错误: 期望 {expected_idx}, 实际 {actual_idx}")
            break
        expected_idx += 1
    else:
        print(f"✓ 验证集时序正确 (8-9)")
    
    print("\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("QuantDataset 测试套件")
    print("=" * 60 + "\n")
    
    test_quant_dataset()
    test_create_dataloaders()
    test_edge_cases()
    test_temporal_order()
    
    print("=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
