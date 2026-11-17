import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
def compute_pos_weights(
    train_csv_path: str = "data/processed/train.csv",
    label_cols: list = ["toxic", "hate", "spam", "clean"]
) -> torch.Tensor:
    """
    Tính toán vector trọng số pos_weights = (số mẫu âm) / (số mẫu dương)
    từ tập train.csv để xử lý mất cân bằng dữ liệu đa nhãn.
    """
    df = pd.read_csv(train_csv_path, encoding="utf-8")
    total_samples = len(df)
    pos_counts = df[label_cols].sum().values
    neg_counts = total_samples - pos_counts
    # Nếu một nhãn chưa có mẫu dương nào (ví dụ spam = 0), đặt trọng số mặc định = 1.0
    weights=[
        float(neg) / float(pos) if pos > 0 else 1.0
        for neg, pos in zip(neg_counts, pos_counts)
    ] 
    return torch.tensor(weights, dtype=torch.float32)
class WeightedBCELoss(nn.Module):
    """
    Binary Cross-Entropy có tích hợp trọng số lớp (Pos-Weights).

    
    """
    def __init__(self, pos_weight: torch.Tensor = None):
        super().__init__()
        self.pos_weight = pos_weight
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )
class BinaryFocalLoss(nn.Module):
    """_summary_
    Binary Focal Loss (gamma = 2.0) kết hợp Pos-Weights cho bài toán Multi-Label.
    Giúp mô hình tập trung vào các mẫu khó và nhãn thiểu số (như 'hate').
    """
    def __init__(self, gamma: float =2.0, pos_weight: torch.Tensor = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none", pos_weight = self.pos_weight
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t) ** self.gamma
        return (focal_factor * bce_loss).mean()