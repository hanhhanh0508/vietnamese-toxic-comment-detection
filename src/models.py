import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """Cơ chế Attention tính trọng số quan trọng của từng từ trong câu."""
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output: torch.Tensor, mask: torch.Tensor = None):
        scores = self.attn(lstm_output).squeeze(-1)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), lstm_output).squeeze(1)
        return context, attn_weights


class BiLSTMAttention(nn.Module):
    """Mô hình BiLSTM kết hợp Attention cho phân loại đa nhãn."""
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, batch_first=True, bidirectional=True
        )
        self.attention = Attention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, input_ids: torch.Tensor):
        mask = (input_ids != 0).float()
        embedded = self.dropout(self.embedding(input_ids))
        lstm_out, _ = self.lstm(embedded)
        context, _ = self.attention(lstm_out, mask=mask)
        logits = self.fc(self.dropout(context))
        return logits