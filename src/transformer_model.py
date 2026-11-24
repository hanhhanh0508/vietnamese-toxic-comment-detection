import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class MultiLabelTransformer(nn.Module):
    """Kiến trúc Pre-trained Transformer kết hợp Mean Pooling cho phân loại đa nhãn."""
    def __init__(
        self,
        model_name: str = "Fsoft-AIC/videberta-base",
        num_classes: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state  # (batch, seq_len, hidden_size)

        # Mean Pooling có trọng số theo attention_mask (bỏ qua vị trí padding)
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        logits = self.classifier(self.dropout(pooled_output))
        return logits