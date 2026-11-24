import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class TransformerToxicDataset(Dataset):
    """Dataset PyTorch hỗ trợ cả PhoBERT và ViDeBERTa cho bài toán đa nhãn."""
    def __init__(
        self,
        csv_path: str,
        tokenizer,
        model_name: str,
        max_len: int = 128,
        label_cols: list = ["toxic", "hate", "spam", "clean"]
    ):
        self.df = pd.read_csv(csv_path, encoding="utf-8")
        raw_texts = self.df["text_processed"].fillna("").astype(str).tolist()
        
        # ViDeBERTa sử dụng văn bản tự nhiên (không nối gạch dưới như PhoBERT)
        if "videberta" in model_name.lower():
            self.texts = [t.replace("_", " ") for t in raw_texts]
        else:
            self.texts = raw_texts

        self.labels = self.df[label_cols].values.astype("float32")
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.float32)
        }


def create_transformer_dataloaders(
    model_name: str = "Fsoft-AIC/videberta-base",
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    test_path: str = "data/processed/test.csv",
    max_len: int = 128,
    batch_size: int = 16
):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    train_ds = TransformerToxicDataset(train_path, tokenizer, model_name, max_len=max_len)
    val_ds = TransformerToxicDataset(val_path, tokenizer, model_name, max_len=max_len)
    test_ds = TransformerToxicDataset(test_path, tokenizer, model_name, max_len=max_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return tokenizer, train_loader, val_loader, test_loader