from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

class Vocabulary:
    """Bộ từ vựng ánh xạ từ (word) sang chỉ số (index)."""
    def __init__(self, min_freq: int = 2):
        self.word2idx = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word = {0: "<PAD>", 1: "<UNK>"}
        self.min_freq = min_freq

    def build_vocab(self, texts: list):
        counter = Counter()
        for text in texts:
            counter.update(str(text).split())

        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, text: str, max_len: int = 128) -> list:
        tokens = str(text).split()[:max_len]
        ids = [self.word2idx.get(token, self.word2idx["<UNK>"]) for token in tokens]
        ids += [self.word2idx["<PAD>"]] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


class ToxicDataset(Dataset):
    """PyTorch Dataset cho dữ liệu bình luận độc hại đa nhãn."""
    def __init__(
        self,
        csv_path: str,
        vocab: Vocabulary,
        max_len: int = 128,
        label_cols: list = ["toxic", "hate", "spam", "clean"]
    ):
        self.df = pd.read_csv(csv_path, encoding="utf-8")
        self.texts = self.df["text_processed"].fillna("").tolist()
        self.labels = self.df[label_cols].values.astype("float32")
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        input_ids = torch.tensor(
            self.vocab.encode(self.texts[idx], self.max_len), dtype=torch.long
        )
        labels = torch.tensor(self.labels[idx], dtype=torch.float32)
        return input_ids, labels


def create_dataloaders(
    train_path: str = "data/processed/train.csv",
    val_path: str = "data/processed/val.csv",
    test_path: str = "data/processed/test.csv",
    max_len: int = 128,
    batch_size: int = 32,
    min_freq: int = 2
):
    """Khởi tạo bộ từ vựng từ tập train và trả về 3 DataLoader."""
    train_df = pd.read_csv(train_path, encoding="utf-8")
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(train_df["text_processed"].fillna("").tolist())

    train_dataset = ToxicDataset(train_path, vocab, max_len=max_len)
    val_dataset = ToxicDataset(val_path, vocab, max_len=max_len)
    test_dataset = ToxicDataset(test_path, vocab, max_len=max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return vocab, train_loader, val_loader, test_loader