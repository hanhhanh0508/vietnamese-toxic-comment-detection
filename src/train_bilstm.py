import os
import json
import numpy as np
import torch
from sklearn.metrics import f1_score, classification_report

from src.dataset import create_dataloaders
from src.models import BiLSTMAttention
from src.loss import BinaryFocalLoss, compute_pos_weights
def evaluate(model, data_loader, device, threshold=0.5):
    model.eval()
    all_probs, all_targets = [], []
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch = x_batch.to(device)
            probs = torch.sigmoid(model(x_batch)).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(y_batch.numpy())
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)
    all_preds = (all_probs >= threshold).astype(int)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return macro_f1, all_targets, all_preds
def train_bilstm(epochs: int = 10, batch_size: int = 32, lr: float = 1e-3, max_len: int = 128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang huấn luyện trên thiết bị: {device}")
    vocab, train_loader, val_loader, test_loader = create_dataloaders(
        max_len=max_len, batch_size=batch_size
    )
    model = BiLSTMAttention(vocab_size=len(vocab)).to(device)
    pos_weights = compute_pos_weights("data/processed/train.csv").to(device)
    criterion = BinaryFocalLoss(gamma=2.0, pos_weight=pos_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    os.makedirs("checkpoints", exist_ok=True)
    best_val_f1 = 0.0
    history = {"train_loss":[], "val_macro_f1":[]}
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_macro_f1, _, _ = evaluate(model, val_loader, device)
        history["train_loss"].append(avg_loss)
        history["val_macro_f1"].append(val_macro_f1)
        print(f"Epoch [{epoch}/{epochs}] - Train Loss: {avg_loss:.4f} | Val Macro-F1: {val_macro_f1:.4f}")
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            torch.save(model.state_dict(), "checkpoints/best_bilstm.pth")
            print(f" --> Đã lưu model tốt nhất tại checkpoints/best_bilstm.pth (Val Macro-F1: {best_val_f1:.4f})")
    # Đánh giá trên tập Test với checkpoint tốt nhất
    