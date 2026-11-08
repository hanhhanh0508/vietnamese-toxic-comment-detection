"""
Script phân chia và chuẩn hóa dữ liệu đa nhãn (Multi-Label Dataset Splitter)
Phụ trách: Sinh viên 1 & Sinh viên 2 (Phối hợp thực hiện)
Mục tiêu:
- Đọc dữ liệu thô ViHSD / ViCTSD từ data/raw/
- Ánh xạ sang 4 nhãn đa mục tiêu: [toxic, hate, spam, clean]
- Chạy qua pipeline tiền xử lý (src/preprocessing.py) để tạo cột text_processed
- Thực hiện Stratified Split chuẩn: Train (80%), Val (10%), Test (10%)
- Lưu kết quả tại data/processed/ (train.csv, val.csv, test.csv)
"""

import os
import sys

# Đảm bảo đường dẫn gốc của project luôn nằm trong sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from src.preprocessing import preprocess_text
from src.build_merged_dataset import build_merged_dataset
def create_multilabel_dataset():
    print("=== BẮT ĐẦU QUY TRÌNH XỬ LÝ & PHÂN CHIA DỮ LIỆU ĐA NHÃN ===")
    """
    # 1. Tải các file ViHSD gốc
    raw_files = [
        "data/raw/train.csv",
        "data/raw/dev.csv",
        "data/raw/test.csv"
    ]
    
    dfs = []
    for f in raw_files:
        if os.path.exists(f):
            temp_df = pd.read_csv(f, encoding="utf-8")
            dfs.append(temp_df)
            print(f"- Đã nạp {f}: {len(temp_df)} dòng")
        else:
            print(f"- Cảnh báo: Không tìm thấy {f}")
            
    if not dfs:
        raise FileNotFoundError("Không tìm thấy dữ liệu thô trong data/raw/")
        
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"\nTổng số mẫu thô ban đầu: {len(full_df)}")
    
    # 2. Xóa các dòng rỗng hoặc NaN
    full_df = full_df.dropna(subset=["free_text"]).reset_index(drop=True)
    full_df = full_df.rename(columns={"free_text": "text_raw"})
    
    # 3. Ánh xạ sang 4 nhãn Multi-Label: [toxic, hate, spam, clean]
    # ViHSD label_id: 0: Clean, 1: Offensive, 2: Hate
    toxic_col = []
    hate_col = []
    spam_col = []
    clean_col = []
    
    for lbl in full_df["label_id"].values:
        if lbl == 0:  # Clean
            toxic_col.append(0)
            hate_col.append(0)
            spam_col.append(0)
            clean_col.append(1)
        elif lbl == 1:  # Offensive / Toxic
            toxic_col.append(1)
            hate_col.append(0)
            spam_col.append(0)
            clean_col.append(0)
        elif lbl == 2:  # Hate (mang cả tính toxic và hate)
            toxic_col.append(1)
            hate_col.append(1)
            spam_col.append(0)
            clean_col.append(0)
        else:
            # Fallback
            toxic_col.append(0)
            hate_col.append(0)
            spam_col.append(0)
            clean_col.append(1)
            
    full_df["toxic"] = toxic_col
    full_df["hate"] = hate_col
    full_df["spam"] = spam_col
    full_df["clean"] = clean_col
    
    # Lưu merged_dataset.csv tại data/raw/
    full_df.to_csv("data/raw/merged_dataset.csv", index=False, encoding="utf-8")
    print(f"- Đã lưu file hợp nhất tại data/raw/merged_dataset.csv ({len(full_df)} dòng)")
    */
    """
    # 1-3. Gộp ViHSD + ViCTSD về 4 nhãn [toxic, hate, spam, clean] (xem src/build_merged_dataset.py)
    # Đổi spam_rules=True nếu nhóm đồng ý dùng luật gán nhãn spam (heuristic)
    full_df = build_merged_dataset(use_victsd=True, spam_rules=False, save=True)
    full_df = full_df.dropna(subset=["text_raw"]).reset_index(drop=True)
    # 4. Chạy qua pipeline tiền xử lý tạo cột text_processed
    print("\n--- Đang thực thi Pipeline tiền xử lý văn bản (Unicode, Teencode, Emoji, Tách từ)... ---")
    tqdm.pandas(desc="Tiền xử lý văn bản")
    full_df["text_processed"] = full_df["text_raw"].progress_apply(preprocess_text)
    
    # Lọc bỏ các câu sau khi tiền xử lý bị rỗng
    clean_mask = full_df["text_processed"].str.strip().str.len() > 0
    full_df = full_df[clean_mask].reset_index(drop=True)
    print(f"Số lượng mẫu hợp lệ sau tiền xử lý: {len(full_df)}")
    # Loại câu TRÙNG nội dung (sau chuẩn hóa) để tránh rò rỉ dữ liệu giữa train/val/test.
    n_before = len(full_df)
    label_cols = ["toxic", "hate", "spam", "clean"]
    conflict = full_df.groupby("text_processed")[label_cols].nunique().max(axis=1) > 1
    full_df = full_df[~full_df["text_processed"].isin(conflict[conflict].index)]
    full_df = full_df.drop_duplicates(subset=["text_processed"]).reset_index(drop=True)
    print(f"Đã loại {n_before - len(full_df)} mẫu trùng/mâu thuẫn nhãn -> còn {len(full_df)}")

    # Cảnh báo nếu có nhãn không có mẫu dương nào (vd: spam) -> pos_weight sẽ chia cho 0
    for col in label_cols:
        if full_df[col].sum() == 0:
            print(f"!!! CẢNH BÁO: nhãn '{col}' không có mẫu dương nào.")
    
    # 5. Phân chia Stratified Split (80% Train, 10% Val, 10% Test)
    # Tạo nhãn phân tầng tổ hợp để đảm bảo tỷ lệ phân bổ các nhãn đồng đều
    strat_label = full_df["toxic"].astype(str) + "_" + full_df["hate"].astype(str) + "_" + full_df["clean"].astype(str)
    
    # Tách 80% Train và 20% Temp
    train_df, temp_df = train_test_split(
        full_df, 
        test_size=0.20, 
        random_state=42, 
        stratify=strat_label
    )
    
    # Tách 20% Temp thành 10% Val và 10% Test
    temp_strat = temp_df["toxic"].astype(str) + "_" + temp_df["hate"].astype(str) + "_" + temp_df["clean"].astype(str)
    val_df, test_df = train_test_split(
        temp_df, 
        test_size=0.50, 
        random_state=42, 
        stratify=temp_strat
    )
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    # 6. Lưu kết quả vào data/processed/
    os.makedirs("data/processed", exist_ok=True)
    train_path = "data/processed/train.csv"
    val_path = "data/processed/val.csv"
    test_path = "data/processed/test.csv"
    
    train_df.to_csv(train_path, index=False, encoding="utf-8")
    val_df.to_csv(val_path, index=False, encoding="utf-8")
    test_df.to_csv(test_path, index=False, encoding="utf-8")
        # Kiểm tra không rò rỉ giữa các tập
    assert not set(train_df["text_processed"]) & set(test_df["text_processed"]), "Rò rỉ train/test"
    assert not set(train_df["text_processed"]) & set(val_df["text_processed"]), "Rò rỉ train/val"
    print("\n=== KẾT QUẢ PHÂN CHIA TẬP DỮ LIỆU ===")
    print(f"- Train (80%): {len(train_df):,} mẫu -> {train_path}")
    print(f"- Val   (10%): {len(val_df):,} mẫu -> {val_path}")
    print(f"- Test  (10%): {len(test_df):,} mẫu -> {test_path}")
    
    print("\n=== TỶ LỆ PHÂN BỔ CÁC NHÃN TRÊN TỪNG TẬP ===")
    for name, df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        print(f"\nTập {name} ({len(df)} mẫu):")
        for col in ["clean", "toxic", "hate", "spam"]:
            count = df[col].sum()
            pct = count / len(df) * 100
            print(f"  + {col:<6}: {count:>5} mẫu ({pct:>6.2f}%)")
            
    print("\n=== HOÀN THÀNH TẤT CẢ CÔNG ĐOẠN PHÂN CHIA DỮ LIỆU! ===")

if __name__ == "__main__":
    create_multilabel_dataset()
