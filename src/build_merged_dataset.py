"""
Script gộp các bộ dữ liệu thô về bảng Multi-Label 4 nhãn nhị phân.
Phụ trách: Sinh viên 2 (Tuần 1 - Hạng mục 2)

Nguồn dữ liệu (data/raw/):
    - ViHSD   : train.csv, dev.csv, test.csv        (cột free_text, label_id: 0 Clean / 1 Offensive / 2 Hate)
    - ViCTSD  : ViCTSD_{train,valid,test}.csv       (cột Comment, Toxicity: 0/1)

Quy ước ánh xạ -> [toxic, hate, spam, clean]:
    ViHSD 0 -> clean=1
    ViHSD 1 -> toxic=1
    ViHSD 2 -> toxic=1, hate=1      (hate mang cả tính toxic)
    ViCTSD Toxicity=1 -> toxic=1 ; Toxicity=0 -> clean=1   (ViCTSD không có nhãn hate)
    spam: KHÔNG nguồn nào có sẵn. Mặc định = 0. Có thể bật luật heuristic bằng --spam-rules
          (chỉ mang tính gán nhãn yếu, phải rà soát thủ công mẫu trước khi dùng).
    clean = 1 khi không có nhãn độc hại nào (toxic/hate/spam đều 0).

Cách chạy (từ thư mục gốc dự án):
    python -m src.build_merged_dataset                 # chỉ ViHSD + ViCTSD, spam = 0
    python -m src.build_merged_dataset --spam-rules    # bật luật gán nhãn spam yếu
    python -m src.build_merged_dataset --no-victsd     # chỉ ViHSD
"""
from __future__ import annotations

import argparse
import os
import re

import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
LABEL_COLS = ["toxic", "hate", "spam", "clean"]

# ---- Luật spam (heuristic, độ chính xác chưa kiểm chứng) --------------------
_URL_RE = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)(?:0|\+84)\d{9,10}(?!\d)")
# Chỉ giữ từ khóa quảng cáo "mạnh". Đã loại 'mua ngay', 'khuyến mãi', 'đặt hàng', 'free ship'
# vì khớp nhầm nhiều bình luận bình thường (đã rà thử trên dữ liệu thật).
_ADS_RE = re.compile(r"(hotline|giá sỉ|tuyển (?:dụng|ctv))", re.IGNORECASE)


def is_spam_by_rules(text: str) -> bool:
    """True nếu câu chứa link, số điện thoại hoặc từ khóa quảng cáo mạnh.

    Cố ý KHÔNG dùng 'ib', 'zalo', 'liên hệ', 'mua ngay'... vì khớp nhầm nhiều câu bình thường.
    """
    return bool(_URL_RE.search(text) or _PHONE_RE.search(text) or _ADS_RE.search(text))


# ---- Đọc & ánh xạ từng nguồn -------------------------------------------------
def _read_csvs(filenames: list[str]) -> pd.DataFrame:
    parts = []
    for name in filenames:
        path = os.path.join(RAW_DIR, name)
        if not os.path.exists(path):
            print(f"- Cảnh báo: không tìm thấy {path}")
            continue
        df = pd.read_csv(path, encoding="utf-8")
        print(f"- Đã nạp {name}: {len(df)} dòng")
        parts.append(df)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def load_vihsd() -> pd.DataFrame:
    df = _read_csvs(["train.csv", "dev.csv", "test.csv"])
    if df.empty:
        return df
    df = df.dropna(subset=["free_text"]).rename(columns={"free_text": "text_raw"})
    lbl = df["label_id"]
    out = pd.DataFrame({"text_raw": df["text_raw"].astype(str)})
    out["toxic"] = lbl.isin([1, 2]).astype(int)
    out["hate"] = (lbl == 2).astype(int)
    out["spam"] = 0
    out["source"] = "ViHSD"
    return out


def load_victsd() -> pd.DataFrame:
    df = _read_csvs(["ViCTSD_train.csv", "ViCTSD_valid.csv", "ViCTSD_test.csv"])
    if df.empty:
        return df
    df = df.dropna(subset=["Comment"])
    out = pd.DataFrame({"text_raw": df["Comment"].astype(str)})
    out["toxic"] = (df["Toxicity"] == 1).astype(int)
    out["hate"] = 0
    out["spam"] = 0
    out["source"] = "ViCTSD"
    return out


def build_merged_dataset(
    use_victsd: bool = True, spam_rules: bool = False, save: bool = True
) -> pd.DataFrame:
    """Gộp các nguồn thành bảng [text_raw, toxic, hate, spam, clean, source]."""
    frames = [load_vihsd()]
    if use_victsd:
        frames.append(load_victsd())
    frames = [f for f in frames if not f.empty]
    if not frames:
        raise FileNotFoundError(f"Không tìm thấy dữ liệu thô trong {RAW_DIR}")

    df = pd.concat(frames, ignore_index=True)

    if spam_rules:
        mask = df["text_raw"].map(is_spam_by_rules)
        df.loc[mask, "spam"] = 1
        print(f"- Luật spam gán {int(mask.sum())} câu (cần rà soát thủ công!)")

    # clean = không mang nhãn độc hại nào
    df["clean"] = (df[["toxic", "hate", "spam"]].sum(axis=1) == 0).astype(int)
    df = df[["text_raw", *LABEL_COLS, "source"]]

    print(f"\nTổng số mẫu sau khi gộp: {len(df)}")
    print(df.groupby("source")[LABEL_COLS].sum().to_string())

    if save:
        out_path = os.path.join(RAW_DIR, "merged_dataset.csv")
        df.to_csv(out_path, index=False, encoding="utf-8")
        print(f"- Đã lưu {out_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--no-victsd", action="store_true", help="Không gộp ViCTSD")
    ap.add_argument("--spam-rules", action="store_true", help="Bật luật gán nhãn spam (heuristic)")
    args = ap.parse_args()
    build_merged_dataset(use_victsd=not args.no_victsd, spam_rules=args.spam_rules)