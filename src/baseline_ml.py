"""
Module Huấn luyện & Đánh giá Baseline Machine Learning (TF-IDF + Logistic Regression / LinearSVC)
Đồ án: Phân loại bình luận độc hại đa nhãn tiếng Việt (DoAnNLP-Tuan2)
Phụ trách: Sinh viên 1 (Baseline Machine Learning & Đánh giá Multi-Label)

Các nội dung triển khai:
1. Trích xuất đặc trưng văn bản bằng TfidfVectorizer (kết hợp unigram & bigram: ngram_range=(1, 2)).
2. Huấn luyện mô hình phân loại đa nhãn OneVsRestClassifier với:
   - Logistic Regression (class_weight='balanced')
   - LinearSVC (kết hợp CalibratedClassifierCV để dự đoán xác suất)
3. Tối ưu ngưỡng phân loại (Threshold Tuning) trên tập Validation (val.csv).
4. Đánh giá toàn diện trên tập Test (test.csv) bằng module src/metrics.py.
5. Vẽ ma trận nhầm lẫn đa nhãn, đường cong biến thiên F1 theo ngưỡng và biểu đồ đối sánh Benchmark.
6. Lưu trữ artifacts (vectorizer, mô hình, báo cáo JSON, ảnh assets).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC

# Bảo đảm đường dẫn gốc của project trong sys.path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.metrics import (
    DEFAULT_LABELS,
    apply_thresholds,
    evaluate_multilabel,
    find_best_thresholds,
    plot_multilabel_confusion_matrix,
    plot_threshold_tuning,
)


def load_datasets(
    data_dir: str = "data/processed",
    label_cols: list[str] = DEFAULT_LABELS,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Nạp dữ liệu train, val, test từ thư mục data/processed."""
    train_path = os.path.join(data_dir, "train.csv")
    val_path = os.path.join(data_dir, "val.csv")
    test_path = os.path.join(data_dir, "test.csv")

    for p in [train_path, val_path, test_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Không tìm thấy file dữ liệu: {p}. Hãy chạy src/split_data.py trước.")

    train_df = pd.read_csv(train_path, encoding="utf-8")
    val_df = pd.read_csv(val_path, encoding="utf-8")
    test_df = pd.read_csv(test_path, encoding="utf-8")

    print(f"- Nạp thành công Train ({len(train_df):,} mẫu), Val ({len(val_df):,} mẫu), Test ({len(test_df):,} mẫu)")
    return train_df, val_df, test_df


def build_tfidf_features(
    train_texts: pd.Series,
    val_texts: pd.Series,
    test_texts: pd.Series,
    ngram_range: Tuple[int, int] = (1, 2),
    max_features: int = 20000,
    min_df: int = 2,
    sublinear_tf: bool = True,
) -> Tuple[TfidfVectorizer, any, any, any]:
    """Xây dựng và trích xuất đặc trưng TF-IDF unigram + bigram."""
    print(f"\n--- [1] TRÍCH XUẤT ĐẶC TRƯNG TF-IDF (ngram_range={ngram_range}, max_features={max_features}) ---")
    t0 = time.time()
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=sublinear_tf,
    )
    X_train = vectorizer.fit_transform(train_texts.fillna(""))
    X_val = vectorizer.transform(val_texts.fillna(""))
    X_test = vectorizer.transform(test_texts.fillna(""))

    elapsed = time.time() - t0
    print(f"-> Hoàn thành trích xuất TF-IDF trong {elapsed:.2f}s")
    print(f"   Kích thước ma trận Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
    print(f"   Kích thước từ vựng thực tế: {len(vectorizer.vocabulary_):,} từ/cụm từ")
    return vectorizer, X_train, X_val, X_test


def train_logistic_regression(
    X_train,
    y_train: np.ndarray,
    C: float = 2.0,
    max_iter: int = 1000,
    random_state: int = 42,
) -> OneVsRestClassifier:
    """Huấn luyện OneVsRestClassifier(LogisticRegression)."""
    print(f"\n--- [2A] HUẤN LUYỆN ONE-VS-REST LOGISTIC REGRESSION (C={C}, class_weight='balanced') ---")
    t0 = time.time()
    base_clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        solver="lbfgs",
        random_state=random_state,
    )
    ovr_clf = OneVsRestClassifier(base_clf)
    ovr_clf.fit(X_train, y_train)
    print(f"-> Hoàn thành huấn luyện Logistic Regression trong {time.time() - t0:.2f}s")
    return ovr_clf


def train_linear_svc(
    X_train,
    y_train: np.ndarray,
    C: float = 1.0,
    max_iter: int = 2000,
    random_state: int = 42,
) -> OneVsRestClassifier:
    """Huấn luyện OneVsRestClassifier(LinearSVC) bọc bởi CalibratedClassifierCV để sinh xác suất."""
    print(f"\n--- [2B] HUẤN LUYỆN ONE-VS-REST LINEAR SVC (C={C}, CalibratedClassifierCV) ---")
    t0 = time.time()
    base_svc = LinearSVC(
        C=C,
        max_iter=max_iter,
        class_weight="balanced",
        random_state=random_state,
    )
    # CalibratedClassifierCV sử dụng Platt Scaling (sigmoid) trên cross-validation 3 folds
    calibrated_svc = CalibratedClassifierCV(estimator=base_svc, cv=3)
    ovr_svc = OneVsRestClassifier(calibrated_svc)
    ovr_svc.fit(X_train, y_train)
    print(f"-> Hoàn thành huấn luyện LinearSVC (Calibrated) trong {time.time() - t0:.2f}s")
    return ovr_svc


def plot_benchmark_barchart(
    benchmark_data: Dict[str, Dict],
    save_path: str = "assets/baseline_benchmark_comparison.png",
) -> plt.Figure:
    """Vẽ biểu đồ cột đối sánh Macro-F1, Toxic-F1, Hate-F1 và Clean-F1 giữa các mô hình Baseline."""
    models = list(benchmark_data.keys())
    metrics_to_plot = ["Macro-F1", "Toxic-F1", "Hate-F1", "Clean-F1", "Subset Acc"]

    data_matrix = []
    for m in models:
        row = [
            benchmark_data[m].get("macro_f1", 0.0),
            benchmark_data[m].get("toxic_f1", 0.0),
            benchmark_data[m].get("hate_f1", 0.0),
            benchmark_data[m].get("clean_f1", 0.0),
            benchmark_data[m].get("exact_match", 0.0),
        ]
        data_matrix.append(row)

    data_matrix = np.array(data_matrix)
    x = np.arange(len(metrics_to_plot))
    width = 0.16

    fig, ax = plt.subplots(figsize=(13, 7))
    colors = ["#3498db", "#2980b9", "#e67e22", "#d35400", "#27ae60"]

    for i, m in enumerate(models):
        offset = (i - len(models) / 2 + 0.5) * width
        rects = ax.bar(x + offset, data_matrix[i], width, label=m, color=colors[i % len(colors)], alpha=0.9)
        # Thêm nhãn giá trị trên mỗi cột
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

    ax.set_title("BẢNG ĐỐI SÁNH HIỆU NĂNG CÁC MÔ HÌNH BASELINE TUẦN 2 (TẬP TEST)", fontsize=13, weight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot, fontsize=11, weight="bold")
    ax.set_ylabel("Điểm số (0.0 -> 1.0)", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"-> Đã lưu biểu đồ đối sánh Benchmark tại: {save_path}")
    return fig


def run_baseline_pipeline(
    data_dir: str = "data/processed",
    saved_models_dir: str = "saved_models",
    assets_dir: str = "assets",
    max_features: int = 20000,
) -> Dict:
    """Quy trình tự động hóa toàn diện Baseline ML."""
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(assets_dir, exist_ok=True)

    # 1. Nạp dữ liệu
    train_df, val_df, test_df = load_datasets(data_dir=data_dir)
    label_cols = DEFAULT_LABELS

    y_train = train_df[label_cols].values.astype(int)
    y_val = val_df[label_cols].values.astype(int)
    y_test = test_df[label_cols].values.astype(int)

    # 2. TF-IDF
    vectorizer, X_train, X_val, X_test = build_tfidf_features(
        train_df["text_processed"],
        val_df["text_processed"],
        test_df["text_processed"],
        ngram_range=(1, 2),
        max_features=max_features,
    )
    joblib.dump(vectorizer, os.path.join(saved_models_dir, "tfidf_vectorizer.joblib"))

    # 3. Logistic Regression
    clf_lr = train_logistic_regression(X_train, y_train)
    joblib.dump(clf_lr, os.path.join(saved_models_dir, "baseline_logreg.joblib"))

    # 4. LinearSVC
    clf_svc = train_linear_svc(X_train, y_train)
    joblib.dump(clf_svc, os.path.join(saved_models_dir, "baseline_linearsvc.joblib"))

    # 5. Dự đoán xác suất trên tập Validation & Threshold Tuning
    print("\n" + "=" * 60)
    print(" [3] TỐI ƯU HÓA NGƯỠNG PHÂN LOẠI TRÊN TẬP VALIDATION")
    print("=" * 60)
    val_probs_lr = clf_lr.predict_proba(X_val)
    val_probs_svc = clf_svc.predict_proba(X_val)

    print("\n>>> Tối ưu ngưỡng Logistic Regression:")
    tune_lr = find_best_thresholds(y_val, val_probs_lr, label_cols=label_cols, verbose=True)

    print("\n>>> Tối ưu ngưỡng LinearSVC:")
    tune_svc = find_best_thresholds(y_val, val_probs_svc, label_cols=label_cols, verbose=True)

    # Vẽ biểu đồ tối ưu ngưỡng
    plot_threshold_tuning(
        tune_lr["search_history"],
        tune_lr["best_thresholds"],
        save_path=os.path.join(assets_dir, "ml_threshold_tuning.png"),
    )

    # 6. Đánh giá trên tập Test
    print("\n" + "=" * 60)
    print(" [4] ĐÁNH GIÁ CÁC MÔ HÌNH TRÊN TẬP KIỂM THỬ (TEST SET)")
    print("=" * 60)
    test_probs_lr = clf_lr.predict_proba(X_test)
    test_probs_svc = clf_svc.predict_proba(X_test)

    # Mô hình 1: Logistic Regression (ngưỡng 0.5)
    print("\n>>> 1. TF-IDF + Logistic Regression (Ngưỡng mặc định 0.50):")
    preds_lr_def = apply_thresholds(test_probs_lr, 0.5, label_cols)
    eval_lr_def = evaluate_multilabel(y_test, preds_lr_def, label_cols=label_cols, verbose=True)

    # Mô hình 2: Logistic Regression (ngưỡng tối ưu)
    print(f"\n>>> 2. TF-IDF + Logistic Regression (Ngưỡng tối ưu {tune_lr['best_thresholds']}):")
    preds_lr_opt = apply_thresholds(test_probs_lr, tune_lr["best_thresholds"], label_cols)
    eval_lr_opt = evaluate_multilabel(y_test, preds_lr_opt, label_cols=label_cols, verbose=True)
    plot_multilabel_confusion_matrix(
        y_test,
        preds_lr_opt,
        label_cols=label_cols,
        title="Ma Trận Nhầm Lẫn - TF-IDF + Logistic Regression (Tuned Thresholds)",
        save_path=os.path.join(assets_dir, "ml_logreg_confusion_matrix.png"),
    )

    # Mô hình 3: LinearSVC (ngưỡng 0.5)
    print("\n>>> 3. TF-IDF + LinearSVC (Ngưỡng mặc định 0.50):")
    preds_svc_def = apply_thresholds(test_probs_svc, 0.5, label_cols)
    eval_svc_def = evaluate_multilabel(y_test, preds_svc_def, label_cols=label_cols, verbose=True)

    # Mô hình 4: LinearSVC (ngưỡng tối ưu)
    print(f"\n>>> 4. TF-IDF + LinearSVC (Ngưỡng tối ưu {tune_svc['best_thresholds']}):")
    preds_svc_opt = apply_thresholds(test_probs_svc, tune_svc["best_thresholds"], label_cols)
    eval_svc_opt = evaluate_multilabel(y_test, preds_svc_opt, label_cols=label_cols, verbose=True)
    plot_multilabel_confusion_matrix(
        y_test,
        preds_svc_opt,
        label_cols=label_cols,
        title="Ma Trận Nhầm Lẫn - TF-IDF + LinearSVC (Tuned Thresholds)",
        save_path=os.path.join(assets_dir, "ml_linearsvc_confusion_matrix.png"),
    )

    # Kết quả BiLSTM từ SV2 (lấy từ thực nghiệm Notebook 03)
    bilstm_results = {
        "macro_f1": 0.5061,
        "active_macro_f1": 0.6748,
        "toxic_f1": 0.6000,
        "hate_f1": 0.5100,
        "spam_f1": 0.0000,
        "clean_f1": 0.9100,
        "exact_match": 0.8250,
        "hamming_loss": 0.0890,
    }

    # Tổng hợp bảng Benchmark
    benchmark_table = {
        "TF-IDF + LogReg (th=0.5)": {
            "macro_f1": eval_lr_def["macro_avg"]["f1_score"],
            "active_macro_f1": eval_lr_def["active_macro_f1"],
            "toxic_f1": eval_lr_def["per_label"]["toxic"]["f1_score"],
            "hate_f1": eval_lr_def["per_label"]["hate"]["f1_score"],
            "spam_f1": eval_lr_def["per_label"]["spam"]["f1_score"],
            "clean_f1": eval_lr_def["per_label"]["clean"]["f1_score"],
            "exact_match": eval_lr_def["exact_match_ratio"],
            "hamming_loss": eval_lr_def["hamming_loss"],
        },
        "TF-IDF + LogReg (Tuned)": {
            "macro_f1": eval_lr_opt["macro_avg"]["f1_score"],
            "active_macro_f1": eval_lr_opt["active_macro_f1"],
            "toxic_f1": eval_lr_opt["per_label"]["toxic"]["f1_score"],
            "hate_f1": eval_lr_opt["per_label"]["hate"]["f1_score"],
            "spam_f1": eval_lr_opt["per_label"]["spam"]["f1_score"],
            "clean_f1": eval_lr_opt["per_label"]["clean"]["f1_score"],
            "exact_match": eval_lr_opt["exact_match_ratio"],
            "hamming_loss": eval_lr_opt["hamming_loss"],
        },
        "TF-IDF + LinearSVC (th=0.5)": {
            "macro_f1": eval_svc_def["macro_avg"]["f1_score"],
            "active_macro_f1": eval_svc_def["active_macro_f1"],
            "toxic_f1": eval_svc_def["per_label"]["toxic"]["f1_score"],
            "hate_f1": eval_svc_def["per_label"]["hate"]["f1_score"],
            "spam_f1": eval_svc_def["per_label"]["spam"]["f1_score"],
            "clean_f1": eval_svc_def["per_label"]["clean"]["f1_score"],
            "exact_match": eval_svc_def["exact_match_ratio"],
            "hamming_loss": eval_svc_def["hamming_loss"],
        },
        "TF-IDF + LinearSVC (Tuned)": {
            "macro_f1": eval_svc_opt["macro_avg"]["f1_score"],
            "active_macro_f1": eval_svc_opt["active_macro_f1"],
            "toxic_f1": eval_svc_opt["per_label"]["toxic"]["f1_score"],
            "hate_f1": eval_svc_opt["per_label"]["hate"]["f1_score"],
            "spam_f1": eval_svc_opt["per_label"]["spam"]["f1_score"],
            "clean_f1": eval_svc_opt["per_label"]["clean"]["f1_score"],
            "exact_match": eval_svc_opt["exact_match_ratio"],
            "hamming_loss": eval_svc_opt["hamming_loss"],
        },
        "BiLSTM + Focal Loss (SV2)": bilstm_results,
    }

    # Vẽ biểu đồ đối sánh Benchmark
    plot_benchmark_barchart(
        benchmark_table,
        save_path=os.path.join(assets_dir, "baseline_benchmark_comparison.png"),
    )

    # In bảng Benchmark Markdown
    print("\n" + "=" * 90)
    print(" BẢNG TỔNG HỢP ĐỐI SÁNH KẾT QUẢ CÁC MÔ HÌNH BASELINE TUẦN 2 (TEST SET)")
    print("=" * 90)
    print(f"| {'Mô hình Baseline':<28} | {'Macro-F1 (4 lớp)':<16} | {'Active Macro-F1':<16} | {'Toxic-F1':<10} | {'Hate-F1':<9} | {'Clean-F1':<10} | {'Subset Acc':<10} |")
    print(f"|{'-'*30}|{'-'*18}|{'-'*18}|{'-'*12}|{'-'*11}|{'-'*12}|{'-'*12}|")
    for name, m in benchmark_table.items():
        print(f"| {name:<28} | {m['macro_f1']:<16.4f} | {m['active_macro_f1']:<16.4f} | {m['toxic_f1']:<10.4f} | {m['hate_f1']:<9.4f} | {m['clean_f1']:<10.4f} | {m['exact_match']*100:<9.2f}% |")

    # Lưu kết quả ra file JSON
    json_path = os.path.join(saved_models_dir, "baseline_benchmark_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "tuning_lr": tune_lr,
                "tuning_svc": tune_svc,
                "benchmark_table": benchmark_table,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"\n-> Đã lưu toàn bộ số liệu Benchmark tại: {json_path}")

    return benchmark_table


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện và đánh giá Baseline ML")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Thư mục dữ liệu")
    parser.add_argument("--saved-models-dir", type=str, default="saved_models", help="Thư mục lưu weights/model")
    parser.add_argument("--assets-dir", type=str, default="assets", help="Thư mục lưu hình ảnh trực quan")
    parser.add_argument("--max-features", type=int, default=20000, help="Số lượng đặc trưng TF-IDF tối đa")
    args = parser.parse_args()

    run_baseline_pipeline(
        data_dir=args.data_dir,
        saved_models_dir=args.saved_models_dir,
        assets_dir=args.assets_dir,
        max_features=args.max_features,
    )
