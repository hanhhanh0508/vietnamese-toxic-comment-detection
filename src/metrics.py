"""
Module tính toán chỉ số đánh giá đa nhãn (Multi-Label Metrics) & Tối ưu ngưỡng (Threshold Tuning)
Phụ trách: Sinh viên 1 (DoAnNLP-Tuan2)

Các chức năng chính:
1. evaluate_multilabel(): Tính toán chuẩn xác Macro-F1, Micro-F1, Precision, Recall cho từng nhãn
   và toàn cục (Macro/Micro/Weighted), cùng Exact Match Ratio và Hamming Loss.
2. find_best_thresholds(): Thử nghiệm điều chỉnh ngưỡng quyết định nhãn (0.3 đến 0.7) trên tập val
   để tối ưu Macro-F1 cho từng nhãn (đặc biệt nhãn hiếm như hate).
3. apply_thresholds(): Chuyển ma trận xác suất thành nhãn nhị phân theo vector ngưỡng tùy biến.
4. plot_multilabel_confusion_matrix(): Trực quan hóa ma trận nhầm lẫn cho từng nhãn đa mục tiêu.
5. plot_threshold_tuning(): Vẽ đường cong F1-score theo ngưỡng để khảo sát điểm cực trị.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)

DEFAULT_LABELS = ["toxic", "hate", "spam", "clean"]


def evaluate_multilabel(
    y_true: Union[np.ndarray, pd.DataFrame, list],
    y_pred: Union[np.ndarray, pd.DataFrame, list],
    label_cols: Optional[List[str]] = None,
    zero_division: int = 0,
    verbose: bool = True,
) -> Dict:
    """Tính toán toàn diện các chỉ số đánh giá cho bài toán phân loại đa nhãn.

    Args:
        y_true: Mảng nhãn thực tế nhị phân (N, num_labels).
        y_pred: Mảng nhãn dự đoán nhị phân (N, num_labels).
        label_cols: Danh sách tên nhãn (mặc định: toxic, hate, spam, clean).
        zero_division: Giá trị gán khi chia cho 0 (mặc định: 0).
        verbose: In báo cáo ra màn hình nếu True.

    Returns:
        dict: Chứa chi tiết per_label, macro_avg, micro_avg, weighted_avg,
              exact_match_ratio, hamming_loss, report_str.
    """
    if label_cols is None:
        label_cols = DEFAULT_LABELS

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if y_t.ndim == 1:
        y_t = y_t.reshape(-1, 1)
    if y_p.ndim == 1:
        y_p = y_p.reshape(-1, 1)

    n_labels = len(label_cols)
    per_label_metrics = {}

    for i, col in enumerate(label_cols):
        yt_i = y_t[:, i]
        yp_i = y_p[:, i]
        p = float(precision_score(yt_i, yp_i, zero_division=zero_division))
        r = float(recall_score(yt_i, yp_i, zero_division=zero_division))
        f1 = float(f1_score(yt_i, yp_i, zero_division=zero_division))
        supp = int(np.sum(yt_i))
        per_label_metrics[col] = {
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1_score": round(f1, 4),
            "support": supp,
        }

    # Tính Macro, Micro, Weighted averages
    macro_p = float(precision_score(y_t, y_p, average="macro", zero_division=zero_division))
    macro_r = float(recall_score(y_t, y_p, average="macro", zero_division=zero_division))
    macro_f1 = float(f1_score(y_t, y_p, average="macro", zero_division=zero_division))

    micro_p = float(precision_score(y_t, y_p, average="micro", zero_division=zero_division))
    micro_r = float(recall_score(y_t, y_p, average="micro", zero_division=zero_division))
    micro_f1 = float(f1_score(y_t, y_p, average="micro", zero_division=zero_division))

    weighted_p = float(precision_score(y_t, y_p, average="weighted", zero_division=zero_division))
    weighted_r = float(recall_score(y_t, y_p, average="weighted", zero_division=zero_division))
    weighted_f1 = float(f1_score(y_t, y_p, average="weighted", zero_division=zero_division))

    # Exact Match Ratio (Subset Accuracy) & Hamming Loss
    exact_match = float(np.mean(np.all(y_t == y_p, axis=1)))
    h_loss = float(hamming_loss(y_t, y_p))

    # Macro-F1 trên các nhãn có mẫu dương (active classes, loại trừ spam nếu support=0)
    active_labels = [c for c in label_cols if per_label_metrics[c]["support"] > 0]
    if active_labels:
        active_f1_scores = [per_label_metrics[c]["f1_score"] for c in active_labels]
        active_macro_f1 = float(np.mean(active_f1_scores))
    else:
        active_macro_f1 = macro_f1

    report_str = classification_report(
        y_t,
        y_p,
        target_names=label_cols,
        zero_division=zero_division,
        digits=4,
    )

    results = {
        "per_label": per_label_metrics,
        "macro_avg": {
            "precision": round(macro_p, 4),
            "recall": round(macro_r, 4),
            "f1_score": round(macro_f1, 4),
        },
        "active_macro_f1": round(active_macro_f1, 4),
        "micro_avg": {
            "precision": round(micro_p, 4),
            "recall": round(micro_r, 4),
            "f1_score": round(micro_f1, 4),
        },
        "weighted_avg": {
            "precision": round(weighted_p, 4),
            "recall": round(weighted_r, 4),
            "f1_score": round(weighted_f1, 4),
        },
        "exact_match_ratio": round(exact_match, 4),
        "hamming_loss": round(h_loss, 4),
        "classification_report": report_str,
    }

    if verbose:
        print("=== BÁO CÁO ĐÁNH GIÁ ĐA NHÃN (MULTI-LABEL EVALUATION) ===")
        print(report_str)
        print(f"Exact Match Ratio (Subset Accuracy): {exact_match:.4f} ({exact_match*100:.2f}%)")
        print(f"Hamming Loss: {h_loss:.4f}")
        print(f"Macro-F1 (All 4 classes): {macro_f1:.4f}")
        if len(active_labels) < n_labels:
            print(f"Macro-F1 (Active classes {active_labels}): {active_macro_f1:.4f}")

    return results


def apply_thresholds(
    y_probs: np.ndarray,
    thresholds: Union[Dict[str, float], List[float], np.ndarray, float] = 0.5,
    label_cols: Optional[List[str]] = None,
) -> np.ndarray:
    """Chuyển ma trận xác suất thành dự đoán nhị phân dựa trên vector ngưỡng.

    Args:
        y_probs: Ma trận xác suất dạng (N, num_labels).
        thresholds: float, list, hoặc dict ánh xạ {tên_nhãn: ngưỡng}.
        label_cols: Danh sách tên nhãn tương ứng các cột của y_probs.

    Returns:
        np.ndarray: Ma trận nhãn nhị phân 0/1 cùng kích thước.
    """
    if label_cols is None:
        label_cols = DEFAULT_LABELS

    probs = np.asarray(y_probs)
    n_samples, n_labels = probs.shape
    y_pred = np.zeros_like(probs, dtype=int)

    if isinstance(thresholds, (int, float)):
        y_pred = (probs >= thresholds).astype(int)
    elif isinstance(thresholds, dict):
        for i, col in enumerate(label_cols):
            th = thresholds.get(col, 0.5)
            y_pred[:, i] = (probs[:, i] >= th).astype(int)
    elif isinstance(thresholds, (list, np.ndarray)):
        th_arr = np.asarray(thresholds)
        for i in range(n_labels):
            th = th_arr[i] if i < len(th_arr) else 0.5
            y_pred[:, i] = (probs[:, i] >= th).astype(int)
    else:
        raise ValueError(f"Định dạng thresholds không hợp lệ: {type(thresholds)}")

    return y_pred


def find_best_thresholds(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_probs: Union[np.ndarray, pd.DataFrame],
    label_cols: Optional[List[str]] = None,
    threshold_range: Optional[np.ndarray] = None,
    default_threshold: float = 0.5,
    verbose: bool = True,
) -> Dict:
    """Tìm kiếm ngưỡng xác suất tối ưu trên dải [0.3, 0.7] để tối đa hóa Macro-F1 từng nhãn.

    Args:
        y_true: Ma trận nhãn thực tế nhị phân (N, num_labels).
        y_probs: Ma trận xác suất dự đoán (N, num_labels).
        label_cols: Danh sách nhãn.
        threshold_range: Mảng các giá trị ngưỡng cần duyệt (mặc định: 0.30 -> 0.70 bước 0.05).
        default_threshold: Ngưỡng mặc định để so sánh (mặc định: 0.5).
        verbose: In tiến trình nếu True.

    Returns:
        dict: Chứa best_thresholds, default_f1_per_label, best_f1_per_label,
              default_macro_f1, best_macro_f1, search_history.
    """
    if label_cols is None:
        label_cols = DEFAULT_LABELS

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_probs, dtype=float)

    if threshold_range is None:
        threshold_range = np.round(np.arange(0.30, 0.75, 0.05), 2)

    best_thresholds: Dict[str, float] = {}
    best_f1_per_label: Dict[str, float] = {}
    default_f1_per_label: Dict[str, float] = {}
    search_history: Dict[str, Dict[float, float]] = {}

    for i, col in enumerate(label_cols):
        yt_col = y_t[:, i]
        yp_col = y_p[:, i]
        n_pos = int(np.sum(yt_col))

        history = {}
        # Nếu nhãn không có mẫu dương nào (như spam), giữ ngưỡng mặc định
        if n_pos == 0:
            best_thresholds[col] = default_threshold
            best_f1_per_label[col] = 0.0
            default_f1_per_label[col] = 0.0
            for th in threshold_range:
                history[round(float(th), 2)] = 0.0
            search_history[col] = history
            continue

        # Điểm F1 ở ngưỡng mặc định
        def_preds = (yp_col >= default_threshold).astype(int)
        def_f1 = float(f1_score(yt_col, def_preds, zero_division=0))
        default_f1_per_label[col] = round(def_f1, 4)

        best_th = default_threshold
        best_f1 = def_f1

        for th in threshold_range:
            th_f = round(float(th), 2)
            preds = (yp_col >= th_f).astype(int)
            score = float(f1_score(yt_col, preds, zero_division=0))
            history[th_f] = round(score, 4)

            # Chọn ngưỡng tốt nhất (nếu bằng điểm, ưu tiên ngưỡng gần 0.5)
            if score > best_f1 or (
                np.isclose(score, best_f1, atol=1e-5)
                and abs(th_f - 0.5) < abs(best_th - 0.5)
            ):
                best_f1 = score
                best_th = th_f

        best_thresholds[col] = best_th
        best_f1_per_label[col] = round(best_f1, 4)
        search_history[col] = history

    # Tính Macro-F1 trước và sau tối ưu
    def_preds_all = apply_thresholds(y_p, default_threshold, label_cols)
    opt_preds_all = apply_thresholds(y_p, best_thresholds, label_cols)

    default_macro_f1 = float(f1_score(y_t, def_preds_all, average="macro", zero_division=0))
    best_macro_f1 = float(f1_score(y_t, opt_preds_all, average="macro", zero_division=0))

    if verbose:
        print("=== KẾT QUẢ TỐI ƯU HÓA NGƯỠNG PHÂN LOẠI (THRESHOLD TUNING) ===")
        print(f"Dải ngưỡng khảo sát: [{threshold_range[0]} ... {threshold_range[-1]}]")
        print(f"{'Nhãn':<8} | {'Ngưỡng Mặc Định':<16} | {'F1 Mặc Định':<12} | {'Ngưỡng Tối Ưu':<14} | {'F1 Tối Ưu':<10} | {'Độ Tăng (Δ)':<10}")
        print("-" * 80)
        for col in label_cols:
            def_th = default_threshold
            def_f1 = default_f1_per_label[col]
            opt_th = best_thresholds[col]
            opt_f1 = best_f1_per_label[col]
            diff = opt_f1 - def_f1
            diff_str = f"+{diff:.4f}" if diff >= 0 else f"{diff:.4f}"
            print(f"{col:<8} | {def_th:<16.2f} | {def_f1:<12.4f} | {opt_th:<14.2f} | {opt_f1:<10.4f} | {diff_str:<10}")
        print("-" * 80)
        print(f"Macro-F1 Toàn cục: Mặc định (0.50) = {default_macro_f1:.4f} -> Tối ưu = {best_macro_f1:.4f} (Δ = +{best_macro_f1 - default_macro_f1:.4f})")

    return {
        "best_thresholds": best_thresholds,
        "default_f1_per_label": default_f1_per_label,
        "best_f1_per_label": best_f1_per_label,
        "default_macro_f1": round(default_macro_f1, 4),
        "best_macro_f1": round(best_macro_f1, 4),
        "search_history": search_history,
    }


def plot_multilabel_confusion_matrix(
    y_true: Union[np.ndarray, pd.DataFrame],
    y_pred: Union[np.ndarray, pd.DataFrame],
    label_cols: Optional[List[str]] = None,
    title: str = "Multi-Label Confusion Matrices",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 10),
) -> plt.Figure:
    """Vẽ ma trận nhầm lẫn nhị phân 2x2 cho từng nhãn trong bài toán Multi-Label.

    Args:
        y_true: Ma trận nhãn thực tế (N, num_labels).
        y_pred: Ma trận nhãn dự đoán (N, num_labels).
        label_cols: Danh sách tên nhãn.
        title: Tiêu đề tổng của biểu đồ.
        save_path: Đường dẫn lưu ảnh PNG/PDF.
        figsize: Kích thước khung hình.

    Returns:
        plt.Figure: Figure đối tượng matplotlib.
    """
    if label_cols is None:
        label_cols = DEFAULT_LABELS

    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    n_labels = len(label_cols)
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()

    colors = ["Blues", "Reds", "Purples", "Greens"]

    for i, col in enumerate(label_cols):
        ax = axes[i]
        yt_i = y_t[:, i]
        yp_i = y_p[:, i]

        cm = confusion_matrix(yt_i, yp_i, labels=[0, 1])
        # [[TN, FP], [FN, TP]]
        tn, fp, fn, tp = cm.ravel()
        total = max(len(yt_i), 1)

        labels_text = np.array([
            [f"TN\n{tn:,}\n({tn/total*100:.1f}%)", f"FP\n{fp:,}\n({fp/total*100:.1f}%)"],
            [f"FN\n{fn:,}\n({fn/total*100:.1f}%)", f"TP\n{tp:,}\n({tp/total*100:.1f}%)"],
        ])

        sns.heatmap(
            cm,
            annot=labels_text,
            fmt="",
            cmap=colors[i % len(colors)],
            cbar=False,
            ax=ax,
            annot_kws={"size": 11, "weight": "bold"},
            linewidths=1.0,
            linecolor="white",
        )

        ax.set_title(f"Nhãn: {col.upper()} (Dương: {tp+fn:,} mẫu)", fontsize=13, weight="bold", pad=8)
        ax.set_xlabel("Nhãn Dự Đoán", fontsize=11)
        ax.set_ylabel("Nhãn Thực Tế", fontsize=11)
        ax.set_xticklabels(["0 (Âm)", "1 (Dương)"], fontsize=10)
        ax.set_yticklabels(["0 (Âm)", "1 (Dương)"], fontsize=10)

    plt.suptitle(title, fontsize=15, weight="bold", y=0.99)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"-> Đã lưu ma trận nhầm lẫn tại: {save_path}")

    return fig


def plot_threshold_tuning(
    search_history: Dict[str, Dict[float, float]],
    best_thresholds: Dict[str, float],
    label_cols: Optional[List[str]] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 7),
) -> plt.Figure:
    """Vẽ biểu đồ đường cong F1-score biến thiên theo dải ngưỡng phân loại cho từng nhãn.

    Args:
        search_history: Dict ánh xạ {nhãn: {ngưỡng: f1_score}}.
        best_thresholds: Dict ánh xạ {nhãn: ngưỡng_tối_ưu}.
        label_cols: Danh sách nhãn cần vẽ.
        save_path: Đường dẫn lưu ảnh.
        figsize: Kích thước biểu đồ.

    Returns:
        plt.Figure: Figure đối tượng matplotlib.
    """
    if label_cols is None:
        label_cols = [c for c in search_history.keys() if c != "spam"]

    fig, ax = plt.subplots(figsize=figsize)
    palette = {"toxic": "#e74c3c", "hate": "#8e44ad", "clean": "#27ae60", "spam": "#f39c12"}

    for col in label_cols:
        hist = search_history.get(col, {})
        if not hist:
            continue
        ths = list(hist.keys())
        f1s = list(hist.values())
        best_th = best_thresholds.get(col, 0.5)
        best_f1 = hist.get(best_th, 0.0)

        color = palette.get(col, "#34495e")
        ax.plot(ths, f1s, marker="o", label=f"{col} (Tối ưu: th={best_th:.2f}, F1={best_f1:.4f})", color=color, linewidth=2)
        ax.scatter([best_th], [best_f1], color=color, s=120, edgecolors="black", zorder=5)

    ax.axvline(x=0.5, color="gray", linestyle="--", alpha=0.7, label="Ngưỡng mặc định (0.50)")
    ax.set_title("Biến thiên F1-Score theo Ngưỡng Phân Loại (Threshold Tuning)", fontsize=14, weight="bold", pad=12)
    ax.set_xlabel("Ngưỡng Quyết Định Xác Suất (Threshold)", fontsize=12)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_xlim(0.28, 0.72)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="best", fontsize=10)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"-> Đã lưu biểu đồ tối ưu ngưỡng tại: {save_path}")

    return fig
