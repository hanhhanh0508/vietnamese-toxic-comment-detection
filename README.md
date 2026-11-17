# Hệ Thống Tự Động Phát Hiện Bình Luận Độc Hại Đa Nhãn (Multi-Label Toxic Comment Detection) Trên Mạng Xã Hội Tiếng Việt

> **Đồ án môn học / Nghiên cứu NLP (Nhóm 2 sinh viên)**  
> **Mô hình cốt lõi:** PhoBERT (`vinai/phobert-base-v2`) & ViDeBERTa (`nguyenvulebinh/vi-deberta-xsmall`) kết hợp Ensemble Soft Voting  
> **Hàm mất mát tối ưu:** Binary Focal Loss ($\gamma = 2.0$) kết hợp Pos-Weights giải quyết mất cân bằng lớp  
> **Ứng dụng giao diện:** Web Demo tương tác thời gian thực bằng Streamlit và Explainable AI (Highlight từ ngữ vi phạm)

---

## 📚 Hệ Thống Tài Liệu Kỹ Thuật (Documentation)

Toàn bộ hệ thống tài liệu kiến trúc, đặc tả yêu cầu, hướng dẫn kỹ thuật và phân công công việc đã được xây dựng hoàn chỉnh tại thư mục [`project-docs/`](project-docs/README.md).

### Các phân hệ tài liệu chính:
1. **[01. Tổng quan dự án](project-docs/01-tong-quan-du-an/tong-quan.md)**
   * [Tổng quan đề tài & Mục tiêu](project-docs/01-tong-quan-du-an/tong-quan.md)
   * [Vấn đề đặc thù tiếng Việt & Giải pháp](project-docs/01-tong-quan-du-an/van-de-va-giai-phap.md)
2. **[02. Yêu cầu & Phạm vi](project-docs/02-yeu-cau-va-pham-vi/yeu-cau-he-thong.md)**
   * [Đặc tả yêu cầu hệ thống (Chức năng & Phi chức năng)](project-docs/02-yeu-cau-va-pham-vi/yeu-cau-he-thong.md)
   * [Phạm vi nâng cấp 6 tuần (NLP Level-Up)](project-docs/02-yeu-cau-va-pham-vi/pham-vi-va-nang-cap.md)
3. **[03. Kiến trúc & Thiết kế](project-docs/03-kien-truc-va-thiet-ke/kien-truc-tong-the.md)**
   * [Kiến trúc tổng thể toàn tuyến (End-to-End Pipeline)](project-docs/03-kien-truc-va-thiet-ke/kien-truc-tong-the.md)
   * [Thiết kế module tiền xử lý văn bản](project-docs/03-kien-truc-va-thiet-ke/pipeline-tien-xu-ly.md)
   * [Thiết kế dữ liệu đa nhãn 4 classes (ViHSD mapping)](project-docs/03-kien-truc-va-thiet-ke/thiet-ke-du-lieu-multi-label.md)
   * [Thiết kế mô hình Transformer, Ensemble & Focal Loss](project-docs/03-kien-truc-va-thiet-ke/thiet-ke-mo-hinh-va-loss.md)
4. **[04. Kế hoạch & Phân công](project-docs/04-ke-hoach-va-phan-cong/lo-trinh-6-tuan.md)**
   * [Lộ trình hành động chi tiết 6 tuần](project-docs/04-ke-hoach-va-phan-cong/lo-trinh-6-tuan.md)
   * [Bảng phân chia công việc 50-50 Tuần 1 (SV1 & SV2)](project-docs/04-ke-hoach-va-phan-cong/phan-cong-tuan-1.md)
   * [Bảng phân chia công việc 50-50 Tuần 2 & Benchmark](project-docs/04-ke-hoach-va-phan-cong/phan-cong-tuan-2.md)
   * [Kế hoạch và Báo cáo tuần 2 (DoAnNLP-Tuan2.md)](DoAnNLP-Tuan2.md)
   * [Quy trình Git workflow & Chiến lược sao lưu Google Drive](project-docs/04-ke-hoach-va-phan-cong/quy-trinh-git-va-backup.md)
5. **[05. Hướng dẫn kỹ thuật](project-docs/05-huong-dan-ky-thuat/cai-dat-moi-truong.md)**
   * [Hướng dẫn cài đặt môi trường virtualenv & requirements.txt](project-docs/05-huong-dan-ky-thuat/cai-dat-moi-truong.md)
   * [Cẩm nang xử lý sự cố (Lỗi 403 Hugging Face, UTF-8 Mojibake)](project-docs/05-huong-dan-ky-thuat/xu-ly-su-co-thuong-gap.md)
   * [Hướng dẫn phát triển Web Demo Streamlit & XAI](project-docs/05-huong-dan-ky-thuat/huong-dan-web-demo-xai.md)
6. **[06. Theo dõi & Đánh giá](project-docs/06-theo-doi-va-danh-gia/quyet-dinh-quan-trong.md)**
   * [Nhật ký các quyết định kiến trúc then chốt (ADR)](project-docs/06-theo-doi-va-danh-gia/quyet-dinh-quan-trong.md)
   * [Danh sách vấn đề tồn đọng & TODO cần xác minh](project-docs/06-theo-doi-va-danh-gia/van-de-ton-dong-va-todo.md)

---

## 📊 Tiến Độ Triển Khai (Progress Tracker)

### Tuần 1: Khảo sát, Tiền xử lý & Phân chia dữ liệu (Hoàn thành)
- [x] Xây dựng từ điển Teencode tiếng Việt 220+ từ (`data/dictionaries/teencode.json`).
- [x] Xây dựng từ điển Emoji tiếng Việt (`data/dictionaries/emoji.json`).
- [x] Phát triển module tiền xử lý văn bản `src/preprocessing.py` (Unicode NFC, regex URL/mentions, rút gọn ký tự lặp, teencode, emoji, tách từ tiếng Việt).
- [x] Hợp nhất dữ liệu và thực hiện Stratified Split 80/10/10 cân đối tỷ lệ phân bổ các nhãn độc hại:
  * `data/processed/train.csv` (31,980 mẫu - 80%)
  * `data/processed/val.csv` (3,998 mẫu - 10%)
  * `data/processed/test.csv` (3,998 mẫu - 10%)
- [x] Phân tích EDA độ dài câu & token trong `notebooks/01_eda_and_preprocessing.ipynb`:
  * Quyết định chốt tham số tối ưu `MAX_LENGTH = 128` (bao phủ >97.5% bình luận, tiết kiệm 4 lần VRAM GPU).

### Tuần 2: Xây Dựng Baseline & Cài Đặt Custom Loss (Hoàn thành)
- [x] Phát triển module đánh giá đa nhãn `src/metrics.py`: hàm `evaluate_multilabel()`, `find_best_thresholds()`, `plot_multilabel_confusion_matrix()`.
- [x] Xây dựng mô hình Baseline Machine Learning `src/baseline_ml.py` và `notebooks/02_baseline_ml.ipynb`:
  * TF-IDF (unigram + bigram, 20,000 features, sublinear TF).
  * OneVsRest Logistic Regression (`class_weight='balanced'`).
  * OneVsRest LinearSVC (kết hợp `CalibratedClassifierCV` 3-fold).
- [x] Xây dựng mô hình Baseline Deep Learning `src/models.py`, `src/train_bilstm.py` và `notebooks/03_baseline_bilstm.ipynb`:
  * Mạng BiLSTM 2 tầng + Self-Attention Layer.
  * Cài đặt `BinaryFocalLoss` ($\gamma=2.0$) kết hợp vector trọng số dương `pos_weight`.
- [x] Thử nghiệm tối ưu hóa ngưỡng phân loại (Threshold Tuning) trên tập Validation (`val.csv`):
  * Cải thiện vượt bậc F1 nhãn `hate` (+8.49% trên LinearSVC) và nhãn `toxic`.
- [x] Tổng hợp bảng điểm chuẩn Benchmark trên tập Test (`test.csv` - 3,998 mẫu):
  * **TF-IDF + Logistic Regression (Tuned)** đạt Macro-F1 **0.5322** (Active Macro-F1 **0.7096**), tạm dẫn đầu mốc chuẩn Baseline trước khi bước vào Tuần 3 (PhoBERT).

| Mô hình Baseline | Ngưỡng Phân Loại | Macro-F1 (4 lớp) | Active Macro-F1 | Toxic F1 | Hate F1 | Clean F1 | Subset Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| TF-IDF + LinearSVC | 0.50 | 0.4914 | 0.6552 | 0.5641 | 0.4644 | 0.9372 | **87.07%** |
| BiLSTM + Focal Loss (SV2) | 0.50 | 0.5061 | 0.6748 | 0.6000 | 0.5100 | 0.9100 | 82.50% |
| TF-IDF + Logistic Regression | 0.50 | 0.5186 | 0.6914 | 0.6120 | 0.5493 | 0.9130 | 81.54% |
| TF-IDF + LinearSVC | **Tuned** | 0.5269 | 0.7026 | 0.6212 | 0.5493 | **0.9372** | 82.57% |
| **TF-IDF + Logistic Regression** | **Tuned** | **0.5322** | **0.7096** | **0.6300** | **0.5629** | 0.9358 | 83.89% |

---

## ⚡ Bắt Đầu Nhanh (Quick Start)

### 1. Khởi tạo môi trường ảo & cài đặt thư viện
```bash
# Tạo môi trường ảo .venv
python -m venv .venv

# Kích hoạt môi trường ảo
# Trên Windows:
.venv\Scripts\activate
# Trên Linux / macOS:
source .venv/bin/activate

# Cài đặt danh sách phụ thuộc
pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Kiểm thử pipeline tiền xử lý
```bash
python src/preprocessing.py
```

### 3. Huấn luyện & Đánh giá mô hình Baseline Machine Learning (Tuần 2)
```bash
python src/baseline_ml.py
```

### 4. Huấn luyện & Đánh giá mô hình Baseline BiLSTM (Tuần 2)
```bash
python src/train_bilstm.py
```

### 5. Khởi chạy Web Demo (Streamlit - Tuần 5)
```bash
streamlit run app/app.py
```
