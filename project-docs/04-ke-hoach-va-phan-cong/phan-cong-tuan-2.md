# 04. Kế Hoạch Phân Chia Công Việc Tuần 2 & Tổng Kết Benchmark

> [!NOTE]
> Tài liệu này được đồng bộ trực tiếp với file thuyết minh phân công chính thức `DoAnNLP-Tuan2.md`.
> Khối lượng công việc và trách nhiệm được phân chia **cân bằng tuyệt đối 50 - 50** giữa hai thành viên nhóm.

---

## 1. Mục Tiêu Của Tuần 2

Tuần 2 tập trung vào:
1. Xây dựng hệ thống các mô hình Baseline từ Machine Learning truyền thống (TF-IDF + Logistic Regression / LinearSVC) đến Deep Learning (BiLSTM + Self-Attention).
2. Cài đặt hàm mất mát chuyên biệt **Binary Focal Loss ($\gamma = 2.0$)** kết hợp trọng số lớp (**Pos-Weights**) để xử lý tình trạng mất cân bằng dữ liệu đa nhãn (`toxic`, `hate`, `spam`, `clean`).
3. Phát triển module đánh giá đa nhãn `src/metrics.py` chuẩn hóa (Macro-F1, Precision, Recall, Confusion Matrix).
4. Thực nghiệm tối ưu hóa ngưỡng phân loại (**Threshold Tuning**) trên tập Validation (`val.csv`) để cải thiện hiệu năng cho các nhãn hiếm như `hate`.
5. Thiết lập bảng điểm chuẩn (**Benchmark Table**) trên tập kiểm thử độc lập (`test.csv`) làm cơ sở đối chiếu cho các mô hình Transformer ở Tuần 3.

---

## 2. Bảng Phân Chia Chi Tiết 50 - 50

### Bảng 1: Công Việc Của Sinh Viên 1 (Baseline Machine Learning, Tối Ưu Ngưỡng & Đánh Giá Multi-Label)

| STT | Hạng mục công việc | Chi tiết nhiệm vụ | Sản phẩm bàn giao | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Xây dựng Baseline TF-IDF + SVM / Logistic Regression** | • Trích xuất đặc trưng văn bản bằng `TfidfVectorizer` kết hợp unigram và bigram (`ngram_range=(1, 2)`), `max_features=20000`, `sublinear_tf=True`.<br>• Huấn luyện mô hình phân loại đa nhãn theo chiến lược `OneVsRestClassifier` với `LinearSVC` (bọc bởi `CalibratedClassifierCV`) và `Logistic Regression` (`class_weight='balanced'`). | • File mã nguồn `src/baseline_ml.py`<br>• Notebook thực nghiệm `notebooks/02_baseline_ml.ipynb`<br>• Trọng số mô hình tại `saved_models/` | **[x] Hoàn thành** |
| **2** | **Xây dựng Module Tính toán Chỉ số Đánh giá** | • Viết hàm tính toán chuẩn xác các chỉ số cho bài toán Multi-Label: Macro-F1, Micro-F1, Weighted-F1, Precision, Recall cho từng nhãn (`toxic`, `hate`, `spam`, `clean`) và trung bình toàn cục, Subset Accuracy, Hamming Loss.<br>• Vẽ biểu đồ trực quan hóa ma trận nhầm lẫn đa nhãn `plot_multilabel_confusion_matrix()`. | • Module `src/metrics.py` (hàm `evaluate_multilabel()` & `plot_multilabel_confusion_matrix()`) | **[x] Hoàn thành** |
| **3** | **Tối ưu Ngưỡng Phân loại (Threshold Tuning)** | • Quét điều chỉnh ngưỡng xác suất quyết định nhãn (dải từ $0.30$ đến $0.70$ bước $0.05$) trên tập Validation (`val.csv`) để tối ưu điểm Macro-F1 cho các nhãn mất cân bằng nặng như `hate` và `toxic`.<br>• Vẽ biểu đồ đường cong biến thiên F1 theo ngưỡng phân loại. | • Hàm tìm ngưỡng tối ưu `find_best_thresholds()` và `plot_threshold_tuning()` trong `src/metrics.py`<br>• Ảnh biểu đồ tại `assets/ml_threshold_tuning.png` | **[x] Hoàn thành** |
| **4** | **Tổng hợp Bảng kết quả Baseline & Quản lý Git** | • Đánh giá mô hình ML trên tập `test.csv` (với cả ngưỡng mặc định 0.50 và ngưỡng tối ưu).<br>• Tổng hợp kết quả đánh giá của cả mô hình ML truyền thống và mô hình Deep Learning của SV2 trên tập `test.csv`.<br>• Review Pull Request của SV2, merge vào nhánh `main` và cập nhật tiến độ vào `README.md`. | • Bảng đối sánh Benchmark Tuần 2<br>• Biểu đồ `assets/baseline_benchmark_comparison.png`<br>• Nhánh `main` cập nhật hoàn chỉnh | **[x] Hoàn thành** |

---

### Bảng 2: Công Việc Của Sinh Viên 2 (Baseline Deep Learning, Cài Đặt Loss Chuyên Biệt & Trọng Số Lớp)

| STT | Hạng mục công việc | Chi tiết nhiệm vụ | Sản phẩm bàn giao | Trạng thái |
| :---: | :--- | :--- | :--- | :---: |
| **1** | **Xây dựng Pipeline Dữ liệu PyTorch & Bộ Từ Vựng** | • Xây dựng class `Vocabulary` ánh xạ từ vựng sang chỉ số (`<PAD>`, `<UNK>`), lọc từ hiếm (`min_freq=2`).<br>• Xây dựng class `ToxicDataset` và hàm `create_dataloaders()` xuất DataLoader chuẩn hóa cho Train, Val, Test. | File mã nguồn `src/dataset.py` | **[x] Hoàn thành** |
| **2** | **Cài đặt Kiến trúc Mô hình BiLSTM + Self-Attention** | • Xây dựng mô hình `BiLSTMAttention`: Embedding (dim 128) $\to$ 2 lớp Bidirectional LSTM (hidden dim 128) $\to$ Cơ chế Self-Attention $\to$ Fully Connected Layer xuất logits 4 nhãn. | File mã nguồn `src/models.py` | **[x] Hoàn thành** |
| **3** | **Cài đặt Hàm Mất Mát Chuyên Biệt (Custom Loss)** | • Tính toán vector trọng số lớp $Pos\_Weight = \frac{N_{negative}}{N_{positive}}$ lưu tại `configs/class_weights.json`.<br>• Cài đặt module `WeightedBCELoss` và `BinaryFocalLoss` ($\gamma=2.0, \alpha$) nhằm tập trung phạt nặng các mẫu khó/thiểu số. | • File mã nguồn `src/loss.py`<br>• Cấu hình `configs/class_weights.json` | **[x] Hoàn thành** |
| **4** | **Huấn luyện, So Sánh Loss & Đánh Giá BiLSTM** | • Huấn luyện so sánh giữa Weighted BCE Loss và Binary Focal Loss.<br>• Đánh giá mô hình BiLSTM tốt nhất trên tập `test.csv`, vẽ ma trận nhầm lẫn và trực quan hóa trọng số Attention. | • Script `src/train_bilstm.py`<br>• Notebook `notebooks/03_baseline_bilstm.ipynb`<br>• Ảnh `assets/bilstm_loss_comparison.png` | **[x] Hoàn thành** |

---

## 3. Bảng Điểm Chuẩn Đối Sánh Baseline Tuần 2 (Test Set Benchmark)

Đánh giá độc lập trên tập kiểm thử `data/processed/test.csv` (3,998 bình luận):

| Mô hình | Ngưỡng | Macro-F1 (4 lớp) | Active Macro-F1 | Toxic-F1 | Hate-F1 | Clean-F1 | Subset Acc |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **TF-IDF + LinearSVC** | 0.50 | 0.4914 | 0.6552 | 0.5641 | 0.4644 | 0.9372 | **87.07%** |
| **BiLSTM + Focal Loss** | 0.50 | 0.5061 | 0.6748 | 0.6000 | 0.5100 | 0.9100 | 82.50% |
| **TF-IDF + Logistic Regression** | 0.50 | 0.5186 | 0.6914 | 0.6120 | 0.5493 | 0.9130 | 81.54% |
| **TF-IDF + LinearSVC** | **Tuned** | 0.5269 | 0.7026 | 0.6212 | 0.5493 | **0.9372** | 82.57% |
| **TF-IDF + Logistic Regression** | **Tuned** | **0.5322** | **0.7096** | **0.6300** | **0.5629** | 0.9358 | 83.89% |
