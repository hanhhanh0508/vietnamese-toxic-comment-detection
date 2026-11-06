# 06. Danh Sách Vấn Đề Tồn Đọng, Mục Cần Xác Nhận (TODO & UNKNOWN)

Tài liệu này tổng hợp các câu hỏi kỹ thuật mở, những thông tin còn thiếu dữ liệu trong quá trình trao đổi, và các đầu mối công việc cần nhóm thực hiện xác minh trong các tuần tiếp theo.

---

## 1. Các Vấn Đề Kỹ Thuật Cần Xác Nhận (Open Questions & UNKNOWN)

| Mã | Vấn đề kỹ thuật | Phân loại | Tác động dự án | Đề xuất hướng giải quyết |
| :---: | :--- | :---: | :--- | :--- |
| **UQ-01** | **Nguồn gán nhãn cho lớp `spam` khi gộp dữ liệu** | `[UNKNOWN]` | Cao | Tập dữ liệu `ViHSD` gốc chỉ có 3 nhãn: Clean (0), Offensive (1), Hate (2). Cần xác nhận: cột `spam` trong vector 4 nhãn sẽ được trích xuất từ tập `ViCTSD` hay dùng luật heuristics / từ khóa nhận diện tin quảng cáo rác? |
| **UQ-02** | **Cấu hình phần cứng huấn luyện mô hình (GPU / VRAM)** | `[UNKNOWN]` | Cao | Huấn luyện PhoBERT v2 và ViDeBERTa với batch size 16 đòi hỏi ít nhất 8GB - 12GB VRAM GPU. Cần làm rõ nhóm huấn luyện trên card GPU cá nhân (RTX series) hay sử dụng Google Colab / Kaggle T4 / P100 GPU? |
| **UQ-03** | **Thống nhất công cụ tách từ tiếng Việt (`underthesea` vs `pyvi`)** | `[UNKNOWN]` | Trung bình | `chat.txt` nhắc đến cả hai thư viện. PhoBERT gốc được huấn luyện dựa trên `VnCoreNLP (rdrsegmenter)` và tương thích cao với `pyvi`. Cần chạy thử nghiệm so sánh tốc độ và độ khớp token giữa `underthesea` và `pyvi` trước khi chốt cố định trong `src/preprocessing.py`. |
| **UQ-04** | **Hiệu năng thời gian thực của XAI (LIME vs Attention)** | `[UNKNOWN]` | Trung bình | Thư viện `lime` chạy trên Transformer tốn từ 5 - 15 giây cho một câu bình luận (gây nghẽn Web Demo Streamlit). Đề xuất ưu tiên phương án trích xuất trực tiếp trọng số Attention của lớp cuối cùng (Attention Rollout) để đạt tốc độ phản hồi dưới 1 giây. |
| **UQ-05** | **Đường dẫn thư mục Google Drive chung** | `[TODO]` | Thấp | Cần SV2 tạo folder chính thức và cập nhật link vào tài liệu nội bộ nhóm để cả hai thành viên lưu trữ dữ liệu đồng bộ. |

---

## 2. Kế Hoạch Nhiệm Vụ Tiếp Theo (Action Items & TODO)

### Nhiệm vụ chốt ngay trong Tuần 1:
- [ ] **TODO-01 (SV2):** Hoàn thiện script hợp nhất `merged_dataset.csv` từ ViHSD, ViCTSD, ViTHSD và định nghĩa rõ luật kích hoạt nhãn `spam`.
- [ ] **TODO-02 (SV1):** Mở rộng từ điển `teencode.json` lên ít nhất 150 - 200 từ lóng và từ viết tắt mạng xã hội phổ biến.
- [ ] **TODO-03 (SV2):** Hoàn thiện từ điển `emoji.json` với ít nhất 40 biểu tượng cảm xúc tiêu biểu nhất.
- [ ] **TODO-04 (SV1 & SV2):** Chạy toàn bộ dữ liệu qua pipeline làm sạch, phân chia 3 tập `train.csv` (80%), `val.csv` (10%), `test.csv` (10%) và backup lên Google Drive.
- [ ] **TODO-05 (SV1 & SV2):** Hoàn tất notebook `notebooks/01_eda_and_preprocessing.ipynb` với 2 biểu đồ quan trọng:
  1. Biểu đồ Histogram phân bố độ dài từ/token $\to$ chốt chính thức giá trị `MAX_LENGTH` cho mô hình (128 hay 256).
  2. Biểu đồ cột phân bố tần suất 4 nhãn $\to$ tính toán trọng số `pos_weight` cho hàm mất mát.

### Nhiệm vụ chuẩn bị cho Tuần 2:
- [ ] **TODO-06:** Viết file `src/loss.py` chứa class `BinaryFocalLoss` kế thừa từ `nn.Module`.
- [ ] **TODO-07:** Viết notebook thử nghiệm Baseline TF-IDF + Logistic Regression / Linear SVM bằng Scikit-Learn.
- [ ] **TODO-08:** Cài đặt mô hình BiLSTM + Self-Attention trong PyTorch để làm mốc so sánh Deep Learning truyền thống.

---

## 3. Quản Trị Rủi Ro Dự Án (Risk Management)

| Rủi ro tiềm ẩn | Mức độ ảnh hưởng | Khả năng xảy ra | Biện pháp phòng ngừa & Giảm thiểu |
| :--- | :---: | :---: | :--- |
| **Tràn bộ nhớ GPU (CUDA Out of Memory)** khi fine-tune PhoBERT | Cao | Trung bình | Giảm `batch_size` xuống 8 hoặc 4, áp dụng Gradient Accumulation (`gradient_accumulation_steps=2` hoặc `4`), bật `fp16=True`. |
| **Overfitting trên các lớp hiếm (`hate`)** | Cao | Cao | Bắt buộc áp dụng Focal Loss ($\gamma=2.0$), bổ sung Dropout ($0.2 - 0.3$), Weight Decay ($0.01$), và lưu checkpoint dựa trên Macro-F1 trên tập Validation thay vì loss tổng. |
| **Xung đột mã nguồn khi cả hai cùng sửa file `src/preprocessing.py`** | Trung bình | Thấp | Tuân thủ tuyệt đối quy trình phân nhánh Git: SV1 làm các hàm chuỗi/regex, SV2 làm hàm tách từ; họp nhanh trước khi tạo Pull Request để thống nhất chữ ký hàm (function signatures). |
| **Mất mát dữ liệu cục bộ** | Cao | Thấp | Luôn tuân thủ mô hình backup 2 tầng: Code trên GitHub Cloud, Data và Checkpoints trên Google Drive. |
