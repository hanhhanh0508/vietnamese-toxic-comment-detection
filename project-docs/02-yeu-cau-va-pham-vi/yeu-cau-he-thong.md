# 02. Yêu Cầu Dự Án Và Hệ Thống

## 1. Yêu Cầu Chức Năng (Functional Requirements)

| Mã yêu cầu | Tên chức năng | Mô tả chi tiết chức năng | Trạng thái |
| :--- | :--- | :--- | :--- |
| **FR-01** | Thu thập & Gộp dữ liệu | Tải dữ liệu thô từ 3 nguồn ViHSD, ViCTSD, ViTHSD và hợp nhất về cấu trúc bảng chuẩn | `[CONFIRMED]` |
| **FR-02** | Chuẩn hóa Teencode & Emoji | Ánh xạ các từ viết tắt, từ lóng và emoji sang văn bản tiếng Việt tường minh thông qua từ điển JSON | `[CONFIRMED]` |
| **FR-03** | Tiền xử lý văn bản tiếng Việt | Chuẩn hóa Unicode NFC, loại bỏ URL, thẻ `@username`, xử lý ký tự lặp và phân tách từ ghép (`_`) | `[CONFIRMED]` |
| **FR-04** | Phân chia tập dữ liệu | Phân chia tập dữ liệu thành Train (80%), Val (10%), Test (10%) đảm bảo cân đối tỷ lệ phân bổ các nhãn (Stratified Split) | `[CONFIRMED]` |
| **FR-05** | Phân loại Đa nhãn (Multi-Label) | Dự đoán đồng thời 4 nhãn nhị phân độc lập: `toxic`, `hate`, `spam`, `clean` | `[CONFIRMED]` |
| **FR-06** | Xử lý mất cân bằng lớp | Hỗ trợ Binary Focal Loss kết hợp trọng số Pos-Weights trong quá trình tối ưu hóa mô hình | `[CONFIRMED]` |
| **FR-07** | Huấn luyện mô hình cơ sở | Cung cấp mã nguồn và kết quả baseline: TF-IDF + Logistic Regression / SVM và BiLSTM + Attention | `[CONFIRMED]` |
| **FR-08** | Fine-tune Transformer | Tinh chỉnh mô hình PhoBERT v2 và ViDeBERTa trên bộ dữ liệu đã làm sạch | `[CONFIRMED]` |
| **FR-09** | Mô hình kết hợp (Ensemble) | Thực hiện Soft Voting kết hợp đầu ra xác suất của PhoBERT và ViDeBERTa | `[CONFIRMED]` |
| **FR-10** | Trực quan hóa giải thích (XAI) | Tô màu từ ngữ vi phạm trong câu bình luận trực tiếp trên giao diện | `[CONFIRMED]` |
| **FR-11** | Giao diện Web tương tác | Cung cấp Web UI (Streamlit) cho phép người dùng nhập văn bản và nhận kết quả phân tích theo thời gian thực | `[CONFIRMED]` |

---

## 2. Yêu Cầu Phi Chức Năng (Non-Functional Requirements)

* **NFR-01 - Độ chính xác & Chỉ số đánh giá:**
  * Độ đo chính để đánh giá toàn bộ hệ thống là **Macro-F1 Score** (trung bình F1 giữa các lớp độc lập), kết hợp với Precision và Recall từng nhãn.
  * Không sử dụng Accuracy tổng làm chỉ số quyết định do hiện tượng mất cân bằng dữ liệu gây thiên lệch lớn.
* **NFR-02 - Thời gian phản hồi (Latency):**
  * Giao diện Web Demo trả kết quả phân loại và tô màu từ ngữ trong vòng dưới 2 giây đối với mỗi câu bình luận đơn lẻ.
* **NFR-03 - Tính mô-đun và Tái sử dụng (Modularity & Maintainability):**
  * Mã nguồn được cấu trúc tường minh trong thư mục `src/`, phân tách rõ giữa tiền xử lý (`preprocessing.py`), mô hình (`models.py`), hàm mất mát (`loss.py`), và huấn luyện (`train.py`).
* **NFR-04 - Khả năng tái lập kết quả (Reproducibility):**
  * Cố định random seed trong toàn bộ các khâu (chia dữ liệu, khởi tạo trọng số, huấn luyện).
  * Cung cấp file `requirements.txt` với phiên bản thư viện rõ ràng để người khác có thể dựng lại môi trường chạy độc lập.
* **NFR-05 - Quản lý an toàn dữ liệu & Mã nguồn:**
  * File dữ liệu nặng (>10MB) và checkpoint mô hình tuyệt đối không commit lên GitHub; phải được cấu hình qua `.gitignore` và backup trên Google Drive.

---

## 3. Yêu Cầu Về Dữ Liệu

* **Nguồn dữ liệu sử dụng:**
  1. `ViHSD` (Vietnamese Hate Speech Detection) - do UIT-NLP xây dựng, chứa ~33.400 bình luận.
  2. `ViCTSD` (Vietnamese Constructive and Toxic Speech Detection) - phân loại tính xây dựng và tính độc hại.
  3. `ViTHSD` (Vietnamese Toxic Hate Speech Dataset).
* **Định dạng lưu trữ:** CSV với chuẩn mã hóa bắt buộc là `UTF-8`.
* **Cấu trúc cột dữ liệu sạch:**
  * `text_raw`: Chuỗi văn bản bình luận ban đầu.
  * `text_processed`: Chuỗi văn bản sau khi qua pipeline tiền xử lý và tách từ ghép.
  * `toxic`: Giá trị nhị phân `0` hoặc `1`.
  * `hate`: Giá trị nhị phân `0` hoặc `1`.
  * `spam`: Giá trị nhị phân `0` hoặc `1`.
  * `clean`: Giá trị nhị phân `0` hoặc `1` (bằng `1` khi cả 3 nhãn kia đều bằng `0`).

---

## 4. Yêu Cầu Về Môi Trường & Công Cụ

* **Ngôn ngữ:** Python $\ge 3.9$ (khuyến nghị Python 3.10 hoặc 3.11).
* **Deep Learning Framework:** PyTorch $\ge 2.0.0$, Hugging Face Transformers $\ge 4.30.0$, Datasets, Accelerate.
* **NLP & Tiền xử lý:** Underthesea $\ge 6.0.0$, PyVi $\ge 0.1.1$, Emoji $\ge 2.5.0$.
* **Data Science & Visualization:** Pandas $\ge 2.0.0$, Scikit-learn $\ge 1.2.0$, Matplotlib $\ge 3.7.0$, Seaborn $\ge 0.12.0$.
* **Giao diện & XAI:** Streamlit $\ge 1.25.0$, Lime $\ge 0.2.0$.
* **Môi trường tính toán:** Máy cục bộ có GPU NVIDIA hỗ trợ CUDA hoặc Google Colab / Kaggle GPU (`[UNKNOWN]`: cấu hình phần cứng cụ thể của nhóm cần xác nhận).
