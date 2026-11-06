# 04. Kế Hoạch Phân Chia Công Việc Tuần 1

> [!NOTE]
> Tài liệu này được đồng bộ trực tiếp với file thuyết minh phân công chính thức `Phan_Chia_Cong_Viec_Tuan_1_Do_An_NLP.docx` đã được phê duyệt.
> Khối lượng công việc và trách nhiệm được phân chia **cân bằng tuyệt đối 50 - 50** giữa hai thành viên nhóm.

---

## 1. Mục Tiêu Của Tuần 1

Tuần 1 tập trung vào:
1. Khảo sát dữ liệu thô từ các nguồn học thuật uy tín (`ViHSD`, `ViCTSD`, `ViTHSD`).
2. Xây dựng các bộ từ điển chuẩn hóa tiếng Việt đặc thù (`teencode.json`, `emoji.json`).
3. Phát triển module tiền xử lý văn bản NLP hoàn chỉnh (`src/preprocessing.py`).
4. Hợp nhất và phân chia tập dữ liệu Multi-Label theo chuẩn Train (80%) - Val (10%) - Test (10%).
5. Phân tích trực quan hóa dữ liệu EDA để xác định tham số tối ưu (`MAX_LENGTH`) và mức độ mất cân bằng lớp chuẩn bị cho Tuần 2.

---

## 2. Bảng Phân Chia Chi Tiết 50 - 50

### Bảng 1: Công Việc Của Sinh Viên 1 (Teencode, Chuẩn Hóa & EDA Độ Dài)

| STT | Hạng mục công việc | Chi tiết nhiệm vụ | Sản phẩm bàn giao |
| :---: | :--- | :--- | :--- |
| **1** | **Khảo sát & Xây dựng Từ điển Teencode** | • Thu thập, phân loại các từ viết tắt, từ lóng, teencode phổ biến trên mạng xã hội tiếng Việt.<br>• Đóng gói thành từ điển chuẩn định dạng JSON ánh xạ về từ vựng gốc (ví dụ: `k` $\to$ `không`, `vcl` $\to$ `vãi lờ`). | File `data/dictionaries/teencode.json` |
| **2** | **Phát triển Module Chuẩn hóa Văn bản** | • Viết các hàm trong `src/preprocessing.py`: đưa văn bản về Unicode NFC, regex loại bỏ URL và tag `@username`, xử lý ký tự lặp.<br>• Tích hợp từ điển teencode để tự động thay thế từ viết tắt. | File `src/preprocessing.py` (chứa các hàm chuẩn hóa chuỗi) |
| **3** | **Chia tập dữ liệu chuẩn (Phối hợp)** | • Phối hợp cùng SV2 thực hiện Stratified Split chia bảng dữ liệu sạch thành 3 phần: Train (80%), Validation (10%), Test (10%).<br>• Đảm bảo tỷ lệ các nhãn độc hại phân bổ đồng đều giữa các tập. | 3 file dữ liệu tại `data/processed/`: `train.csv`, `val.csv`, `test.csv` |
| **4** | **Phân tích EDA: Độ dài câu & Token** | • Viết phần phân tích thống kê độ dài văn bản trong Jupyter Notebook.<br>• Trực quan hóa phân bố số từ/token để làm cơ sở chọn tham số `MAX_LENGTH` tối ưu cho PhoBERT (128 hoặc 256). | Phần 1 Notebook `notebooks/01_eda_and_preprocessing.ipynb` |
| **5** | **Đóng gói mã nguồn & Cập nhật Tiến độ** | • Kiểm tra chất lượng code, tạo và review Pull Request trên GitHub.<br>• Cập nhật thông tin tổng kết tiến độ Tuần 1 vào tài liệu dự án và file `README.md`. | Nhánh `main` tích hợp đủ code & file `README.md` cập nhật |

---

### Bảng 2: Công Việc Của Sinh Viên 2 (Emoji, Cấu Trúc Đa Nhãn & EDA Nhãn)

| STT | Hạng mục công việc | Chi tiết nhiệm vụ | Sản phẩm bàn giao |
| :---: | :--- | :--- | :--- |
| **1** | **Thu thập & Xây dựng Từ điển Emoji** | • Thu thập các tập dữ liệu thô (`ViHSD`, `ViCTSD`, `ViTHSD`) lưu trữ an toàn làm backup.<br>• Xây dựng file JSON ánh xạ các biểu tượng cảm xúc (Emoji) sang từ mô tả tiếng Việt tương ứng (ví dụ: 😡 $\to$ `phẫn nộ`). | • File thô lưu tại `data/raw/`<br>• File `data/dictionaries/emoji.json` |
| **2** | **Chuẩn hóa Cấu trúc Multi-Label & Tách từ NLP** | • Viết script gộp các bộ dữ liệu về bảng 4 nhãn nhị phân: `toxic`, `hate`, `spam`, `clean`.<br>• Viết hàm tách từ tiếng Việt bằng `underthesea` bổ sung vào `src/preprocessing.py`. | • File `data/raw/merged_dataset.csv`<br>• Hàm `word_segmentation()` trong `src/preprocessing.py` |
| **3** | **Thực thi Pipeline Tiền xử lý Dữ liệu** | • Nhập toàn bộ dữ liệu thô qua module tiền xử lý hoàn chỉnh (kết hợp cả teencode, emoji và tách từ).<br>• Rà soát loại bỏ các mẫu dữ liệu rỗng hoặc bị lỗi định dạng ký tự lạ. | Bảng dữ liệu hoàn chỉnh đã được làm sạch |
| **4** | **Phân tích EDA: Phân bố Nhãn & Mất cân bằng** | • Trực quan hóa số lượng mẫu và tỷ lệ phần trăm của từng nhãn (`toxic`, `hate`, `spam`, `clean`).<br>• Đánh giá mức độ mất cân bằng lớp để tính toán trọng số phục vụ cài đặt Focal Loss ở Tuần 2. | Phần 2 Notebook `notebooks/01_eda_and_preprocessing.ipynb` |
| **5** | **Quản lý Lưu trữ & Backup Dữ liệu Cloud** | • Đưa toàn bộ các tập dữ liệu sạch (`train.csv`, `val.csv`, `test.csv`) lên thư mục lưu trữ Cloud (Google Drive).<br>• Tạo Pull Request gộp các phần công việc cá nhân lên GitHub. | • Thư mục dữ liệu đã backup trên Google Drive<br>• Pull Request hoàn chỉnh trên GitHub |

---

## 3. Quy Trình Phối Hợp & Nghiệm Thu Bàn Giao

1. **Lưu trữ Data & Checkpoints:**
   * Các file dữ liệu CSV thô và sạch được lưu trên Google Drive chung của nhóm.
   * File code Python, từ điển JSON và Jupyter Notebooks được quản lý tập trung trên GitHub.
2. **Quy chuẩn Đóng góp GitHub:**
   * Mỗi sinh viên làm việc trên nhánh tính năng cá nhân (`feature/preprocessing` và `feature/eda-dataset`).
   * Khi hoàn thành một module, thành viên tạo Pull Request (PR) để người còn lại thực hiện Code Review chéo trước khi bấm merge vào nhánh `main`.
