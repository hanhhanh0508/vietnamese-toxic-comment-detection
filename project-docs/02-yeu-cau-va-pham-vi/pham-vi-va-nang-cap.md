# 02. Phạm Vi Dự Án & Định Hướng Nâng Cấp

## 1. So Sánh Phạm Vi Ban Đầu Và Phạm Vi Nâng Cấp Đã Chốt

Trong quá trình trao đổi, nhóm sinh viên đã quyết định kéo dài thời gian thực hiện từ 4 tuần lên **6 tuần** và lựa chọn nâng cấp theo chiều sâu khoa học xử lý ngôn ngữ tự nhiên (NLP Level-Up). 

Bảng đối chiếu chi tiết giữa phạm vi ban đầu và phạm vi nâng cấp chính thức:

| Tiêu chí | Phạm vi ban đầu (4 tuần) | Phạm vi nâng cấp chính thức (6 tuần) | Tình trạng quyết định |
| :--- | :--- | :--- | :--- |
| **Loại bài toán** | Phân loại nhị phân đơn giản (`Toxic` vs `Clean`) | **Phân loại đa nhãn (Multi-Label Classification)** với 4 nhãn song song: `[toxic, hate, spam, clean]` | `[CONFIRMED]` |
| **Hàm mất mát** | Cross-Entropy Loss tiêu chuẩn hoặc Weighted Cross-Entropy | **Binary Focal Loss ($\gamma=2.0$)** kết hợp Pos-Weights cho từng nhãn để trị dứt điểm mất cân bằng | `[CONFIRMED]` |
| **Số lượng Transformer** | Chỉ Fine-tune PhoBERT (`phobert-base-v2`) | Fine-tune độc lập cả **PhoBERT v2** và **ViDeBERTa** | `[CONFIRMED]` |
| **Chiến lược suy luận** | Single Model Inference (chỉ lấy kết quả từ PhoBERT) | **Ensemble Soft Voting** (kết hợp phân phối xác suất PhoBERT + ViDeBERTa) | `[CONFIRMED]` |
| **Cơ chế Highlight từ ngữ** | Dựa trên danh sách từ điển tĩnh `bad_words` | Giai đoạn 1: Từ điển tĩnh $\rightarrow$ Giai đoạn 2: Trích xuất bằng **Attention Score / LIME (XAI)** | `[CONFIRMED]` |
| **Sản phẩm Web Demo** | Nhập text và xuất nhãn nhị phân | Giao diện Streamlit hiển thị thanh xác suất 4 nhãn đa mục tiêu và tô màu từ ngữ trực quan | `[CONFIRMED]` |

---

## 2. Lý Do Lựa Chọn Nâng Cấp Theo Hướng Nghiên Cứu NLP

Khi được tư vấn giữa hai hướng:
* *Hướng A: Nâng cấp chiều sâu NLP & Mô hình hóa (NLP Level-up & Research).*
* *Hướng B: Nâng cấp hệ thống kỹ nghệ & MLOps (Web Scraping Facebook/TikTok, ONNX quantization, Feedback loop).*

Nhóm sinh viên đã chủ động xác nhận: **"Em muốn theo hướng nâng cấp của cái đầu ạ"**.

**Lợi ích chiến lược của quyết định này:**
1. **Phản ánh bản chất ngôn ngữ mạng xã hội:** Trong thực tế, một câu nói tục chửi thề (`toxic`) chưa chắc đã là thù ghét sắc tộc hay tôn giáo (`hate`). Việc tách riêng nhãn giúp hệ thống đạt độ tinh xảo cao trong phân tích nội dung.
2. **Nâng cao giá trị học thuật:** Việc thiết kế Custom Loss (Binary Focal Loss), xây dựng kiến trúc PyTorch Multi-Label Classifier và Ensemble đa kiến trúc Transformer (RoBERTa vs DeBERTa) mang lại hàm lượng chuyên môn cao cho báo cáo đồ án tốt nghiệp / môn học.
3. **Tối ưu hóa chỉ số Macro-F1:** Kỹ thuật Soft Voting giúp tận dụng thế mạnh phân tách từ ghép của PhoBERT và cơ chế Disentangled Attention của ViDeBERTa, bù trừ sai số cá thể giữa hai mô hình.

---

## 3. Các Tính Năng Đề Xuất Được Giữ Lại Hoặc Tạm Hoãn (Scope Exclusions)

Để đảm bảo tính khả thi trong 6 tuần của 2 sinh viên, các tính năng thuộc hướng MLOps tạm thời không đưa vào phạm vi bắt buộc của đồ án:

| Tính năng | Phân loại | Ghi chú & Lý do điều chỉnh |
| :--- | :--- | :--- |
| **Cào dữ liệu từ link Facebook / TikTok** | `[PROPOSED - EXCLUDED]` | Tiềm ẩn rủi ro vi phạm chính sách API của Meta/ByteDance, dễ bị chặn IP (Cloudflare/CAPTCHA), làm chệch trọng tâm nghiên cứu NLP |
| **Nén mô hình sang định dạng ONNX / INT8** | `[PROPOSED - OPTIONAL]` | Có thể thử nghiệm ở Tuần 5 nếu còn thời gian để tối ưu tốc độ Demo trên Hugging Face Spaces |
| **Active Learning & Lưu Feedback vào Database** | `[PROPOSED - EXCLUDED]` | Cần thêm hạ tầng cơ sở dữ liệu (SQLite/MongoDB), không nằm trong tiêu chí đánh giá trọng tâm của đề tài NLP |
| **Tải hàng loạt qua file CSV trên Web Demo** | `[PROPOSED - OPTIONAL]` | Có thể tích hợp nhanh vào Streamlit ở Tuần 5 nếu tiến độ kiểm thử mô hình hoàn thành sớm |
