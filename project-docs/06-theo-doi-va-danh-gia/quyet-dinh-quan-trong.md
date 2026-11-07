# 06. Nhật Ký Các Quyết Định Kiến Trúc (Architecture Decision Records - ADR)

Tài liệu này ghi nhận toàn bộ các quyết định kỹ thuật then chốt đã được hai thành viên nhóm và chuyên gia NLP thống nhất trong suốt quá trình phát triển đồ án.

---

## Danh Sách Các Quyết Định Kỹ Thuật (ADR Log)

### ADR-01: Chuyển đổi bài toán từ Phân loại nhị phân sang Phân loại Đa nhãn (Multi-Label)
* **Bối cảnh:** Ban đầu đề tài dự kiến chỉ phân loại nhị phân (`Toxic` vs `Clean`). Tuy nhiên bình luận mạng xã hội có tính phức hợp (một câu có thể vừa chửi bới thô tục, vừa kích động thù ghét sắc tộc, hoặc quảng cáo rác).
* **Quyết định:** Nâng cấp bài toán sang **Multi-Label Classification** với 4 nhãn độc lập: `[toxic, hate, spam, clean]`.
* **Hệ quả:** Thay đổi tầng đầu ra từ Softmax sang độc lập Sigmoid; thay đổi hàm mất mát từ Cross-Entropy sang BCE/Focal Loss.
* **Trạng thái:** `[CONFIRMED]`

---

### ADR-02: Lựa chọn hướng nâng cấp NLP Level-Up (Nghiên cứu sâu) cho lộ trình 6 tuần
* **Bối cảnh:** Khi mở rộng thời gian từ 4 tuần lên 6 tuần, nhóm có hai phương án: Hướng A (Tập trung bài toán NLP, Focal Loss, Ensemble Transformer, XAI) hoặc Hướng B (MLOps, cào dữ liệu Facebook/TikTok bằng Selenium, nén ONNX).
* **Quyết định:** Nhóm quyết định lựa chọn **Hướng A (NLP Level-Up)** nhằm tăng hàm lượng nghiên cứu học thuật và tránh các rủi ro kỹ thuật khi cào dữ liệu từ nền tảng đóng.
* **Hệ quả:** Loại bỏ các module Selenium/Facebook Scraper; tập trung nguồn lực vào tối ưu mô hình học sâu và hàm mất mát.
* **Trạng thái:** `[CONFIRMED]`

---

### ADR-03: Áp dụng Binary Focal Loss ($\gamma=2.0$) xử lý mất cân bằng lớp
* **Bối cảnh:** Tập dữ liệu ViHSD có tỷ lệ mất cân bằng nghiêm trọng: nhãn Clean chiếm 75-80%, nhãn Hate chỉ chiếm 3-5%. Mô hình dễ rơi vào bẫy đoán tất cả là Clean.
* **Quyết định:** Cài đặt Custom Loss **Binary Focal Loss** với $\gamma=2.0, \alpha=0.25$ kết hợp Pos-Weights trong PyTorch.
* **Hệ quả:** Mô hình tập trung học các mẫu khó (hard negative/positive), cải thiện vượt trội chỉ số Macro-F1 trên lớp thiểu số.
* **Trạng thái:** `[CONFIRMED]`

---

### ADR-04: Huấn luyện độc lập PhoBERT & ViDeBERTa và kết hợp bằng Soft Voting Ensemble
* **Bối cảnh:** PhoBERT mạnh về biểu diễn từ ghép tiếng Việt (RoBERTa), trong khi ViDeBERTa mạnh về tương quan vị trí ngữ cảnh (DeBERTa). Sử dụng một mô hình đơn lẻ dễ bị sai số cục bộ.
* **Quyết định:** Huấn luyện song song cả 2 mô hình trên cùng tập dữ liệu chuẩn hóa, sau đó suy luận kết hợp bằng kỹ thuật **Soft Voting** (trung bình có trọng số các phân phối xác suất).
* **Hệ quả:** Đạt Macro-F1 cao hơn so với từng mô hình thành phần, giảm hiện tượng Overfitting.
* **Trạng thái:** `[CONFIRMED]`

---

### ADR-05: Phân chia công việc cân bằng tuyệt đối 50 - 50 giữa 2 sinh viên
* **Bối cảnh:** Cần tránh tình trạng dồn việc kỹ thuật cho trưởng nhóm, đồng thời đảm bảo cả 2 sinh viên đều nắm vững toàn bộ quy trình từ dữ liệu đến mô hình để bảo vệ trước hội đồng phản biện.
* **Quyết định:** Phân công 50-50 theo các hạng mục công việc cụ thể (đã thể hiện trong file Word `Phan_Chia_Cong_Viec_Tuan_1_Do_An_NLP.docx`). Mỗi bạn phụ trách một luồng dữ liệu/mô hình tương ứng và review chéo qua Pull Request.
* **Trạng thái:** `[CONFIRMED]`

---

### ADR-06: Thiết lập chiến lược sao lưu 2 tầng (Two-Tier Storage & Backup)
* **Bối cảnh:** GitHub giới hạn kích thước file <100MB, trong khi các bộ dữ liệu CSV thô/sạch và checkpoint mô hình Transformer thường có dung lượng từ vài trăm MB đến hàng GB.
* **Quyết định:** Tầng 1 (GitHub) chỉ lưu mã nguồn `.py`, `.ipynb`, `.json`, `.md`. Tầng 2 (Google Drive chung) lưu toàn bộ dữ liệu CSV thô, dữ liệu sạch và các file checkpoint trọng số `.pth`, `.bin`. Cấu hình `.gitignore` triệt để.
* **Trạng thái:** `[CONFIRMED]`

---

### ADR-07: Phương án giải quyết xung đột bảng mã UTF-8 (Mojibake)
* **Bối cảnh:** Khi mở file CSV tiếng Việt trên Windows Excel, chữ bị lỗi hiển thị thành các ký tự lạ (`Coi cÆ°á» i`, `NÃ³`).
* **Quyết định:** Giữ nguyên file dữ liệu chuẩn UTF-8 gốc, không can thiệp sửa thô bằng Excel; luôn sử dụng tham số `encoding="utf-8"` trong Python Pandas để đọc dữ liệu. Hướng dẫn thiết lập Import CSV UTF-8 trong Excel cho mục đích xem kiểm tra.
* **Trạng thái:** `[CONFIRMED]`

---

### ADR-08: Phương án xử lý lỗi 403 Forbidden trên Hugging Face
* **Bối cảnh:** Dataset `uitnlp/vihsd` trên Hugging Face đặt chế độ Gated Dataset, dẫn đến lỗi 403 khi tải qua script tự động lúc chưa được duyệt quyền.
* **Quyết định:** Ưu tiên tải trực tiếp file thô từ kho mã nguồn mở GitHub của tác giả (`sonlam1102/vihsd`, `tarudesu/ViCTSD`, `bakansm/ViTHSD`) để không bị tắc nghẽn tiến độ Tuần 1. Song song đó tạo token xin quyền chính thức trên Hugging Face.
* **Trạng thái:** `[CONFIRMED]`
