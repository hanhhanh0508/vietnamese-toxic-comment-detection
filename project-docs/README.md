# Hệ Thống Tài Liệu Kỹ Thuật Đồ Án NLP
## Đề tài: Hệ Thống Tự Động Phát Hiện Bình Luận Độc Hại Đa Nhãn (Multi-Label Toxic Comment Detection) Trên Mạng Xã Hội Tiếng Việt

> **Trạng thái tài liệu:** Tài liệu chính thức (Phiên bản đồng bộ theo tiến độ Tuần 1)  
> **Nhóm thực hiện:** Nhóm 2 Sinh viên (SV1 & SV2)  
> **Thời gian triển khai:** 6 tuần  
> **Công nghệ lõi:** Python, PyTorch, Hugging Face Transformers, PhoBERT, ViDeBERTa, Streamlit  

---

## 1. Giới Thiệu Cấu Trúc Tài Liệu

Bộ tài liệu này được xây dựng dựa trên toàn bộ lịch sử phân tích và thống nhất trong quá trình phát triển đồ án, đóng vai trò là kim chỉ nam kỹ thuật (Single Source of Truth) cho cả hai thành viên nhóm trong suốt quá trình triển khai 6 tuần.

Tài liệu được phân bổ thành 6 phân hệ logic:

```text
project-docs/
├── README.md                           # Bản đồ điều hướng tài liệu (file hiện tại)
├── 01-tong-quan-du-an/                 # Bối cảnh, mục tiêu và phân tích bài toán
│   ├── tong-quan.md                    # Tổng quan đề tài, mục tiêu và kết quả kỳ vọng
│   └── van-de-va-giai-phap.md          # Thách thức đặc thù tiếng Việt & giải pháp kỹ thuật
├── 02-yeu-cau-va-pham-vi/              # Định nghĩa phạm vi và yêu cầu kỹ thuật
│   ├── yeu-cau-he-thong.md             # Yêu cầu chức năng, phi chức năng và môi trường
│   └── pham-vi-va-nang-cap.md          # So sánh phạm vi ban đầu và định hướng nâng cấp 6 tuần
├── 03-kien-truc-va-thiet-ke/           # Kiến trúc hệ thống và thiết kế kỹ thuật chi tiết
│   ├── kien-truc-tong-the.md           # Sơ đồ pipeline toàn diện từ dữ liệu đến Web Demo
│   ├── pipeline-tien-xu-ly.md          # Chi tiết module tiền xử lý văn bản tiếng Việt
│   ├── thiet-ke-du-lieu-multi-label.md # Thiết kế cấu trúc nhãn đa mục tiêu và quy tắc gộp tập dữ liệu
│   └── thiet-ke-mo-hinh-va-loss.md     # Kiến trúc PhoBERT, ViDeBERTa, Ensemble và Binary Focal Loss
├── 04-ke-hoach-va-phan-cong/           # Quản lý tiến độ và phân chia công việc nhóm
│   ├── lo-trinh-6-tuan.md              # Lộ trình tổng thể 6 tuần chi tiết từng mốc
│   ├── phan-cong-tuan-1.md             # Kế hoạch phân chia 50-50 chi tiết Tuần 1
│   └── quy-trinh-git-va-backup.md      # Quy chuẩn Git workflow và chiến lược lưu trữ 2 tầng
├── 05-huong-dan-ky-thuat/              # Cẩm nang thực hành và xử lý lỗi
│   ├── cai-dat-moi-truong.md           # Hướng dẫn thiết lập virtualenv và requirements.txt
│   ├── xu-ly-su-co-thuong-gap.md       # Khắc phục lỗi UTF-8 Mojibake và lỗi 403 Hugging Face
│   └── huong-dan-web-demo-xai.md       # Xây dựng Web Demo Streamlit và Highlight thời gian thực
└── 06-theo-doi-va-danh-gia/           # Nhật ký quyết định và các nội dung cần làm rõ
    ├── quyet-dinh-quan-trong.md        # Nhật ký các quyết định kiến trúc (ADR)
    └── van-de-ton-dong-va-todo.md      # Danh sách câu hỏi mở, TODO và mục cần xác nhận
```

---

## 2. Bản Đồ Điều Hướng Nhanh (Navigation Index)

| Phân hệ | Tài liệu chi tiết | Mô tả nội dung chính |
| :--- | :--- | :--- |
| **01. Tổng quan** | [tong-quan.md](01-tong-quan-du-an/tong-quan.md) | Mục tiêu, tính cấp thiết và kết quả bàn giao của đồ án |
| | [van-de-va-giai-phap.md](01-tong-quan-du-an/van-de-va-giai-phap.md) | Thách thức teencode, emoji, mất cân bằng nhãn & hướng xử lý |
| **02. Yêu cầu & Phạm vi** | [yeu-cau-he-thong.md](02-yeu-cau-va-pham-vi/yeu-cau-he-thong.md) | Yêu cầu chức năng, phi chức năng và độ đo F1/Precision/Recall |
| | [pham-vi-va-nang-cap.md](02-yeu-cau-va-pham-vi/pham-vi-va-nang-cap.md) | Phân tích quyết định nâng cấp lên Multi-Label 4 lớp & Focal Loss |
| **03. Kiến trúc & Thiết kế** | [kien-truc-tong-the.md](03-kien-truc-va-thiet-ke/kien-truc-tong-the.md) | Sơ đồ Mermaid toàn diện luồng dữ liệu (Data -> Pipeline -> Model -> UI) |
| | [pipeline-tien-xu-ly.md](03-kien-truc-va-thiet-ke/pipeline-tien-xu-ly.md) | Thiết kế module `src/preprocessing.py`, regex, teencode, emoji, tách từ |
| | [thiet-ke-du-lieu-multi-label.md](03-kien-truc-va-thiet-ke/thiet-ke-du-lieu-multi-label.md) | Cấu trúc nhãn vector `[toxic, hate, spam, clean]`, quy tắc gộp ViHSD/ViCTSD/ViTHSD |
| | [thiet-ke-mo-hinh-va-loss.md](03-kien-truc-va-thiet-ke/thiet-ke-mo-hinh-va-loss.md) | Chi tiết kiến trúc mô hình PhoBERT, ViDeBERTa, Ensemble & Focal Loss |
| **04. Kế hoạch & Phân công** | [lo-trinh-6-tuan.md](04-ke-hoach-va-phan-cong/lo-trinh-6-tuan.md) | Chi tiết kế hoạch hành động từng tuần từ Tuần 1 đến Tuần 6 |
| | [phan-cong-tuan-1.md](04-ke-hoach-va-phan-cong/phan-cong-tuan-1.md) | Bảng phân chia 50-50 chi tiết các đầu mối bàn giao Tuần 1 |
| | [quy-trinh-git-va-backup.md](04-ke-hoach-va-phan-cong/quy-trinh-git-va-backup.md) | Quy định Git branch, pull request review & chiến lược sao lưu Drive |
| **05. Hướng dẫn kỹ thuật** | [cai-dat-moi-truong.md](05-huong-dan-ky-thuat/cai-dat-moi-truong.md) | Hướng dẫn cài đặt Python `venv` và các gói trong `requirements.txt` |
| | [xu-ly-su-co-thuong-gap.md](05-huong-dan-ky-thuat/xu-ly-su-co-thuong-gap.md) | Cẩm nang fix lỗi 403 Hugging Face và lỗi bảng mã UTF-8 trên Windows Excel |
| | [huong-dan-web-demo-xai.md](05-huong-dan-ky-thuat/huong-dan-web-demo-xai.md) | Xây dựng giao diện Streamlit và thuật toán tô màu từ ngữ vi phạm |
| **06. Theo dõi & Đánh giá** | [quyet-dinh-quan-trong.md](06-theo-doi-va-danh-gia/quyet-dinh-quan-trong.md) | Tổng hợp các quyết định kỹ thuật cốt lõi (ADR) đã được chốt |
| | [van-de-ton-dong-va-todo.md](06-theo-doi-va-danh-gia/van-de-ton-dong-va-todo.md) | Danh sách mục còn thiếu, cần xác nhận và khuyến nghị các bước tiếp theo |

---

## 3. Quy Ước Đánh Dấu Trạng Thái Nội Dung

Trong toàn bộ tài liệu, các thẻ phân định trạng thái thông tin được áp dụng nghiêm ngặt theo nguyên tắc kỹ thuật:

* `[CONFIRMED]`: Nội dung đã được thành viên nhóm xác nhận, áp dụng hoặc đã triển khai thành công.
* `[PROPOSED]`: Nội dung đề xuất kỹ thuật được nhóm chấp thuận đưa vào kế hoạch nâng cấp, đang chuẩn bị thực hiện theo lộ trình.
* `[TODO]`: Nhiệm vụ kỹ thuật cần nhóm tiếp tục triển khai hoặc kiểm thử.
* `[UNKNOWN]`: Thông tin kỹ thuật chưa có dữ liệu cụ thể trong lịch sử trao đổi, cần cả hai thành viên thảo luận và thống nhất thêm.
