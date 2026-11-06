# 05. Cẩm Nang Xử Lý Sự Cố Thường Gặp (Troubleshooting Guide)

Tài liệu này ghi lại các lỗi thực tế phát sinh trong quá trình chuẩn bị dữ liệu Tuần 1 và phương án giải quyết đã được thực nghiệm thành công.

---

## 1. Sự Cố 1: Lỗi 403 Forbidden Khi Truy Cập Dataset Trên Hugging Face

### Hiện tượng & Thông báo lỗi:
Khi truy cập liên kết `https://huggingface.co/datasets/uitnlp/vihsd` hoặc gọi qua thư viện `datasets`, hệ thống trả về thông báo lỗi:
```text
403 Client Error: Forbidden for url: https://huggingface.co/datasets/uitnlp/vihsd
Access to dataset uitnlp/vihsd is restricted.
```

### Nguyên nhân gốc rễ (Root Cause):
Bộ dữ liệu `ViHSD` do phòng nghiên cứu UIT-NLP cấu hình dưới dạng **Gated Dataset** (kho dữ liệu có cổng kiểm soát). Hugging Face yêu cầu người dùng phải đăng nhập tài khoản và xác nhận điều khoản nghiên cứu học thuật trước khi cấp quyền tải xuống.

### Phương án giải quyết:

#### Cách 1: Tải trực tiếp từ Repository GitHub (Khuyên dùng - Nhanh nhất)
Không phụ thuộc vào cơ chế kiểm duyệt tài khoản của Hugging Face, nhóm có thể tải trực tiếp các file CSV thô từ các kho GitHub công khai:
* **ViHSD Repo:** [https://github.com/sonlam1102/vihsd](https://github.com/sonlam1102/vihsd) $\to$ Tải `train.csv`, `dev.csv` (Validation), `test.csv`.
* **ViCTSD Repo:** [https://github.com/tarudesu/ViCTSD](https://github.com/tarudesu/ViCTSD)
* **ViTHSD Repo:** [https://github.com/bakansm/ViTHSD](https://github.com/bakansm/ViTHSD)

#### Cách 2: Xin cấp quyền chính thức trên Hugging Face & Dùng Token
1. Đăng nhập vào Hugging Face và truy cập trang dataset: [https://huggingface.co/datasets/uitnlp/vihsd](https://huggingface.co/datasets/uitnlp/vihsd).
2. Tích chọn đồng ý các điều khoản sử dụng nghiên cứu và gửi yêu cầu (Access Request).
3. Sau khi được duyệt, chuyển sang tab **Files and versions** để tải trực tiếp từng file:
   * `train.csv` (1.62 MB)
   * `dev.csv` (177 kB - tập validation)
   * `test.csv` (442 kB)
4. Nếu dùng Python script để tải tự động, cấu hình Hugging Face Access Token:
   ```python
   from datasets import load_dataset
   dataset = load_dataset("uitnlp/vihsd", token="hf_your_access_token_here")
   ```

---

## 2. Sự Cố 2: Lỗi Bảng Mã Tiếng Việt Bị Biến Thành Ký Tự Lạ (Mojibake)

### Hiện tượng:
Khi mở file `train.csv` bằng Microsoft Excel trên Windows, toàn bộ chữ tiếng Việt có dấu bị biến thành chuỗi ký tự rác vô nghĩa:
* `Coi cười` $\rightarrow$ `Coi cÆ°á» i`
* `Nó góp` $\rightarrow$ `NẤ³ gÃ³p`
* `chậm cái địt mẹ` $\rightarrow$ `cháº-m cÃji Äá»t máº¹`

### Nguyên nhân gốc rễ:
File dữ liệu được lưu trữ chuẩn bảng mã quốc tế **UTF-8**. Tuy nhiên, Microsoft Excel trên Windows khi mở trực tiếp file CSV qua thao tác nhấp đúp chuột (Double click) sẽ mặc định giải mã theo bảng mã **ANSI / Windows-1252**, dẫn đến hiện tượng lệch bảng mã (Mojibake).

### Phương án giải quyết:

#### Cách 1: Đọc bằng Python Pandas với tham số `encoding='utf-8'` (Chuẩn dự án)
Dữ liệu gốc hoàn toàn nguyên vẹn, không bị hỏng. Khi đọc trong mã nguồn Python, luôn chỉ định rõ tham số mã hóa UTF-8:

```python
import pandas as pd

# Luôn chỉ định encoding="utf-8"
df = pd.read_csv("data/raw/train.csv", encoding="utf-8")

# Hiển thị 5 dòng đầu kiểm tra tiếng Việt
for idx, row in df.head(5).iterrows():
    print(f"[{idx+1}] Text: {row['free_text']} | Label: {row['label_id']}")
```

#### Cách 2: Thiết lập hiển thị chuẩn tiếng Việt trên Microsoft Excel
Nếu cần xem và lọc dữ liệu bằng tay trên Excel:
1. Mở một bảng tính Excel trắng (Blank Workbook).
2. Vào thẻ **Data** trên thanh Ribbon $\to$ chọn **From Text/CSV** (hoặc *Get Data* $\to$ *From File* $\to$ *From Text/CSV*).
3. Chọn file `train.csv`.
4. Trong cửa sổ xem trước, tại mục **File Origin (Nguồn tệp)**, chuyển từ *ANSI/Windows-1252* sang **`65001 : Unicode (UTF-8)`**.
5. Bấm **Load**. Toàn bộ tiếng Việt sẽ hiển thị chuẩn xác 100%.

---

## 3. Script Kiểm Thử Dữ Liệu Nhanh (`test_data.py`)

Nhóm đã sử dụng đoạn script kiểm tra nhanh này tại thư mục gốc để xác thực tính toàn vẹn và phân bố nhãn của file dữ liệu thô:

```python
import pandas as pd

file_path = "data/raw/train.csv"

try:
    df = pd.read_csv(file_path, encoding="utf-8")
    print("=== TẢI DỮ LIỆU THÀNH CÔNG ===")
    print(f"Tổng số dòng: {len(df)}")
    
    print("\n=== THỐNG KÊ PHÂN BỐ NHÃN GỐC (label_id) ===")
    print(df["label_id"].value_counts())
    
    print("\n=== TỶ LỆ PHẦN TRĂM (%) ===")
    print(df["label_id"].value_counts(normalize=True) * 100)
    
    print("\n=== 5 DÒNG ĐẦU TIÊN (TIẾNG VIỆT CHUẨN) ===")
    for idx, row in df.head(5).iterrows():
        print(f"[{idx+1}] Label: {row['label_id']} | Text: {row['free_text']}")

except Exception as e:
    print("Có lỗi xảy ra khi kiểm tra dữ liệu:", e)
```
