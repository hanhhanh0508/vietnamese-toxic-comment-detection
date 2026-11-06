# 05. Hướng Dẫn Cài Đặt Môi Trường & Khởi Tạo Dự Án

## 1. Yêu Cầu Tiên Quyết (Prerequisites)

* **Hệ điều hành:** Hỗ trợ Windows 10/11, macOS hoặc Linux.
* **Phiên bản Python:** Python $3.9$ đến $3.11$ (khuyến nghị Python 3.10 hoặc 3.11 64-bit).
* **Git:** Đã cài đặt Git và cấu hình tài khoản cá nhân (`git config --global user.name` và `user.email`).
* **Trình soạn thảo:** Visual Studio Code (khuyên dùng kèm extension Python và Jupyter) hoặc PyCharm.

---

## 2. Các Bước Cài Đặt Môi Trường Chi Tiết

### Bước 1: Khởi tạo và kích hoạt Môi trường ảo (Virtual Environment)
Mở Terminal / PowerShell / Git Bash tại thư mục dự án `vietnamese-toxic-comment-detection`:

```bash
# 1. Di chuyển vào thư mục dự án
cd vietnamese-toxic-comment-detection

# 2. Khởi tạo môi trường ảo tên là 'venv'
python -m venv venv

# 3. Kích hoạt môi trường ảo
# Trên Windows (Command Prompt hoặc PowerShell):
venv\Scripts\activate

# Trên macOS / Linux:
source venv/bin/activate
```
> *Dấu hiệu thành công:* Bạn sẽ nhìn thấy tiền tố `(venv)` xuất hiện ở đầu dấu nhắc lệnh trong terminal.

---

### Bước 2: Danh sách phụ thuộc trong file `requirements.txt`
File `requirements.txt` tại thư mục gốc dự án chứa toàn bộ thư viện cần thiết:

```plaintext
# --- PyTorch & Deep Learning ---
torch>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0
scikit-learn>=1.2.0

# --- NLP & Tiền Xử Lý Tiếng Việt ---
underthesea>=6.0.0
pyvi>=0.1.1
emoji>=2.5.0

# --- Xử Lý Dữ Liệu & Trực Quan Hóa ---
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0

# --- Web Demo & Explainable AI (XAI) ---
streamlit>=1.25.0
lime>=0.2.0

# --- Jupyter Notebook Environment ---
jupyterlab
notebook
```

---

### Bước 3: Cài đặt các gói thư viện
Chạy lệnh cập nhật pip và cài đặt toàn bộ danh sách gói phụ thuộc:

```bash
# Nâng cấp công cụ cài đặt pip
python -m pip install --upgrade pip

# Cài đặt tất cả thư viện trong requirements.txt
pip install -r requirements.txt
```

---

### Bước 4: Kiểm tra trạng thái cài đặt & Hỗ trợ GPU CUDA
Tạo một lệnh kiểm tra nhanh để xác nhận PyTorch và CUDA (nếu máy có GPU NVIDIA):

```bash
python -c "import torch, transformers, underthesea; print('PyTorch Version:', torch.__version__); print('CUDA Available:', torch.cuda.is_available()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU Mode')"
```

---

## 3. Khởi Tạo Tự Động Khung Thư Mục Dự Án

Nếu cần tạo lại toàn bộ khung thư mục rỗng và các file mã nguồn ban đầu, có thể chạy tập lệnh sau:

### Trên Linux / macOS hoặc Git Bash:
```bash
mkdir -p assets data/raw data/processed data/dictionaries notebooks src app saved_models
touch src/__init__.py src/preprocessing.py src/utils.py src/models.py src/loss.py src/train.py
touch app/app.py app/highlight.py
```

### Trên Windows PowerShell:
```powershell
New-Item -ItemType Directory -Force -Path "assets", "data\raw", "data\processed", "data\dictionaries", "notebooks", "src", "app", "saved_models"
New-Item -ItemType File -Force -Path "src\__init__.py", "src\preprocessing.py", "src\utils.py", "src\models.py", "src\loss.py", "src\train.py", "app\app.py", "app\highlight.py"
```
