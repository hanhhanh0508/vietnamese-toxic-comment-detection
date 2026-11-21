|  |
| --- |
| **Giai đoạn thực hiện:** Tuần 3 — Tinh chỉnh mô hình Pre-trained Language Models (PhoBERT & ViDeBERTa) |
| **Kho mã nguồn (GitHub):** github.com/hanhhanh0508/vietnamese-toxic-comment-detection |
| **Nhánh làm việc Tuần 3:** sv1/tuan3 & sv2/tuan3 -> Merge vào main |

**1. Mục tiêu trọng tâm Tuần 3**

* **Chuyển đổi biểu diễn ngôn ngữ sang Subword Tokenization:** Nâng cấp từ phương pháp tách từ/đếm từ truyền thống sang bộ mã hóa chuyên biệt cho Transformer tiếng Việt gồm PhoBERT-base-v2 (vinai/phobert-base-v2) và ViDeBERTa-base (Fsoft-AIC/videberta-base).
* **Kế thừa hàm mất mát xử lý mất cân bằng dữ liệu:** Tái sử dụng module BinaryFocalLoss (gamma = 2.0) kết hợp vector trọng số lớp pos\_weights từ src/loss.py đã kiểm chứng ở Tuần 2 để huấn luyện ổn định các mô hình ngôn ngữ lớn.
* **Tối ưu hóa ngưỡng quyết định (Threshold Tuning):** Tìm kiếm bộ ngưỡng xác suất tối ưu riêng cho từng nhãn [toxic, hate, spam, clean] trên tập Validation (val.csv) nhằm tối đa hóa chỉ số Macro-F1 trước khi chấm điểm khách quan trên tập Test (test.csv).
* **Tổng hợp đối sánh toàn diện 4 kiến trúc mô hình:** Xây dựng bảng đánh giá và biểu đồ trực quan so sánh hiệu năng xuyên suốt giữa TF-IDF + LinearSVC, BiLSTM-Attention, PhoBERT-base-v2 và ViDeBERTa-base.

**2. Bảng phân công nhiệm vụ chi tiết (2 thành viên)**

|  |  |  |
| --- | --- | --- |
| **Hạng mục công việc** | **Sinh viên 1 (SV1) Phụ trách PhoBERT-base-v2 & Tối ưu ngưỡng** | **Sinh viên 2  Phụ trách ViDeBERTa-base & Kiến trúc lõi** |
| **1. Tiền xử lý & Tokenization** | • Đảm bảo văn bản đầu vào giữ đúng định dạng tách từ tiếng Việt bằng underthesea (ví dụ: nhà\_nước, vô\_học) theo yêu cầu bắt buộc của PhoBERT. • Sử dụng AutoTokenizer.from\_pretrained('vinai/phobert-base-v2') với max\_len = 128. | • Cấu hình bộ mã hóa SentencePiece của ViDeBERTa ('Fsoft-AIC/videberta-base') trên văn bản tiếng Việt tự nhiên (tự động xử lý khoảng trắng, không cần nối dấu gạch dưới). • Xây dựng module dùng chung src/transformer\_dataset.py hỗ trợ nạp động tokenizer theo tên mô hình. |
| **2. Kiến trúc mạng & Hàm mất mát** | • Tái sử dụng BinaryFocalLoss(gamma=2.0, pos\_weight=pos\_weights) từ src/loss.py và compute\_macro\_f1 từ src/metrics.py. • Cấu hình tham số fine-tune chuẩn cho PhoBERT: lr = 2e-5, weight\_decay = 0.01, batch\_size = 16, epochs = 4–5, Linear Warmup 10%. | • Xây dựng class MultiLabelTransformer(nn.Module) trong src/transformer\_model.py kết hợp AutoModel + Mean/CLS Pooling + Dropout(0.3) + Linear(hidden\_size, 4). • Tích hợp bộ tối ưu AdamW và lịch điều chỉnh tốc độ học get\_linear\_schedule\_with\_warmup. |
| **3. Huấn luyện GPU & Tinh chỉnh ngưỡng** | • Tạo và chạy thực nghiệm trên GPU T4 (Google Colab) qua file notebooks/04\_finetune\_phobert.ipynb. • Viết hàm dò tìm bộ ngưỡng tối ưu (từ 0.25 đến 0.75) cho từng nhãn trên tập val.csv và lưu ra configs/optimal\_thresholds.json. | • Tạo và chạy thực nghiệm trên GPU T4 (Google Colab) qua file notebooks/05\_finetune\_videberta.ipynb. • Áp dụng quy trình tinh chỉnh ngưỡng trên val.csv và đánh giá độc lập trên test.csv, xuất ma trận nhầm lẫn đa nhãn. |
| **4. Lưu trữ & Báo cáo tổng hợp** | • Lưu trọng số tốt nhất phobert\_best.pt lên thư mục Google Drive chung của nhóm. • Xuất biểu đồ tổng hợp so sánh Macro-F1 và F1 từng nhãn của cả 4 mô hình vào assets/final\_model\_comparison.png. | • Lưu trọng số tốt nhất videberta\_best.pt lên Google Drive chung của nhóm. • Phân tích lỗi sai đối chiếu với BiLSTM Tuần 2, tổng hợp bảng số liệu và cập nhật tài liệu phan-cong-tuan-3.md. |

**3. Lịch trình thực hiện chi tiết theo ngày (7 ngày)**

|  |  |  |
| --- | --- | --- |
| **Mốc thời gian** | **Nội dung công việc trọng tâm** | **Sản phẩm đầu ra (Deliverables)** |
| **Ngày 1 – Ngày 2** | • Kéo code mới nhất từ nhánh main về máy cá nhân (git pull origin main) và tách nhánh sv1/tuan3, sv2/tuan3. • SV2 hoàn thiện mã nguồn lõi src/transformer\_dataset.py và src/transformer\_model.py, đẩy lên GitHub để SV1 dùng chung. | • Nhánh sv1/tuan3, sv2/tuan3 • src/transformer\_dataset.py • src/transformer\_model.py |
| **Ngày 3 – Ngày 4** | • Đồng bộ thư mục src/ và data/processed/ lên Google Drive. • Chạy huấn luyện fine-tune PhoBERT-base-v2 (SV1) và ViDeBERTa-base (SV2) trên GPU T4 của Google Colab (4–5 epochs). | • Trọng số phobert\_best.pt & videberta\_best.pt trên Drive • Log quá trình huấn luyện |
| **Ngày 5** | • Chạy thuật toán tìm ngưỡng tối ưu (Threshold Tuning) cho từng nhãn trên tập val.csv. • Chấm điểm trên tập test.csv, vẽ ma trận nhầm lẫn và lưu toàn bộ output vào 2 file Jupyter Notebook. | • configs/optimal\_thresholds.json • 04\_finetune\_phobert.ipynb • 05\_finetune\_videberta.ipynb |
| **Ngày 6 – Ngày 7** | • Phân tích các trường hợp lỗi sai điển hình và so sánh cải thiện so với BiLSTM Tuần 2. • Xuất biểu đồ tổng hợp 4 mô hình, tạo Pull Request merge vào nhánh main và hoàn thiện báo cáo tiến độ. | • assets/final\_model\_comparison.png • phan-cong-tuan-3.md • Merge hoàn chỉnh vào main |

**4. Cấu trúc thư mục cập nhật sau Tuần 3**

|  |
| --- |
| vietnamese-toxic-comment-detection/ ├── assets/ │ ├── phobert\_confusion\_matrices.png # [SV1] Ma trận nhầm lẫn PhoBERT-base-v2 │ ├── videberta\_confusion\_matrices.png # [SV2] Ma trận nhầm lẫn ViDeBERTa-base │ └── final\_model\_comparison.png # [SV1 & SV2] Biểu đồ so sánh toàn diện 4 mô hình ├── configs/ │ └── optimal\_thresholds.json # [SV1 & SV2] Bộ ngưỡng tối ưu cho 4 nhãn ├── notebooks/ │ ├── 04\_finetune\_phobert.ipynb # [SV1] Notebook huấn luyện & đánh giá PhoBERT │ └── 05\_finetune\_videberta.ipynb # [SV2] Notebook huấn luyện & đánh giá ViDeBERTa ├── project-docs/04-ke-hoach-va-phan-cong/ │ └── phan-cong-tuan-3.md # [SV2] Tài liệu phân công chi tiết Tuần 3 └── src/  ├── transformer\_dataset.py # [SV2] PyTorch Dataset hỗ trợ PhoBERT & ViDeBERTa  └── transformer\_model.py # [SV2] Kiến trúc MultiLabelTransformer |

|  |
| --- |
| **LƯU Ý KỸ THUẬT & QUY TẮC QUẢN LÝ MÃ NGUỒN (GIT / GOOGLE DRIVE)**  1. Môi trường GPU: Fine-tune PhoBERT và ViDeBERTa cần chạy trên GPU T4 (Google Colab hoặc Kaggle) để đạt tốc độ ~8–12 phút/mô hình; tránh chạy trực tiếp trên CPU máy cá nhân.  2. Quản lý trọng số mô hình (.pt / .bin): File trọng số của Transformer có dung lượng lớn (500 MB – 1.5 GB). Bắt buộc thêm \*.pt và \*.bin vào file .gitignore và chỉ lưu trữ trên Google Drive chung của nhóm, tuyệt đối không đẩy lên GitHub.  3. Khác biệt đầu vào Tokenizer: PhoBERT yêu cầu văn bản đã tách từ ghép bằng dấu gạch dưới (nhà\_nước), trong khi ViDeBERTa sử dụng văn bản tiếng Việt tự nhiên (nhà nước) — cần xử lý tách bạch trong src/transformer\_dataset.py. |