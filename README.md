# Quishing Detection using Machine Learning

## Giới thiệu

Dự án này tập trung vào bài toán **phát hiện quishing** (QR phishing) bằng **học máy**, dựa trên việc phân tích **ảnh mã QR** thay vì giải mã trực tiếp nội dung bên trong.

Mục tiêu của đề tài là:
- Xây dựng pipeline phát hiện QR code độc hại
- Thực nghiệm với nhiều mô hình học máy
- So sánh hiệu năng giữa các mô hình
- Thực hiện feature selection và explainability
- Xây dựng phần demo tạo QR và dự đoán nhãn

> **Lưu ý:** Phần demo chỉ mang tính chất minh họa ứng dụng, không phải đánh giá chính thức của mô hình.

---

## Mục tiêu đề tài

- Phát hiện QR code **benign** và **malicious**
- Chuẩn hóa dữ liệu QR về cùng định dạng
- Huấn luyện và đánh giá nhiều mô hình ML
- Áp dụng **GridSearchCV** và **RandomizedSearchCV**
- Phân tích độ quan trọng của đặc trưng
- Tạo ứng dụng demo để kiểm tra QR sinh ra

---

## Ý tưởng chính

Thay vì:
1. Quét mã QR
2. Giải mã nội dung
3. Kiểm tra URL/payload

Hệ thống sẽ:
1. Đọc ảnh QR
2. Tiền xử lý ảnh
3. Flatten ảnh thành vector đặc trưng
4. Dự đoán trực tiếp nhãn:
   - `0`: Benign
   - `1`: Malicious

Cách tiếp cận này giúp hỗ trợ phát hiện rủi ro từ sớm, trước khi truy cập nội dung QR.

---

## Dataset

Dataset được xây dựng từ các URL hợp lệ và độc hại, sau đó chuyển thành ảnh QR với cấu hình chuẩn:

- **QR Version:** 13
- **Kích thước:** 69 × 69
- **Error Correction:** Low
- **Border:** 0
- **Box Size:** 1

Sau khi sinh QR code, mỗi ảnh được chuyển về dạng grayscale và flatten để đưa vào mô hình học máy.

---

## Pipeline

```text
Load Dataset
    ↓
EDA
    ↓
Preprocessing
    ↓
Flatten QR Image (69×69 → 4761 features)
    ↓
Train/Test Split
    ↓
Train baseline models
    ↓
GridSearchCV / RandomizedSearchCV
    ↓
Evaluation
    ↓
Feature Importance / Explainability
    ↓
Feature Selection
    ↓
Retrain models
    ↓
Demo prediction

````markdown
# Quishing Detection using Machine Learning

## Giới thiệu

Dự án này tập trung vào bài toán **phát hiện quishing** (QR phishing) bằng **học máy**, dựa trên việc phân tích **ảnh mã QR** thay vì giải mã trực tiếp nội dung bên trong.

Mục tiêu của đề tài là:
- Xây dựng pipeline phát hiện QR code độc hại
- Thực nghiệm với nhiều mô hình học máy
- So sánh hiệu năng giữa các mô hình
- Thực hiện feature selection và explainability
- Xây dựng phần demo tạo QR và dự đoán nhãn

> **Lưu ý:** Phần demo chỉ mang tính chất minh họa ứng dụng, không phải đánh giá chính thức của mô hình.

---

## Mục tiêu đề tài

- Phát hiện QR code **benign** và **malicious**
- Chuẩn hóa dữ liệu QR về cùng định dạng
- Huấn luyện và đánh giá nhiều mô hình ML
- Áp dụng **GridSearchCV** và **RandomizedSearchCV**
- Phân tích độ quan trọng của đặc trưng
- Tạo ứng dụng demo để kiểm tra QR sinh ra

---

## Ý tưởng chính

Thay vì:
1. Quét mã QR
2. Giải mã nội dung
3. Kiểm tra URL/payload

Hệ thống sẽ:
1. Đọc ảnh QR
2. Tiền xử lý ảnh
3. Flatten ảnh thành vector đặc trưng
4. Dự đoán trực tiếp nhãn:
   - `0`: Benign
   - `1`: Malicious

Cách tiếp cận này giúp hỗ trợ phát hiện rủi ro từ sớm, trước khi truy cập nội dung QR.

---

## Dataset

Dataset được xây dựng từ các URL hợp lệ và độc hại, sau đó chuyển thành ảnh QR với cấu hình chuẩn:

- **QR Version:** 13
- **Kích thước:** 69 × 69
- **Error Correction:** Low
- **Border:** 0
- **Box Size:** 1

Sau khi sinh QR code, mỗi ảnh được chuyển về dạng grayscale và flatten để đưa vào mô hình học máy.

---

## Pipeline

```text
Load Dataset
    ↓
EDA
    ↓
Preprocessing
    ↓
Flatten QR Image (69×69 → 4761 features)
    ↓
Train/Test Split
    ↓
Train baseline models
    ↓
GridSearchCV / RandomizedSearchCV
    ↓
Evaluation
    ↓
Feature Importance / Explainability
    ↓
Feature Selection
    ↓
Retrain models
    ↓
Demo prediction
````

---

## Các mô hình sử dụng

Dự án thực nghiệm với các mô hình:

* Logistic Regression
* Decision Tree
* Random Forest
* Gaussian Naive Bayes
* LightGBM
* XGBoost

Ngoài ra có:

* Huấn luyện **không dùng Cross Validation**
* Huấn luyện **có Cross Validation** với các fold:

  * 5-fold
  * 10-fold
  * 15-fold
  * 20-fold

---

## Các chỉ số đánh giá

Các metric được sử dụng:

* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC

Ngoài ra còn có:

* Confusion Matrix
* ROC Curve
* Precision-Recall Curve

---

## Feature Importance và Feature Selection

Sau khi huấn luyện, các mô hình tree-based như:

* Random Forest
* LightGBM
* XGBoost

được dùng để:

* Phân tích mức độ quan trọng của từng pixel
* Xác định vùng ảnh có ảnh hưởng lớn tới dự đoán
* Thực hiện chọn lọc đặc trưng để giảm chiều dữ liệu
* Huấn luyện lại mô hình trên tập đặc trưng đã chọn

---

## Cấu trúc thư mục

```text
project/
│
├── Dataset/
│   ├── qr_codes_29.pickle
│   └── qr_codes_29_labels.pickle
│
├── outputs_quishing_merged/
│   ├── app/
│   ├── eda/
│   ├── explainability/
│   ├── generated_qr/
│   ├── models/
│   ├── plots/
│   └── results/
│
├── Detect_Quishing_Done_Fold10.ipynb
├── 2_Trainning_Model.ipynb
├── quishing_detect_final.py
└── README.md
```

---

## Cài đặt môi trường

### Yêu cầu

* Python 3.10+
* pip hoặc conda

### Cài thư viện

```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm joblib pillow qrcode[pil]
pip install shap streamlit ipywidgets
```

---

## Cách chạy notebook

Mở notebook và chạy lần lượt các cell theo thứ tự:

1. Import thư viện
2. Cấu hình đường dẫn dữ liệu
3. Load dataset
4. EDA
5. Tiền xử lý
6. Huấn luyện mô hình
7. Hyperparameter tuning
8. Đánh giá kết quả
9. Explainability / Feature selection
10. Demo QR prediction

---

## Cách chạy file Python

```bash
python quishing_detect_final.py
```

> Nhớ kiểm tra lại đường dẫn dataset trước khi chạy.

---

## Kết quả đầu ra

Sau khi chạy, hệ thống sẽ sinh ra:

* Model đã train (`.joblib`)
* Bảng kết quả metric (`.csv`)
* Biểu đồ EDA
* Confusion matrix
* ROC / PR curve
* Heatmap feature importance
* QR code minh họa
* File app demo Streamlit

---

## Demo ứng dụng

Phần demo hỗ trợ:

* Tạo QR benign
* Tạo QR malicious mô phỏng
* Đưa QR vào mô hình để dự đoán
* Hiển thị kết quả phân loại

Ví dụ lệnh chạy Streamlit:

```bash
streamlit run outputs_quishing_merged/app/app_qr_demo_streamlit.py
```

---

## Điểm nổi bật

* Phát hiện quishing mà không cần giải mã nội dung QR trước
* Hỗ trợ nhiều mô hình học máy
* Có cả train thường và train với cross-validation
* Có GridSearchCV và RandomizedSearchCV
* Có explainability và feature selection
* Có lưu model để tái sử dụng
* Có demo minh họa ứng dụng thực tế

---

## Hạn chế

* Dataset hiện chủ yếu xoay quanh QR chứa URL
* Chưa mở rộng mạnh sang các payload khác như WiFi, SMS, contact card
* Flatten ảnh làm giảm thông tin không gian 2D
* Chưa so sánh sâu với các mô hình deep learning như CNN/ViT
* Chưa đánh giá trên ảnh QR bị méo, nhiễu hoặc chụp từ môi trường thực tế

---

## Hướng phát triển

* Thử nghiệm CNN, ResNet, Vision Transformer
* Kết hợp đặc trưng ảnh và đặc trưng nội dung
* Mở rộng dataset cho nhiều loại QR payload
* Đánh giá trên dữ liệu QR thực tế từ camera
* Xây dựng ứng dụng quét QR thời gian thực

---

## Tài liệu tham khảo

1. Fouad Trad, Ali Chehab, *Detecting Quishing Attacks with Machine Learning Techniques Through QR Code Analysis*, 2025.
2. PhishStorm Dataset
3. Báo cáo môn học của nhóm về phát hiện quishing bằng học máy

---

## Nhóm thực hiện

* Vũ Thị Diệu Anh
* Diệp Kim Chi
* Huỳnh Trọng Nghĩa
* Võ Minh Nhật

**Giảng viên hướng dẫn:**
TS. Nguyễn An Khương

---

## Ghi chú

Repository này phục vụ cho mục đích:

* Học tập
* Nghiên cứu
* Thực nghiệm lại phương pháp từ bài báo

Không sử dụng cho mục đích tấn công hoặc phát tán nội dung độc hại.

