# 📧 Phishing Email Detection System  
## Bảo mật thông tin trong Thương mại điện tử – Nhóm 2

---

## 1. Giới thiệu

Phishing email là một trong những hình thức tấn công phổ biến trong thương mại điện tử, nhằm đánh cắp thông tin đăng nhập, tài khoản ngân hàng hoặc dữ liệu nhạy cảm của người dùng.

Dự án này xây dựng hệ thống phát hiện email lừa đảo bằng cách áp dụng các mô hình Machine Learning và Deep Learning, đồng thời triển khai thành một ứng dụng web tương tác bằng Streamlit.

---

## 2. Mục tiêu

- Phân loại email thành **PHISHING** hoặc **LEGIT**
- So sánh hiệu năng giữa các mô hình ML truyền thống và Deep Learning
- Xây dựng hệ thống demo thực tế phục vụ mục đích học thuật

---

## 3. Dataset

Các tập dữ liệu sử dụng được lấy từ nguồn: https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset

---

## 4. Quy trình xử lý

### 4.1 Text Preprocessing

- Lowercase
- Remove punctuation
- Remove special characters
- Remove stopwords
- Lemmatization
- Text normalization

---

### 4.2 Feature Engineering

#### Đối với Machine Learning:
- **TF-IDF Vectorization**
- N-gram features
- Sparse feature matrix

#### Đối với Deep Learning:
- Tokenization
- Padding sequences
- Word index encoding
- Embedding layer

---

## 5. Mô hình sử dụng

### Machine Learning:
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- Naive Bayes

### Deep Learning:
- Bidirectional LSTM (Bi-LSTM)

---

## 6. Evaluation Metrics

Các chỉ số đánh giá:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

---

## 7. Kết quả thực nghiệm

| Model | Type | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|------|----------|-----------|--------|----------|---------|
| SVM | Machine Learning | **0.9872** | 0.9863 | 0.9892 | **0.9877** | **0.9992** |
| Bi-LSTM | Deep Learning | 0.9854 | 0.9842 | 0.9878 | 0.9860 | 0.9983 |
| Logistic Regression | Machine Learning | 0.9814 | 0.9802 | 0.9843 | 0.9823 | 0.9985 |
| Random Forest | Machine Learning | 0.9812 | 0.9816 | 0.9825 | 0.9820 | 0.9981 |
| Naive Bayes | Machine Learning | 0.9518 | 0.9809 | 0.9257 | 0.9525 | 0.9936 |

---

## 8. Nhận xét

- **SVM đạt hiệu suất cao nhất**, đặc biệt về Accuracy và F1-Score.
- Bi-LSTM có hiệu năng rất cạnh tranh, cho thấy khả năng học đặc trưng ngữ cảnh tốt.
- Naive Bayes có Recall thấp hơn, cho thấy hạn chế trong việc bắt đúng toàn bộ email phishing.
- Tất cả mô hình đều có ROC-AUC rất cao (>0.99), chứng tỏ khả năng phân biệt hai lớp rất tốt.

Kết quả cho thấy mô hình Machine Learning truyền thống (đặc biệt là SVM) vẫn có thể đạt hiệu suất rất cao khi kết hợp với TF-IDF.

---

## 9. Ứng dụng Web (Streamlit)

Hệ thống được triển khai thành web app cho phép:

- Nhập Subject và Body
- Load email mẫu phishing / không phishing
- So sánh nhiều mô hình cùng lúc
- Majority vote quyết định kết quả cuối cùng
- Hiển thị xác suất dự đoán

Chạy ứng dụng:

```streamlit run app.py```
## 10. Công nghệ sử dụng
- Python
- Scikit-learn
- TensorFlow / Keras
- Pandas / NumPy
- Streamlit
- Git & Git LFS
- ...

---

## 11. Kết luận
Hệ thống đạt độ chính xác cao (>98%) trong việc phát hiện phishing email.

Nghiên cứu cho thấy:
  TF-IDF + SVM là sự kết hợp rất mạnh cho bài toán phân loại văn bản.
  Deep Learning (Bi-LSTM) có khả năng khai thác ngữ cảnh tốt nhưng không vượt trội rõ ràng so với SVM trong bài toán này.

Dự án chứng minh việc áp dụng Machine Learning và Deep Learning vào lĩnh vực bảo mật thông tin trong thương mại điện tử là khả thi và hiệu quả.

---
# LỜI CẢM ƠN

Nhóm xin trân trọng cảm ơn TS. Nguyễn Mạnh Tuấn đã tận tình hướng dẫn, hỗ trợ chuyên môn và góp ý trong suốt quá trình thực hiện đề tài.

Xin cảm ơn các thành viên nhóm đã phối hợp và đóng góp tích cực:
- Thái Ngọc Bảo Châu (Trưởng nhóm)
- Nguyễn Thị Hải Anh
- Hoàng Gia Bảo
- Phạm Thị Thanh Lam
- Nguyễn Hà Hữu Luân
- Lê Như Thanh Tú
- Lương Gia Vĩ

Sự hỗ trợ và tinh thần làm việc nghiêm túc của tất cả các thành viên là yếu tố quan trọng giúp nhóm hoàn thành nghiên cứu này.
