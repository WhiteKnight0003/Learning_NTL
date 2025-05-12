from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

# Dữ liệu ví dụ - mở rộng thêm để minh họa
documents = [
    "Tôi thích trứng và mèo",
    "Tôi ghét mèo",
    "Tôi thích trứng và tôi thích mèo",
    "Tôi thích ăn trứng vào buổi sáng",
    "Tôi nuôi hai con mèo ở nhà",
    "Tôi rất ghét khi trời mưa",
    "Mèo của tôi thích ăn cá",
    "Tôi không thích chó nhưng thích mèo",
    "Tôi ghét chuột nhưng thích mèo",
    "Trời mưa làm tôi buồn"
]

# Gán nhãn cho từng document
# Giả sử ta có 3 nhãn: 
# 0: nói về ghét (Negative)
# 1: nói về thích (Positive)
# 2: nói về mèo (Cat)
labels = [1, 0, 1, 1, 2, 0, 2, 1, 1, 0]

# Tạo DataFrame để dễ theo dõi
df = pd.DataFrame({
    'Document': documents,
    'Label': labels,
    'Label_Name': ['Positive', 'Negative', 'Positive', 'Positive', 'Cat', 
                   'Negative', 'Cat', 'Positive', 'Positive', 'Negative']
})

print("Dữ liệu huấn luyện:")
print(df)
print("\n" + "="*70 + "\n")

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(
    documents, labels, test_size=0.3, random_state=42)

print(f"Số lượng mẫu huấn luyện: {len(X_train)}")
print(f"Số lượng mẫu kiểm tra: {len(X_test)}")

# Xây dựng pipeline cho quá trình phân loại
text_clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())  # Sử dụng mô hình Naive Bayes
])

# Huấn luyện mô hình
text_clf.fit(X_train, y_train)

# Kiểm tra độ chính xác trên tập kiểm tra
accuracy = text_clf.score(X_test, y_test)
print(f"\nĐộ chính xác trên tập kiểm tra: {accuracy:.2f}")

# Dự đoán một số câu mới
new_documents = [
    "Tôi thích ăn trứng",
    "Tôi ghét mưa",
    "Mèo của tôi màu đen",
    "Tôi rất thích mèo nhưng ghét chuột"
]

print("\n" + "="*70)
print("Dự đoán nhãn cho những câu mới:")
print("="*70)

# Dự đoán và hiển thị kết quả
predicted_labels = text_clf.predict(new_documents)
predicted_proba = text_clf.predict_proba(new_documents)

# Ánh xạ chỉ số nhãn với tên nhãn
label_names = {0: "Negative", 1: "Positive", 2: "Cat"}

for i, doc in enumerate(new_documents):
    print(f"Câu: '{doc}'")
    print(f"Nhãn dự đoán: {predicted_labels[i]} ({label_names[predicted_labels[i]]})")
    
    # Hiển thị xác suất cho từng nhãn
    print("Xác suất cho từng nhãn:")
    for j, prob in enumerate(predicted_proba[i]):
        print(f"  - {label_names[j]}: {prob:.4f}")
    print()

print("\n" + "="*70 + "\n")
print("PHÂN TÍCH VECTOR")
print("="*70)

# Hiển thị vector của một câu mới
vectorizer = CountVectorizer()
vectorizer.fit(documents)  # Fit với tất cả tài liệu huấn luyện

# Lấy từ điển
vocabulary = vectorizer.get_feature_names_out()
print(f"Từ điển: {vocabulary}")

# Tạo vector cho câu mới
new_doc = "Tôi thích mèo màu trắng"
new_vector = vectorizer.transform([new_doc]).toarray()

print(f"\nCâu mới: '{new_doc}'")
print(f"Vector: {new_vector[0]}")

# Hiển thị chi tiết hơn
print("\nChi tiết vector:")
for i, word in enumerate(vocabulary):
    if new_vector[0][i] > 0:
        print(f"  - '{word}': {new_vector[0][i]}")

# Dự đoán nhãn cho câu mới này
predicted_label = text_clf.predict([new_doc])[0]
print(f"\nNhãn dự đoán: {predicted_label} ({label_names[predicted_label]})")

print("\n" + "="*70)
print("Kết luận:")
print("1. Chúng ta đã chuyển đổi các document thành vector")
print("2. Huấn luyện mô hình phân loại với các vector này")
print("3. Với document mới, ta chuyển nó thành vector, rồi dự đoán nhãn")
print("="*70)