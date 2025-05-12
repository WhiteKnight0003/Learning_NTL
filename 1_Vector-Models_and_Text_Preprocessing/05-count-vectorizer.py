from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np

# Tạo dữ liệu ví dụ
documents = [
    "Tôi thích trứng và mèo",
    "Tôi ghét mèo",
    "Tôi thích trứng và tôi thích mèo"
]

# Tạo DataFrame để hiển thị dữ liệu gốc
df = pd.DataFrame({
    'ID': [1, 2, 3],
    'Nội dung bài viết (Document)': documents
})

print("Dữ liệu ban đầu:")
print(df)
print("\n" + "="*50 + "\n")

# Tạo và fit CountVectorizer
count_vectorizer = CountVectorizer()
count_matrix = count_vectorizer.fit_transform(documents)

# Lấy danh sách các từ trong từ điển theo thứ tự mặc định (a->z)
vocabulary = count_vectorizer.get_feature_names_out()
print(f"Thứ tự các từ trong từ điển (a->z): {vocabulary}")

# Hiển thị từ điển với chỉ số
vocabulary_dict = count_vectorizer.vocabulary_
print("\nTừ điển với chỉ số:")
for word, idx in sorted(vocabulary_dict.items()):
    print(f"{word}: {idx}")

# Hiển thị ma trận kết quả
print("\nMa trận kết quả (Document-Term Matrix):")
count_array = count_matrix.toarray()
count_df = pd.DataFrame(count_array, columns=vocabulary)
count_df.index = [f"Document {i+1}" for i in range(len(documents))]
print(count_df)

# Hiển thị các vector cho từng document
print("\nVector cho từng document:")
for i, doc in enumerate(documents):
    print(f"Document {i+1}: {doc}")
    print(f"Vector: {count_array[i]}")
    print()

print("\n" + "="*50)
print("Giải thích:")
print("- Mỗi hàng trong ma trận tương ứng với một document")
print("- Mỗi cột tương ứng với một từ trong từ điển theo thứ tự bảng chữ cái")
print("- Giá trị trong ma trận là số lần xuất hiện của từ trong document")
print("="*50)