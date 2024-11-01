import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.metrics import davies_bouldin_score
from collections import Counter

# Load dữ liệu Iris
data = load_iris()
X = data.data
y = data.target

# Bước 1: Khởi tạo các tâm cụm ban đầu
def initialize_centroids(X, k):
    np.random.seed(42)
    indices = np.random.choice(X.shape[0], k, replace=False)
    centroids = X[indices]
    print("Tâm cụm khởi tạo:", centroids)
    return centroids

# Bước 2: Gán mỗi điểm dữ liệu vào cụm gần nhất
def assign_clusters(X, centroids):
    clusters = []
    print("\nGán điểm vào cụm gần nhất:")
    for idx, point in enumerate(X):
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        cluster = np.argmin(distances)
        clusters.append(cluster)
        print(f"Điểm {idx} ({point}) -> cụm {cluster}, khoảng cách {distances}")
    return clusters

# Bước 3: Tính lại các tâm cụm
def update_centroids(X, clusters, k):
    new_centroids = np.zeros((k, X.shape[1]))
    print("\nCập nhật tâm cụm:")
    for i in range(k):
        points_in_cluster = X[np.array(clusters) == i]
        if len(points_in_cluster) > 0:
            new_centroids[i] = points_in_cluster.mean(axis=0)
            print(f"Cụm {i}: {points_in_cluster} -> Tâm cụm mới {new_centroids[i]}")
        else:
            new_centroids[i] = X[np.random.choice(X.shape[0])]
            print(f"Cụm {i} không có điểm nào, chọn tâm mới ngẫu nhiên: {new_centroids[i]}")
    return new_centroids

# Thuật toán K-means
def kmeans(X, k, max_iter=100):
    centroids = initialize_centroids(X, k)
    for i in range(max_iter):
        print(f"\n--- Vòng lặp {i + 1} ---")
        clusters = assign_clusters(X, centroids)
        new_centroids = update_centroids(X, clusters, k)
        if np.all(centroids == new_centroids):
            print("Tâm cụm không thay đổi, dừng lại.")
            break
        centroids = new_centroids
    return clusters

# Số cụm bằng số lượng nhãn thực tế
k = len(np.unique(y))
clusters = kmeans(X, k)

# Chuyển đổi nhãn cho mỗi cụm sang nhãn thật để đánh giá
def convert_labels(clusters, true_labels):
    label_map = {}
    for cluster in np.unique(clusters):
        idx = np.where(np.array(clusters) == cluster)[0]
        true_labels_in_cluster = true_labels[idx]
        most_common = Counter(true_labels_in_cluster).most_common(1)[0][0]
        label_map[cluster] = most_common
    predicted_labels = [label_map[cluster] for cluster in clusters]
    print("\nGán nhãn dự đoán theo nhãn thật:", predicted_labels)
    return predicted_labels

predicted_labels = convert_labels(clusters, y)

# Đánh giá các chỉ số

# 1. F1-score
f1 = f1_score(y, predicted_labels, average='weighted')
print("\nChỉ số F1-score (trung bình có trọng số):", f1)

# 2. RAND index
rand_index = adjusted_rand_score(y, predicted_labels)
print("Chỉ số RAND index:", rand_index)

# 3. Normalized Mutual Information (NMI)
nmi = normalized_mutual_info_score(y, predicted_labels)
print("Chỉ số NMI (Normalized Mutual Information):", nmi)

# 4. Davies-Bouldin index (DB index)
db_index = davies_bouldin_score(X, clusters)
print("Chỉ số Davies-Bouldin Index:", db_index)
