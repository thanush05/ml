import numpy as np
import matplotlib.pyplot as plt
file_path = "./data/data4_19.csv"
data = np.loadtxt(file_path, delimiter=',', usecols=(0, 1, 2, 3))
K = 3

centroids = data[np.random.choice(data.shape[0], K, replace=False)]
max_iterations = 10

ground_truth_labels = np.loadtxt(file_path, delimiter=',', usecols=(4,), dtype=str)

jaccard_distances = np.zeros(K)

for iteration in range(max_iterations):
    distances = np.sqrt(np.sum((data[:, np.newaxis] - centroids) ** 2, axis=2))

    labels = np.argmin(distances, axis=1)

    new_centroids = np.array([data[labels == k].mean(axis=0) for k in range(K)])

    if np.all(centroids == new_centroids):
        break

    centroids = new_centroids

print("Final cluster means:")
for k in range(K):
    print(f"Cluster {k + 1}: {centroids[k]}")

for k in range(K):
    cluster_indices = np.where(labels == k)[0]
    ground_truth_cluster_indices = np.where(ground_truth_labels == f'Iris {k + 1}')[0]
    intersection = len(np.intersect1d(cluster_indices, ground_truth_cluster_indices))
    union = len(np.union1d(cluster_indices, ground_truth_cluster_indices))
    jaccard_distances[k] = 1.0 - (intersection / union)

print("\nJaccard distances:")
for k in range(K):
    print(f"Cluster {k + 1}: {jaccard_distances[k]:.4f}")

# Create a scatter plot
plt.scatter(data[:, 0], data[:, 1], c=labels, cmap='viridis')
plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.legend()
plt.title('K-means Clustering')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.show()