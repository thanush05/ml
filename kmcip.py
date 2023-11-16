import numpy as np

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((np.array(point1) - np.array(point2))**2))

def k_means_clustering(data, centroids, num_iterations=100):
    for iteration in range(num_iterations):
        # Assign each data point to the nearest centroid
        assignments = {}
        for data_point in data:
            min_distance = float('inf')
            assigned_centroid = None
            for centroid_name, centroid_location in centroids.items():
                distance = euclidean_distance(data_point, centroid_location)
                if distance < min_distance:
                    min_distance = distance
                    assigned_centroid = centroid_name
            assignments.setdefault(assigned_centroid, []).append(data_point)

        # Update centroid locations
        for centroid_name, assigned_data in assignments.items():
            new_centroid_location = np.mean(assigned_data, axis=0)
            centroids[centroid_name] = new_centroid_location.tolist()

    return centroids, assignments

# Example data and centroids
data = [[0.34, -0.2, 1.13, 4.3], [5.1, -12.6, -7.0, 1.9], [-15.7, 0.06, -7.1, 11.2]]
centroids = {"centroid1": [1.1, 0.2, -3.1, -0.4], "centroid2": [9.3, 6.1, -4.7, 0.18]}

# Run K-means clustering
final_centroids, assignments = k_means_clustering(data, centroids)

# Print results
print("Final Centroids:")
for centroid_name, centroid_location in final_centroids.items():
    print(f"{centroid_name}: {centroid_location}")

print("\nAssignments:")
for centroid_name, assigned_data in assignments.items():
    print(f"{centroid_name}: {len(assigned_data)} data points")