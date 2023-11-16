import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('housing.csv')

# Preprocessing - scale the median_income feature
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data['median_income_scaled'] = scaler.fit_transform(data[['median_income']])

# Select the feature for clustering
X = data[['median_income_scaled']]

# Determine the optimal number of clusters using the Elbow method
inertia = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Plot the Elbow method results to choose K
plt.plot(range(1, 11), inertia, marker='o')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()

# Choose an appropriate value of K based on the plot

# Apply K-means clustering with the chosen K
k = 3  # Example: Choose an appropriate K based on the plot
kmeans = KMeans(n_clusters=k, random_state=42)
data['income_cluster'] = kmeans.fit_predict(X)

# Analyze and visualize the clusters
cluster_centers = scaler.inverse_transform(kmeans.cluster_centers_)
data['median_income_cluster'] = cluster_centers[data['income_cluster']]

# Print cluster centers (median income values)
print(cluster_centers)

# Visualize the clusters
plt.scatter(data['longitude'], data['latitude'], c=data['income_cluster'], cmap='rainbow')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('K-means Clusters of Median Income')
plt.show()

# Interpret the results and label the clusters as needed