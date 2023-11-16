import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

# Load the dataset
data = pd.read_csv('Data.csv')

# Remove '$' and commas and convert 'Income' to a float
data['Income'] = data['Income'].str.replace('$', '').str.replace(',', '').astype(float)

# Extract the 'Age' and 'Income' attributes
X = data[['Age', 'Income']].values
y = data['Risk'].values  # Assuming 'Risk' is the correct column name

# Create a Min-Max scaler
scaler = MinMaxScaler()

# Fit and transform the data using Min-Max scaling
X_scaled = scaler.fit_transform(X)

# Calculate the distance for the new record (#10)
new_record = np.array([[66, 36120.34]])  # Replace with the values of the new record
new_record_scaled = scaler.transform(new_record)

# Initialize the k-NN classifier with k=9
k = 9
knn = KNeighborsClassifier(n_neighbors=k)

# Fit the k-NN classifier to the scaled data
knn.fit(X_scaled, y)

# Print the Min-Max standardized values for Age and Income
print("Min-Max Standardized Values for Age and Income:")
print(X_scaled)

# Calculate the distances from the new record to the nine stored records
distances = knn.kneighbors(new_record_scaled, return_distance=True)

# Print the distances
print("\nDistances from New Record (#10) to Nine Records:")
print(distances)

# Get the indices of the nine nearest neighbors
indices_of_neighbors = distances[1][0]

# Print the indices of the nine nearest neighbors
print("\nIndices of the Nine Nearest Neighbors:")
print(indices_of_neighbors)

# Get the risk factors of the nine nearest neighbors
neighbors_risk_factors = y[indices_of_neighbors]

# Print the risk factors of the nine nearest neighbors
print("\nRisk Factors of the Nine Nearest Neighbors:")
print(neighbors_risk_factors)

# Predict the risk factor for the new record using unweighted voting
predicted_risk_factor = knn.predict(new_record_scaled)

# Print the predicted risk factor for the new record
print("\nPredicted Risk Factor for new record (#10):", predicted_risk_factor[0])