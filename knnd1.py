import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv("data4_19.csv")


# Split the data into features and labels
X = data.iloc[:, 0:4]
y = data.iloc[:, 4]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to find the best K-value and accuracy for a given distance metric
def find_best_k_value(distance_metric):
    best_k = None
    best_accuracy = 0

    for k in range(1, 21):  # You can adjust the range as needed
        knn = KNeighborsClassifier(n_neighbors=k, metric=distance_metric)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_k = k

    return best_k, best_accuracy

# Define the distance metrics to be considered
distance_metrics = ['euclidean', 'manhattan', 'chebyshev']

# Find the best K-value and accuracy for each distance metric
best_k_values = {}
for metric in distance_metrics:
    best_k, best_accuracy = find_best_k_value(metric)
    best_k_values[metric] = (best_k, best_accuracy)

# Tabulate the results
print("Distance Metric | Best K-Value | Accuracy")
for metric in distance_metrics:
    best_k, best_accuracy = best_k_values[metric]
    print(f"{metric:15} | {best_k:12} | {best_accuracy:.4f}")

# Choose the distance metric with the highest accuracy and use that for prediction
best_metric = max(best_k_values, key=lambda metric: best_k_values[metric][1])
best_k, _ = best_k_values[best_metric]

# Predict the species of the new flower
new_flower = [[5.2, 3.1, 1.4, 0.2]]
knn = KNeighborsClassifier(n_neighbors=best_k, metric=best_metric)
knn.fit(X, y)
predicted_species = knn.predict(new_flower)

print(f"Predicted Species: {predicted_species[0]}")