import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Load the dataset
df = pd.read_csv('your_dataset.csv')

# Assuming 'X' and 'y' are column names in your dataset
X = df[['X']]
y = df['y']

# Create a linear regression model
model = LinearRegression()

# Train the model
model.fit(X, y)

# Make predictions on new data points
X_new = np.array([[min(X.values)], [max(X.values)]])  # Assuming X is a numeric column
y_pred = model.predict(X_new)

# Plot the data and the linear regression line
plt.scatter(X, y, alpha=0.5, label='Data')
plt.plot(X_new, y_pred, 'r-', label='Linear Regression')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
