import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Load CSV and columns
df = pd.read_csv("CHD1.csv")

# Extract the independent variable (age) and dependent variable (coronary_heart_disease)
X = df['age'].values.reshape(-1, 1)
Y = df['coronary_heart_disease'].values.reshape(-1, 1)

# Create a linear regression model
regression = LinearRegression()

# Fit the model to the data
regression.fit(X, Y)

# Predict the values using the model
predicted_Y = regression.predict(X)

# Calculate the coefficient of determination (R²)
r2 = r2_score(Y, predicted_Y)

# Plot the original data and the regression line
plt.scatter(X, Y, color='black', label='Original Data')
plt.plot(X, predicted_Y, color='red', label='Regression Line')
plt.title('Linear Regression for CHD Data')
plt.xlabel('Age')
plt.ylabel('Coronary Heart Disease (CHD)')
plt.legend()
plt.show()

# Get the model parameters
intercept = regression.intercept_[0]
slope = regression.coef_[0][0]

# Report the parameters of the model (w0 and w1)
print(f"Intercept (w0): {intercept:.4f}")
print(f"Slope (w1): {slope:.4f}")

# Report the coefficient of determination (R²)
print(f"Coefficient of Determination (R²): {r2:.4f}")

# Predict the probability of someone 41 years old suffering from CHD
age_41 = 41
predicted_probability_41 = regression.predict([[age_41]])[0][0]
print(f"Probability of someone aged 41 suffering from CHD: {predicted_probability_41:.4f}")