import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# Create a DataFrame for the data
data = pd.DataFrame({
    'Diameter': [1, 2, 3, 1, 2, 3, 1, 2, 3],
    'Slope': [0.001, 0.001, 0.001, 0.01, 0.01, 0.01, 0.05, 0.05, 0.05],
    'Flow': [1.4, 8.3, 24.2, 4.7, 28.9, 84.0, 11.1, 69.0, 200.0]
})

# Create new columns for the logarithmic transformations
data['log_Diameter'] = np.log(data['Diameter'])
data['log_Slope'] = np.log(data['Slope'])
data['log_Flow'] = np.log(data['Flow'])

# Perform multiple linear regression
X = data[['log_Diameter', 'log_Slope']]
y = data['log_Flow']
model = LinearRegression().fit(X, y)

# Extract the coefficients (a0, a1, a2)
a0 = np.exp(model.intercept_)
a1 = model.coef_[0]
a2 = model.coef_[1]

# Print the coefficients
print(f'a0 (constant): {a0:.4f}')
print(f'a1 (coefficient for log(Diameter)): {a1:.4f}')
print(f'a2 (coefficient for log(Slope)): {a2:.4f}')

# Predict the flow for a pipe with Diameter = 2.5 ft and Slope = 0.025 ft/ft
Diameter_pred = 2.5
Slope_pred = 0.025
log_Flow_pred = model.predict(np.array([[np.log(Diameter_pred), np.log(Slope_pred)]]))[0]
Flow_pred = np.exp(log_Flow_pred)

print(f'Predicted Flow for Diameter=2.5 ft and Slope=0.025 ft/ft: {Flow_pred:.2f} ft^3/s')
