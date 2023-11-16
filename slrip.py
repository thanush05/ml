import matplotlib.pyplot as plt                                                                                                                                                                                    
from sklearn.linear_model import LinearRegression                                                                                                                                                                  
import numpy as np                                                                                                                                                                                                 

# Data                                                                                                                                                                                                             
Id = [0.734, 0.886, 1.04, 1.19, 1.35, 1.50, 1.66, 1.81, 1.97, 2.12]                                                                                                                                                
Vgs = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]                                                                                                                                                           

# Convert lists to numpy arrays and reshape them                                                                                                                                                                   
Id = np.array(Id).reshape(-1, 1)                                                                                                                                                                                   
Vgs = np.array(Vgs).reshape(-1, 1)                                                                                                                                                                                 

# Create a scatter plot                                                                                                                                                                                            
plt.scatter(Vgs, Id)                                                                                                                                                                                               

# Add labels and title                                                                                                                                                                                             
plt.xlabel("Gate-to-Source Voltage (Vgs)")                                                                                                                                                                         
plt.ylabel("Drain Current (Id, mA)")                                                                                                                                                                               
plt.title("Scatter Plot of Drain Current vs. Gate-to-Source Voltage")                                                                                                                                              

# Fit the linear regression model                                                                                                                                                                                  
regression = LinearRegression()                                                                                                                                                                                    
regression.fit(Vgs, Id)                                                                                                                                                                                            

# Predict values using the regression model                                                                                                                                                                        
predicted_Id = regression.predict(Vgs)                                                                                                                                                                             
# Plot the regression line                                                                                                                                                                                         
plt.plot(Vgs, predicted_Id, color='red', label='Regression Line')                                                                                                                                                  

# Show the plot                                                                                                                                                                                                    
plt.legend()                                                                                                                                                                                                       
plt.show()                                                                                                                                                                                                         

# Get the parameters of the model                                                                                                                                                                                  
intercept = regression.intercept_[0]                                                                                                                                                                               
slope = regression.coef_[0][0]                                                                                                                                                                                     


print(f"Intercept (β₀): {intercept}")                                                                                                                                                                              
print(f"Slope (β₁): {slope}")                                                                                                                                                                                      

# Report the equation of the least-squares regression line                                                                                                                                                         
print(f"Equation of the least-squares regression line: Id = {intercept:.4f} + {slope:.4f} * Vgs")     