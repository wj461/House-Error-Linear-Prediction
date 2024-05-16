from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

data = pd.read_csv('./data/housing_data.csv')
# Selecting the columns for prediction and the target
# features = ['INDUS', 'TAX']
features = ['CHAS', 'ZN']
target = 'MEDV'

# Normalize the data
scaler = StandardScaler()
normalized_features = scaler.fit_transform(data[features])

# Adding a column of ones to the features for the intercept term
normalized_features = np.column_stack((np.ones(normalized_features.shape[0]), normalized_features))
print(normalized_features[:5])
import numpy as np

# Perform QR factorization using numpy for normalized features
Q_np, R_np = np.linalg.qr(normalized_features)

# Compute the coefficients using numpy's solve function
coefficients_np = np.linalg.solve(R_np, np.dot(Q_np.T, data[target]))

# Calculate the predictions using numpy's solution
predictions_np = np.dot(normalized_features, coefficients_np)

# Calculate the MSE using numpy's solution
mse_np = np.mean((data[target] - predictions_np)**2)
print("MSE using numpy's solution: ", mse_np)

correlation = data.corr()['MEDV'].abs().sort_values(ascending=False)
print(correlation)