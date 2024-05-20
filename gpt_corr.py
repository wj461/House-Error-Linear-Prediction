import numpy as np
import pandas as pd

x = pd.read_csv('./data/housing_data.csv')

def self_correlation_matrix(x):
    n = len(x)
    corr_matrix = np.zeros((n, n))

    mean_x = np.mean(x)
    std_x = np.std(x)
    print(mean_x)
    print(std_x)

    for i in range(n):
        for j in range(n):
            print(i, j)
            print(std_x[i])
            print(std_x[j])
            corr_matrix[i, j] = np.mean((x - mean_x) * (x - mean_x)) / (std_x[i] * std_x[j])

    # Ensure mean_x and std_x are arrays

    return corr_matrix



# 例子数组
# x = np.array([1, 2, 3, 4, 5])

# 计算自身的相关系数矩阵
self_corr_matrix = self_correlation_matrix(x)

print("Self Correlation Matrix:")
print(self_corr_matrix)
