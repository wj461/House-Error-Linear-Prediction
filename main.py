import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('./data/2d.csv')
test = pd.read_csv('./data/test_2d.csv')

df = data[['CRIM','MEDV']]
df.plot(kind = 'scatter', x = 'CRIM', y = 'MEDV')


data = data.values
test = test.values

A = data[:, :-1]
b = data[:, -1].T

Q, R = np.linalg.qr(A)

x = np.linalg.solve(R, np.dot(Q.T, b))

def function(p_x):
    return p_x * x[0]

p_x = np.arange(0, 80, 0.1)
p_y = function(p_x)
print(x)
plt.plot(p_x, p_y)

ans = np.dot(test, x)
print(ans)

plt.plot(test[:, 0], ans, 'ro')

plt.show() 

# plt
# plt.plot([1, 2, 3, 4])
# plt.ylabel('some numbers')
# plt.show()



# test data
# A = np.array([
#     [10, 5, 3], 
#     [20, 7, 4],
#     [15, 6, 3.5]])

# b = np.array( [100 , 150, 120]).T
# test = np.array([10, 5, 3])