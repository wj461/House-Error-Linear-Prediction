import numpy as np

A = np.array([
    [10, 5, 3], 
    [20, 7, 4],
    [15, 6, 3.5]])

b = np.array( [100 , 150, 120]).T

Q, R = np.linalg.qr(A)

x = np.linalg.solve(R, np.dot(Q.T, b))

print(x)

test = np.array([10, 5, 3])

print(np.dot(test, x))