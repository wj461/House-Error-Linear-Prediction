import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# data_o = pd.read_csv('./data/housing_data.csv')
# test_o = pd.read_csv('./data/test.csv')

data_o = pd.read_csv('./data/housing_data_normalized.csv')
test_o = pd.read_csv('./data/test_normalized.csv')

def function_2(p_x, x):
    return p_x * x[0]

def two_d_linear_regression(need):
    data = data_o[need]
    # test = pd.read_csv('./data/test_3d.csv')
    test = test_o[need[: -1]]

    df = data[need]
    df.plot(kind = 'scatter', x = need[0], y = need[1])

    data = data.values
    test = test.values

    A = data[:, :-1]
    b = data[:, -1].T

    Q, R = np.linalg.qr(A)

    x = np.linalg.solve(R, np.dot(Q.T, b))

    p_x = np.arange(0, 80, 0.1)
    p_y = function_2(p_x, x)
    print(x)
    plt.plot(p_x, p_y)

    ans = np.dot(test, x)
    print(ans)

    plt.plot(test[:, 0], ans, 'ro')


def function_3(p_m, x):
        a = np.dot(p_m, x)
        return a

def three_d_linear_regression(need):
    data = data_o[need]
    # test = pd.read_csv('./data/test_3d.csv')
    test = test_o[need[: -1]]

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter(data[need[0]], data[need[1]], data[need[2]])
    ax1.set_xlabel(need[0])
    ax1.set_ylabel(need[1])
    ax1.set_zlabel(need[2])

    data = data.values
    test = test.values

    A = data[:, :-1]
    b = data[:, -1].T

    Q, R = np.linalg.qr(A)

    x = np.linalg.solve(R, np.dot(Q.T, b))

    p_m = np.array([[0, 0]])
    x_p,y_p,_ = data_o[need].max().astype(int)
    print(x_p, y_p)
    # for i in range(0, x_p+1):
    #     for j in range(0, y_p+1):
    for i in np.arange(0, x_p+1, 0.01):
        for j in np.arange(0, y_p+1, 0.01):
            p_m = np.vstack([p_m, [i, j]])

    p_z = function_3(p_m, x)
    ax1.plot(p_m[:,0], p_m[:,1], p_z, c='r', alpha=0.5)

    ans = np.dot(test, x)
    print("ans : ", ans)

    # ax1.scatter(test[:, 0], test[:, 1], ans, c='r')
    # ax1.text(test[:, 0][0], test[:, 1][0], ans[0], f'{ans[0]:.1f}', None)
    # ax1.scatter(test[:, 0], test[:, 1], test_o[need[-1]], c='purple')
    # ax1.text(test[:, 0][0], test[:, 1][0], test_o[need[-1]][0], f'{test_o[need[-1]][0]:.1f}', None)


def all_D():
    data = data_o.values
    temp = test_o.drop('MEDV', axis = 1)
    test = temp.values

    A = data[:, :-1]
    b = data[:, -1].T

    Q, R = np.linalg.qr(A)

    x = np.linalg.solve(R, np.dot(Q.T, b))
    print(np.dot(test, x))

# need = ['AGE', 'MEDV']
# two_d_linear_regression(need)

# need = ['ZN', 'RM', 'MEDV']
# three_d_linear_regression(need)

# need = ['TAX', 'RM', 'MEDV']
# three_d_linear_regression(need)

need = ['TAX', 'AGE', 'MEDV']
three_d_linear_regression(need)

plt.show() 
all_D()


# test data
# A = np.array([
#     [10, 5, 3], 
#     [20, 7, 4],
#     [15, 6, 3.5]])

# b = np.array( [100 , 150, 120]).T
# test = np.array([10, 5, 3])