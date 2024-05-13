import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.decomposition import TruncatedSVD
from sklearn import preprocessing

data_o = pd.read_csv('./data/housing_data.csv')
test_o = pd.read_csv('./data/test.csv')
# data_o = pd.read_csv('./data/housing_data_normalized.csv')
# test_o = pd.read_csv('./data/test_normalized.csv')


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
    # ax1.scatter(data[need[0]], data[need[1]], data[need[2]])
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
    print("x, y : ", x_p, y_p)
    # for i in np.arange(0, x_p+1, 0.01):
    #     for j in np.arange(0, y_p+1, 0.01):
    for i in np.arange(0, x_p+1,0.1):
        for j in np.arange(0, y_p+1,0.1):
            p_m = np.vstack([p_m, [i, j]])

    p_z = function_3(p_m, x)
    ax1.plot(p_m[:,0], p_m[:,1], p_z, c='r', alpha=0.5)

    ans = np.dot(test, x)
    print("ans : ", ans)

    ax1.scatter(test[:, 0], test[:, 1], ans, c='r')
    ax1.scatter(test[:, 0], test[:, 1], test_o['MEDV'], c='purple')

def svd(need):
    np.set_printoptions(suppress=True)
    k = 9

    data = data_o.values
    # test = test_o[need[: -1]].values
    test = test_o.drop(columns='MEDV').values

    A = data[:, :-1]
    b = data[:, -1].T

    U, S, VT = np.linalg.svd(A)

    U_k = U[:, :k]
    S_k = np.diag(S[:k])
    VT_k = VT[:k, :]
    print(U_k.shape, S_k.shape, VT_k.shape)

    # Construct the approximate matrix
    A_k = np.dot(U_k, np.dot(S_k, VT_k))

    Q, R = np.linalg.qr(A_k)
    # print("SVD Q,R", Q, R)

    x = np.linalg.solve(R, np.dot(Q.T, b))
    print(np.dot(test, x))

def svd_2():
    data = data_o.values
    test = test_o.drop(columns=['MEDV']).values

    A = data[:, :-1]
    b = data[:, -1].T

    svd = TruncatedSVD(n_components=2)

    X_svd = svd.fit_transform(A)
    # print(X_svd)
    plt.figure(figsize=(8, 6))
    ax1 = plt.axes(projection='3d')
    ax1.scatter(X_svd[:, 0], X_svd[:, 1], b, cmap='viridis')
    # ax1.xlabel('1')
    # ax1.ylabel('2')

    plt.show()

def all_D():
    data = data_o.values
    test = test_o.drop(columns=['MEDV']).values

    A = data[:, :-1]
    b = data[:, -1].T

    Q, R = np.linalg.qr(A)
    # print("all Q,R", Q, R)

    x = np.linalg.solve(R, np.dot(Q.T, b))
    print(np.dot(test, x))

# need = ['AGE', 'MEDV']
# two_d_linear_regression(need)

need = ['RM', 'LSTAT', 'MEDV']
three_d_linear_regression(need)

plt.show() 
all_D()
# svd(need)
# svd_2()

correlation = data_o.corr()['MEDV'].abs().sort_values(ascending=False)
print(correlation)
