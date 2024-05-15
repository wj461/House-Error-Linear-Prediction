import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_o = pd.read_csv('./data/housing_data.csv')
test_o = pd.read_csv('./data/test.csv')

data_norm_o= pd.read_csv('./data/housing_data_normalized.csv')
test_norm_o = pd.read_csv('./data/test_normalized.csv')

def function_3(p_m, x):
        a = np.dot(p_m, x)
        return a

def three_d_linear_regression(need, data, test, norm = False):
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

    if not norm:
        print("not norm")
        x_p,y_p,_ = data_o[need].max().astype(int)
        for i in range(0, x_p+1):
            for j in range(0, y_p+1):
                p_m = np.vstack([p_m, [i, j]])
    else:
        print("norm")
        x_p,y_p,_ = data_norm_o[need].max().astype(int)
        for i in np.arange(0, x_p+1, 0.01):
            for j in np.arange(0, y_p+1, 0.01):
                p_m = np.vstack([p_m, [i, j]])

    p_z = function_3(p_m, x)
    ax1.plot(p_m[:,0], p_m[:,1], p_z, c='r', alpha=0.5)

    ans = np.dot(test, x)
    print("ans : ", ans)

    ax1.scatter(test[:, 0], test[:, 1], ans, c='r')
    ax1.text(test[:, 0][0], test[:, 1][0], ans[0], f'{ans[0]:.1f}', None)
    ax1.scatter(test[:, 0], test[:, 1], test_o[need[-1]], c='purple')
    ax1.text(test[:, 0][0], test[:, 1][0], test_o[need[-1]][0], f'{test_o[need[-1]][0]:.1f}', None)

    return ans

def all_D():
    data = data_o.values
    temp = test_o.drop('MEDV', axis = 1)
    test = temp.values

    A = data[:, :-1]
    b = data[:, -1].T

    Q, R = np.linalg.qr(A)

    x = np.linalg.solve(R, np.dot(Q.T, b))
    print(np.dot(test, x))
    return (np.dot(test, x))
    

def cal_loss(ans, test):
    loss = 0
    for i in range(len(ans)):
        loss += (ans[i] - test[i])**2
    return loss

need = ['TAX', 'AGE', 'MEDV']
data = data_o[need]
test = test_o[need[: -1]]
ans = three_d_linear_regression(need,data, test)
print("Least squares", cal_loss(ans, test_o[need[-1]]))


data = data_norm_o[need]
test = test_norm_o[need[: -1]]
ans = three_d_linear_regression(need,data, test, True)
print("Least squares", cal_loss(ans, test_o[need[-1]]))

# plt.show() 
print("original data")
ans = all_D()
print("Least squares", cal_loss(ans, test_o[need[-1]]))
