import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data_o = pd.read_csv('./data/chat_gpt.csv')
test_o = pd.read_csv('./data/chat_gpt_test.csv')

data_norm_o = pd.read_csv('./data/chat_gpt_normalized.csv')
test_norm_o = pd.read_csv('./data/chat_gpt_test_normalized.csv')

def function_3(p_m, x):
        a = np.dot(p_m, x)
        return a

def three_d_linear_regression(need, data, test, norm = False):

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

    if not norm:
        print("not norm")
        x_p,y_p,_ = data_o[need].max().astype(int)
        for i in range(0, x_p+1):
            for j in range(0, y_p+1):
                p_m = np.vstack([p_m, [i, j]])
    else:
        print("norm")
        x_p,y_p,_ = data_norm_o[need].max().astype(int)
        for i in np.arange(0, x_p+1, 0.001):
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

need = ['House Area (Square Meters)',
'Distance to City Center (Kilometers)',
'Sale Price (Ten Thousand Yuan)']

data = data_o[need]
test = test_o[need[: -1]]
three_d_linear_regression(need,data, test)

# data = data_norm_o[need]
# test = test_norm_o[need[: -1]]
# three_d_linear_regression(need,data, test, True)

plt.show() 

