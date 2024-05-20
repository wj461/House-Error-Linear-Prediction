import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import time

data_o = pd.read_csv('./data/housing_data.csv')
data_norm_o = pd.read_csv('./data/housing_data_normalized.csv')
# test_o = pd.read_csv('./data/test.csv')
# test_norm_o = pd.read_csv('./data/test_norm.csv')
# test_o = pd.read_csv('./data/housing_data.csv')
# test_norm_o = pd.read_csv('./data/housing_data_normalized.csv')


def function_3(p_m, x):
        a = np.dot(p_m, x)
        return a

def three_d_linear_regression(need, norm = False, draw = True):
    current_time = time.time()
    if norm:
        scaler = StandardScaler()
        # data = scaler.fit_transform(data_o[need[:-1]])
        data = data_norm_o[need[:-1]]
        draw_d = np.column_stack((data, data_o['MEDV']))
        draw_d = pd.DataFrame(draw_d, columns=need)
        test = test_norm_o[need[:-1]]
        test = test.values

        A = np.column_stack((np.ones(data.shape[0]), data))
        b = data_o['MEDV']
    else:
        draw_d = data_o[need]
        data = data_o[need]
        data = data.values
        test = test_o[need[:-1]]
        test = test.values

        A = data[:, :-1]
        A = np.column_stack((np.ones(A.shape[0]), A))
        b = data[:, -1]

    if draw:
        fig = plt.figure()
        ax1 = plt.axes(projection='3d')
        # ax1.scatter(draw_d[need[0]], draw_d[need[1]], draw_d[need[2]])
        ax1.set_xlabel(need[0])
        ax1.set_ylabel(need[1])
        ax1.set_zlabel(need[2])


    Q, R = np.linalg.qr(A)

    x = np.linalg.solve(R, np.dot(Q.T, b))


    if not norm and draw:
        print("not norm")
        p_m = np.array([[0, 0]])
        x_p,y_p,_ = data_o[need].max().astype(int)
        for i in np.arange(0, x_p+1, 0.5):
            for j in np.arange(0, y_p+1, 0.5):
                p_m = np.vstack([p_m, [i, j]])
    elif draw:
        print("norm")
        p_m = np.array([[0, 0]])
        x_p,y_p,_ = draw_d[need].max().astype(int)
        for i in np.arange(-2, x_p+1, 0.05):
            for j in np.arange(-2, y_p+1, 0.05):
                p_m = np.vstack([p_m, [i, j]])

    print("x : ", x)
    test = np.column_stack((np.ones(test.shape[0]), test))
    ans = np.dot(test, x)
    # print("ans : ", ans)

    if draw:
        p_z = function_3(p_m, x[1:]) + x[0]
        ax1.plot(p_m[:,0], p_m[:,1], p_z, c='r', alpha=0.5)
        ax1.scatter(test[:, 1], test[:, 2], ans, c='r')
        ax1.text(test[:, 1][0], test[:, 2][0], ans[0], f'{ans[0]:.1f}', None)
        ax1.scatter(test[:, 1], test[:, 2], test_o[need[-1]], c='purple')
        ax1.text(test[:, 1][0], test[:, 2][0], test_o[need[-1]][0], f'{test_o[need[-1]][0]:.1f}', None)

    mse_np = np.mean((test_o[need[-1]] - ans)**2)
    print("MSE using numpy's solution: ", mse_np)
    print("cost time : ", time.time() - current_time)

def all_D():
    current_time = time.time()
    data = data_o.values
    temp = test_o.drop('MEDV', axis = 1)
    test = temp.values

    A = data[:, :-1]
    A = np.column_stack((np.ones(A.shape[0]), A))
    b = data[:, -1]

    Q, R = np.linalg.qr(A)

    x = np.linalg.solve(R, np.dot(Q.T, b))
    ans = np.dot(A, x)

    mse_np = np.mean((b - ans)**2)
    print("MSE using numpy's solution: ", mse_np)
    print("cost time : ", time.time() - current_time)

def cal_correlation():
    # calculate correlation by matrix
    sigma = data_o.cov()
    e = np.eye(sigma.shape[0])
    variance = e * sigma
    v = np.sqrt(variance)
    I = np.linalg.inv(v)
    cov = np.dot(I, np.dot(sigma, I))
    print(cov)

need = ['INDUS', 'TAX', 'MEDV']
print("need : ", need)
three_d_linear_regression(need)
three_d_linear_regression(need, True)
print()

need = ['RM', 'CHAS', 'MEDV']
print("need : ", need)
three_d_linear_regression(need)
three_d_linear_regression(need, True)
print()

plt.show() 
print("original data")
ans = all_D()

cal_correlation()

