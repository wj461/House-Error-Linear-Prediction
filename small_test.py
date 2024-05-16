import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler

data_o = pd.read_csv('./data/gpt.csv')
test_o = pd.read_csv('./data/gpt_test.csv')
test_norm_o = pd.read_csv('./data/gpt_test_norm.csv')
target = 'Sale Price (Ten Thousand Yuan)'

def function_3(p_m, x):
        a = np.dot(p_m, x)
        return a

def three_d_linear_regression(need, norm = False, draw = True):
    if norm:
        scaler = StandardScaler()
        data = scaler.fit_transform(data_o[need[:-1]])
        draw_d = np.column_stack((data, data_o[target]))
        draw_d = pd.DataFrame(draw_d, columns=need)
        test = test_norm_o[need[:-1]]
        test = test.values

        A = np.column_stack((np.ones(data.shape[0]), data))
        b = data_o[target]
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

def all_D():
    data = data_o.values
    temp = test_o.drop(target , axis = 1)
    test = temp.values

    A = data[:, :-1]
    A = np.column_stack((np.ones(A.shape[0]), A))
    b = data[:, -1]

    Q, R = np.linalg.qr(A)

    x = np.linalg.solve(R, np.dot(Q.T, b))
    ans = np.dot(A, x)

    mse_np = np.mean((b - ans)**2)
    print("MSE using numpy's solution: ", mse_np)


need = ['House Area (Square Meters)',
'Distance to City Center (Kilometers)',
'Sale Price (Ten Thousand Yuan)']

three_d_linear_regression(need)
three_d_linear_regression(need, True)

plt.show() 
print("original data")
ans = all_D()