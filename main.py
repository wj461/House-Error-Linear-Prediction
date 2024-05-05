import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

data = pd.read_csv('./data/housing_data.csv') 
test = pd.read_csv('./data/test.csv')

def function_2(p_x, x):
    return p_x * x[0]

def two_d_linear_regression():
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

    p_x = np.arange(0, 80, 0.1)
    p_y = function_2(p_x, x)
    print(x)
    plt.plot(p_x, p_y)

    ans = np.dot(test, x)
    print(ans)

    plt.plot(test[:, 0], ans, 'ro')

    plt.show() 

def function_3(p_m, x):
        # return p_x * x[0] + p_y * x[1]
        a = np.dot(p_m, x)
        print (f'f3:{p_m}\n{a}')
        return a

def three_d_linear_regression():
    data = pd.read_csv('./data/3d.csv')
    test = pd.read_csv('./data/test_3d.csv')

    fig = plt.figure()
    ax1 = plt.axes(projection='3d')
    ax1.scatter(data['CRIM'], data['AGE'], data['MEDV'])
    ax1.set_xlabel('CRIM')
    ax1.set_ylabel('AGE')
    ax1.set_zlabel('MEDV')

    data = data.values
    test = test.values

    A = data[:, :-1]
    b = data[:, -1].T

    Q, R = np.linalg.qr(A)

    x = np.linalg.solve(R, np.dot(Q.T, b))

    p_m = np.array([[0, 0]])
    for i in range(0, 100):
        for j in range(0, 100):
            p_m = np.vstack([p_m, [i, j]])

    p_z = function_3(p_m, x)
    ax1.plot(p_m[:,0], p_m[:,1], p_z)

    ans = np.dot(test, x)

    # print(ans)
    ax1.scatter(test[:, 0], test[:, 1], ans, c='r')
    # ax1.plot_surface(p_mx, p_my, ans, c='r')

    plt.show() 


three_d_linear_regression()
# two_d_linear_regression()

# need = ['CRIM', 'ZN', 'INDUS', 'CHAS']
# split_data = data[need]
# print(split_data)


# test data
# A = np.array([
#     [10, 5, 3], 
#     [20, 7, 4],
#     [15, 6, 3.5]])

# b = np.array( [100 , 150, 120]).T
# test = np.array([10, 5, 3])