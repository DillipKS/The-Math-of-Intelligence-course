# https://github.com/llSourcell/Intro_to_the_Math_of_intelligence
# Create a linear regression model and train using Gradient Descent from scratch.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

datapath = "data.csv"
alpha = 3e-5         # learning rate
num_iter = 100     # number of iterations for gradient descent

def data_process(path):
    "Import and preprocess data"
    df = pd.read_csv(path)
    df = df.reindex(np.random.permutation(df.index))
    return df
'''
def randomize(df):
    "Randomly shuffle the items of data"
    r_data = df.reindex(np.random.permutation(df.index))
    return r_data
'''
def gradient_descent(x, y, N, m, b, alpha, num_iter):
    "Optimize the model using Gradient Descent algorithm"
    loss_info = []
    inter_loss = 0
    for i in range(num_iter):
        b_loss = 0
        m_loss = 0
        for j in range(N):
            y_pred = m * x[j] + b
            b_loss += (y[j] - y_pred)
            m_loss += (y[j] - y_pred) * x[j]

        m = m - alpha * (-2/float(N) * m_loss)
        b = b - alpha * (-2/float(N) * b_loss)

        for k in range(N):
            y_pred = m * x[k] + b
            inter_loss += (y_pred - y[k])**2
        inter_loss = inter_loss / float(N)
        loss_info.append([i, inter_loss])

    return m, b, loss_info

def linear_regressor(r_data):
    "Create a linear regression model"
    
    x = r_data["Distance cycled"]
    y = r_data["Calories burnt"]
    N = len(r_data)
    m_init = 0    #1.4
    b_init = 0
    loss = 0

    m, b, loss_info = gradient_descent(x, y, N, m_init, b_init, alpha, num_iter)
    print "m, b, alpha = %0.7s, %0.7s, %s" % (m, b, alpha)

    for i in range(N):
        #y_pred = m_init * x[i] + b_init
        y_pred = m * x[i] + b
        loss += (y_pred - y[i])**2

    final_loss = loss / float(N)
    return final_loss, loss_info, m, b

def plot_loss(loss_info):
    iter_no = []
    loss_value = []
    for item in loss_info:
        iter_no.append(item[0])
        loss_value.append(item[1])
    
    plt.plot(iter_no, loss_value)
    plt.show()


def main():
    data = data_process(datapath)
    avg_loss, loss_info, slope, intercept = linear_regressor(data)
    #plot_loss(loss_info)

    x = data["Distance cycled"]
    y = data["Calories burnt"]
    y_pred = slope * x + intercept
    
    x_new = [72, 73.5, 74.9, 76.5, 78.5, 80]
    y_new = [(slope * i + intercept) for i in x_new]
    
    plt.plot(x, y, 'ro', x, y_pred, x_new, y_new, 'bs')
    plt.show()
    
    print "Final avg loss = %s" % avg_loss

if __name__ == '__main__':
    main()

'''
Results-
m, b, alpha = 1.478566841371454, 0.047075978779471006, 0.0003
Final loss = 112.639852759
'''