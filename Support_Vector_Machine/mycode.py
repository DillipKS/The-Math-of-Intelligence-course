# https://github.com/llSourcell/Classifying_Data_Using_a_Support_Vector_Machine/blob/master/support_vector_machine_lesson.ipynb
# Build a SVM model to classify a set of data points.

import numpy as np
import matplotlib.pyplot as plt

def plot_data(X, Y, w):
    # Plot the data on a graph
    for i, sample in enumerate(X):
        if Y[i] == -1:
            plt.scatter(sample[0], sample[1], s=120, marker='_', linewidths=2)
        elif Y[i] == 1:
            plt.scatter(sample[0], sample[1], s=120, marker='+', linewidths=2)
    plt.plot([-3,6],[6,-1])     # guessed hyperplane
    
    plt.scatter(2,2, s=120, marker='*')
    plt.scatter(4,3, s=120, marker='*')

    # Print the hyperplane calculated by svm_sgd()
    x2=[w[0],w[1],-w[1],w[0]]
    x3=[w[0],w[1],w[1],-w[0]]

    x2x3 =np.array([x2,x3])
    X,Y,U,V = zip(*x2x3)
    ax = plt.gca()
    ax.quiver(X,Y,U,V,scale=1, color='blue')
    plt.show()

def data_process():
    # Define our data points
    # Input data of the form [X value, Y value, Bias term]
    # X1 = np.array([
    #     [-2,4,-1],
    #     [4,-2,-1],
    #     [0,7,-1],
    #     [2,6,-1],
    #     [4,2,-1]
    # ])
    X = np.array([
        [-2, 4, -1],
        [4, 1, -1],
        [1, 6, -1],
        [2, 4, -1],
        [6, 2, -1],

    ])

    # Output labels
    # First two are -ve samples and last three are +ve samples
    Y = np.array([-1,-1,1,1,1])
    return X,Y

def loss_fn(X, Y, w):
    loss = 0
    for i, x in enumerate(X):
        if Y[i] * np.dot(x,w) < 1:
            loss += (1 - Y[i] * np.dot(w, x))
        else:
            loss += 0
    return loss

def svm_sgd_model(X, Y):
    epochs = 200000                 # no. of epochs to train
    alpha = 1                   # learning rate
    w = np.zeros(len(X[0]))     # weight initialization
    lambda_val = 1/float(epochs)

    epoch_no = []
    loss_info = []              # track loss value with iterations
    #plot_data(X,Y,w)

    initial_loss = loss_fn(X,Y,w)
    for epoch in range(1, epochs):
        for i, x in enumerate(X):
            if (Y[i] * np.dot(x,w)) < 1:
                w = w - alpha * (2 * lambda_val * w - Y[i] * x)
            elif (Y[i] * np.dot(x,w)) >= 1:
                w = w - alpha * (2 * lambda_val * w)
        loss = loss_fn(X,Y,w)
        epoch_no.append(epoch)
        loss_info.append(loss)
    
    plt.plot(epoch_no, loss_info)
    plt.show()

    final_loss = loss_fn(X,Y,w)

    return final_loss, w


def main():
    data, label = data_process()
    loss, weights = svm_sgd_model(data, label)
    print "Loss = %s" % loss
    print "Weights = %s" % weights
    plot_data(data, label, weights)

if __name__ == '__main__':
    main()


'''
My result for data-
X1 : Weights = [2.52142564 2.52203993 6.30117324]
X  : Weights = [1.49469084  2.98808089 10.46163978]
Alpha = 1, Epochs = 200000


Original result-
The weight vector of the SVM including the bias term after 100000 epochs is $(1.56, 3.17, 11.12)$.
We can extract the following prediction function now:
    f(x) = sgn(<x,(1.56,3.17)> - 11.12)

The weight vector is (1.56,3.17) and the bias term is the third entry 11.12.
'''