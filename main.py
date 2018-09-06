# python: 3.5.2
# encoding: utf-8
# numpy: 1.14.1

import numpy as np
import matplotlib.pyplot as plt


def load_data(filename):
    xys = []
    with open(filename, 'r') as f:
        for line in f:
            xys.append(map(float, line.strip().split()))
        xs, ys = zip(*xys)
        return np.asarray(xs), np.asarray(ys)


def evaluate(ys, ys_pred):
    std = np.sqrt(np.mean(np.abs(ys - ys_pred) ** 2))
    return std


def main(x_train, y_train):

    # bias phi0
    phi0 = np.expand_dims(np.ones_like(x_train), axis=1)

    # polynomial as basic function
    polydegree = 4

    polyphx = x_train[:]
    polyphx = np.expand_dims(polyphx, axis=1)
    for i in range(2, polydegree+1):
        xi = np.array(np.power(polyphx[:,0], i))
        polyphx = np.concatenate((polyphx, np.expand_dims(xi, axis=1)), axis=1)
    
    # gaussians as basic function
    gaussdegree = 90
    gaussmeanj = 1
    s2 = 1

    gaussphx = np.array(np.exp(- np.power(x_train - gaussmeanj, 2) / 2.0*s2))
    gaussphx = np.expand_dims(gaussphx, axis=1)
    for i in range(2, gaussdegree + 1):
        xi = np.array(np.exp(- np.power(x_train - gaussmeanj * i, 2) / 2.0*s2))
        gaussphx = np.concatenate((gaussphx, np.expand_dims(xi, axis=1)), axis=1)
    
    # sigmoid as basic function
    sigmoiddegree = 80
    sigmoidmeanj = 1
    s = 8

    sigmoidphx = np.array(1.0 / (1 + np.exp((sigmoidmeanj - x_train) / s)))
    sigmoidphx = np.expand_dims(sigmoidphx, axis=1)
    for i in range(2, sigmoiddegree+1):
        xi = np.array(1.0 / (1 + np.exp((sigmoidmeanj*i - x_train) / s)))
        sigmoidphx = np.concatenate((sigmoidphx, np.expand_dims(xi, axis=1)), axis=1)

    # choose one basic function and get w on train dataset
    xList = {'polynomial': polyphx, 'gaussian': gaussphx, 'sigmoid': sigmoidphx}

    phi1 = xList['sigmoid']
    phi = np.concatenate([phi0, phi1], axis=1)
    w = np.dot(np.linalg.pinv(phi), y_train)
    
    # evaluate train 
    _y_train = np.dot(phi, w)
    print ("train evaluate is : ", evaluate(y_train, _y_train))

    def f(x):

        # bias phi0
        phi0 = np.expand_dims(np.ones_like(x), axis=1)

        # polynomial basic function
        polyphx = x[:]
        polyphx = np.expand_dims(polyphx, axis=1)
        for i in range(2, polydegree+1):
            xi = np.array(np.power(polyphx[:,0], i))
            polyphx = np.concatenate((polyphx, np.expand_dims(xi, axis=1)), axis=1)
        
        # guassian basic function
        gaussphx = np.array(np.exp(- np.power(x - gaussmeanj, 2) / 2.0*s2))
        gaussphx = np.expand_dims(gaussphx, axis=1)
        for i in range(2, gaussdegree+1):
            xi = np.array(np.exp(- np.power(x - gaussmeanj * i, 2) / 2.0*s2))
            gaussphx = np.concatenate((gaussphx, np.expand_dims(xi, axis=1)), axis=1)
        
        # sigmoid basic function
        sigmoidphx = np.array(1.0 / (1 + np.exp((sigmoidmeanj - x) / s)))
        sigmoidphx = np.expand_dims(sigmoidphx, axis=1)
        for i in range(2, sigmoiddegree+1):
            xi = np.array(1.0 / (1 + np.exp((sigmoidmeanj*i - x) / s)))
            sigmoidphx = np.concatenate((sigmoidphx, np.expand_dims(xi, axis=1)), axis=1)
        
        # choose one basic function
        xList = {'polynomial': polyphx, 'gaussian': gaussphx, 'sigmoid': sigmoidphx}

        phi1 = xList['sigmoid']
        phi = np.concatenate([phi0, phi1], axis=1)

        # calculate predict y
        y = np.dot(phi, w)
        return y

    return f


if __name__ == '__main__':
    train_file = 'train.txt'
    test_file = 'test.txt'

    x_train, y_train = load_data(train_file)
    x_test, y_test = load_data(test_file)
    print(x_train.shape)
    print(x_test.shape)
    
    f = main(x_train, y_train)

    y_test_pred = f(x_test)


    std = evaluate(y_test, y_test_pred)
    print('{:.1f}'.format(std))

    plt.plot(x_train, y_train, 'ro', markersize=3)
    plt.plot(x_test, y_test, 'k')
    plt.plot(x_test, y_test_pred)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Linear Regression')
    plt.legend(['train', 'test', 'pred'])
    plt.show()
