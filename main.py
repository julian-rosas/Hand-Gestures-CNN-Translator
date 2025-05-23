import numpy as np
import pandas as pd
import numba as nb

from sklearn.utils import shuffle

import matplotlib.pyplot as plt
from IPython.display import clear_output
from CNN import *


def processInput():
    df_train = pd.read_csv('./input/train/sign_mnist_train.csv')
    df_test = pd.read_csv('./input/test/sign_mnist_test.csv')
    # only hand gestures from 0 to 10, X represents all the pixels (from  col 1 to 1784)
    X_train, y_train = df_train[df_train['label'] < 10].values[:,1:], df_train[df_train['label'] < 10].values[:,0]
    # Normalize
    X_test, y_test = df_test[df_test['label'] < 10].values[:,1:], df_test[df_test['label'] < 10].values[:,0]
    X_train, X_test = X_train / 255., X_test / 255
    # Format (#, 1, 28, 28)
    X_train, X_test = X_train.reshape(X_train.shape[0], 1, 28, 28), X_test.reshape(X_test.shape[0], 1, 28, 28)
    return X_train, y_train, X_test, y_test


def train(X_train,y_train,X_test,y_test):
    

    NUM_OUTPUT = 10
    LEARNING_RATE = 0.0001
    IMG_WIDTH = 28
    IMG_DEPTH = 1
    FILTER_SIZE=5
    NUM_FILT1 = 8
    NUM_FILT2 = 8
    BATCH_SIZE = 8
    NUM_EPOCHS = 3000
    MU = 0.95
    NUM_IMAGES = int(len(X_train) / BATCH_SIZE)

    np.random.seed(3424242)

    scale = 1.0
    stddev = scale * np.sqrt(1./FILTER_SIZE*FILTER_SIZE*IMG_DEPTH)

    filt1 =np.random.normal(loc = 0, scale=stddev, size=((NUM_FILT1,IMG_DEPTH,FILTER_SIZE,FILTER_SIZE)))
    bias1 = np.zeros((NUM_FILT1,))
    filt2 =np.random.normal(loc = 0, scale=stddev, size=((NUM_FILT2,NUM_FILT1,FILTER_SIZE,FILTER_SIZE)))
    bias2 = np.zeros((NUM_FILT2,))

    w1 = IMG_WIDTH-FILTER_SIZE+1
    w2 = w1-FILTER_SIZE+1

    theta3 = np.random.rand(NUM_OUTPUT, (w2//2)*(w2//2)*NUM_FILT2) * 0.01
    bias3 = np.zeros((NUM_OUTPUT,1))

    theta = filt1, filt2, bias1, bias2, theta3, bias3
    (l,w,w) = X_train[0].shape
    (l1,f,f) = filt2[0].shape
    l2 = len(filt2)
    w1 = w-f+1
    w2 = w1-f+1
    sizes = l, w, f, l1, l2, w1, w2
    accuracies = []
    costs = []
    
    for epoch in range(NUM_EPOCHS):

        filt1, filt2, bias1, bias2, theta3, bias3 = theta
        dfilt1, dfilt2, dtheta3, dbias1, dbias2, dbias3, v1, v2, v3, bv1, bv2, bv3 = init_grads_velocity(filt1, filt2, bias1, bias2, theta3, bias3)
        
        X_train, y_train = shuffle(X_train, y_train, random_state=0)
        minibatches = batches(X_train, y_train, BATCH_SIZE)
        
        idx = np.random.randint(0, len(minibatches))
        X_batch, y_batch = minibatches[idx]

        acc = 0
        cost_epoch = 0

        for i in range(0, BATCH_SIZE):

            label = np.zeros((theta3.shape[0],1))
            label[int(y_batch[i]),0] = 1

            delta = forward(X_batch[i], theta, sizes)

            cost, probs = softmax_cost(delta[-1], label)
            if np.argmax(probs) == np.argmax(label):
                acc += 1
            cost_epoch += cost
                
            nanarg = get_nanargmax(delta[1], sizes)
            dfilt1, dfilt2, dbias1, dbias2, dtheta3, dbias3 = backward(X_batch[i], label, theta, delta, probs, nanarg, sizes)

            # dfilt2 += dfilt2
            dbias2 += dbias2

            dfilt1 += dfilt1
            dbias1 += dbias1

            dtheta3 += dtheta3
            dbias3 += dbias3
            
        accuracies.append(acc)
        costs.append(cost_epoch/BATCH_SIZE)

        v1 = MU * v1 - LEARNING_RATE * dfilt1 / BATCH_SIZE
        filt1 += v1
        bv1 = MU * bv1 - LEARNING_RATE * dbias1 / BATCH_SIZE
        bias1 += bv1

        v2 = MU * v2 - LEARNING_RATE * dfilt2 / BATCH_SIZE
        filt2 += v2
        bv2 = MU * bv2 - LEARNING_RATE * dbias2 / BATCH_SIZE
        bias2 += bv2

        v3 = MU * v3 - LEARNING_RATE * dtheta3 / BATCH_SIZE
        theta3 += v3
        bv3 = MU * bv3 - LEARNING_RATE * dbias3 / BATCH_SIZE
        bias3 += bv3

        theta = filt1, filt2, bias1, bias2, theta3, bias3

        if(epoch % 150 == 0):
            print("Epoch:{0:3d}, Accuracy:{1:0.3f}, Cost:{2:3.3f}".format(epoch, acc/BATCH_SIZE, costs[-1][0]))
    return costs, accuracies, theta, sizes, label 


def analyseResults(costs, accuracies):
    costs = np.array(costs)
    accuracies = np.array(accuracies)

    plt.figure(figsize=(10,10))
    plt.plot(costs, "-b", label="Cost")
    plt.plot(accuracies, "-r", label="Accuracy")
    plt.legend(loc="upper right")
    plt.show()



# Test accuracy:88.74%
def accuracy(X_test, y_test, theta, sizes, label):
    acc = 0
    for i in range(len(X_test)):
        delta = forward(X_test[i], theta, sizes)
        cost, probs = softmax_cost(delta[-1], label)
        
        if np.argmax(probs) == y_test[i]:
            acc += 1
    print("Test accuracy:{0:2.2f}%".format(acc/len(X_test) * 100))



def main():
    X_train, y_train, X_test, y_test = processInput()
    costs, accuracies, theta, sizes, label = train(X_train,y_train,X_test,y_test)
    analyseResults(costs,accuracies)
    accuracy(X_test,y_test,theta,sizes,label)


if __name__ == "__main__":
    main()