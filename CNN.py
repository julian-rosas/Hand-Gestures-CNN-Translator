import numpy as np
import pandas as pd
import numba as nb

from sklearn.utils import shuffle


@nb.jit(nopython=True)
def maxpool(X, f, s):
    (l, w, w) = X.shape
    pool = np.zeros((l, (w-f)//s+1,(w-f)//s+1))
    for jj in range(0,l):
        for i in range(0, w, s):
            for j in range(0, w, s):
                pool[jj,i//2,j//2] = np.max(X[jj,i:i+f,j:j+f])
    return pool



@nb.jit(nopython=True)
def relu(x):
    return x * (x > 0)

@nb.jit(nopython=True)
def drelu(x):
    return 1. * (x > 0)



def softmax_cost(out,y):
    eout = np.exp(out, dtype=np.float64)
    probs = eout/sum(eout)

    p = sum(y*probs)
    cost = -np.log(p)
    return cost, probs

@nb.jit(nopython=True)
def forward(image, theta, sizes):
    filt1, filt2, bias1, bias2, theta3, bias3 = theta
    l, w, f, l1, l2, w1, w2 = sizes

    conv1 = np.zeros((l1,w1,w1))
    conv2 = np.zeros((l2,w2,w2))

    for jj in range(0,l1):
        for x in range(0,w1):
            for y in range(0,w1):
                conv1[jj,x,y] = np.sum(image[:,x:x+f,y:y+f]*filt1[jj])+bias1[jj]
    conv1 = relu(conv1)

    for jj in range(0,l2):
        for x in range(0,w2):
            for y in range(0,w2):
                conv2[jj,x,y] = np.sum(conv1[:,x:x+f,y:y+f]*filt2[jj])+bias2[jj]
    conv2 = relu(conv2)

    ## Pooled layer with 2*2 size and stride 2,2
    pooled_layer = maxpool(conv2, 2, 2)	

    fc1 = pooled_layer.reshape(((w2//2)*(w2//2)*l2,1))

    out = theta3.dot(fc1) + bias3

    return conv1, conv2, pooled_layer, fc1, out


# get the highest (index) value of the arrays
def get_nanargmax(conv2, sizes):

    def nanargmax(a):
        idx = np.argmax(a, axis=None)
        multi_idx = np.unravel_index(idx, a.shape)
        if np.isnan(a[multi_idx]):
            nan_count = np.sum(np.isnan(a))
            idx = np.argpartition(a, -nan_count-1, axis=None)[-nan_count-1]
            multi_idx = np.unravel_index(idx, a.shape)
        return multi_idx

    l, w, f, l1, l2, w1, w2 = sizes
    nanarg = np.zeros((l2, w2, w2, 2))

    for jj in range(0, l2):
        for i in range(0, w2, 2):
            for j in range(0, w2, 2):
                (a,b) = nanargmax(conv2[jj,i:i+2,j:j+2])
                nanarg[jj, i, j, 0] = a
                nanarg[jj, i, j, 1] = b

    return nanarg




@nb.jit(nopython=True)
def init_grads(sizes):
    l, w, f, l1, l2, w1, w2 = sizes
    
    dconv2 = np.zeros((l2, w2, w2))
    dconv1 = np.zeros((l1, w1, w1))

    dfilt2 = np.zeros((l2, l1,f,f))
    dbias2 = np.zeros((l2, ))

    dfilt1 = np.zeros((l1,l,f,f))
    dbias1 = np.zeros((l1,))
    
    return dconv1, dconv2, dfilt1, dfilt2, dbias1, dbias2



@nb.jit(nopython=True)
def backward(image, label, theta, delta, probs, nanarg, sizes):
    filt1, filt2, bias1, bias2, theta3, bias3 = theta  # Added this line to unpack theta
    l, w, f, l1, l2, w1, w2 = sizes
    conv1, conv2, pooled_layer, fc1, out = delta
    dconv1, dconv2, dfilt1, dfilt2, dbias1, dbias2 = init_grads(sizes)

    dout = probs - label
    dtheta3 = dout.dot(fc1.T)
    dbias3 = dout

    dfc1 = theta3.T.dot(dout)
    dpool = dfc1.T.reshape((l2, w2//2, w2//2))

    for jj in range(0, l2):
        for i in range(0, w2, 2):
            for j in range(0, w2, 2):
                a = int(nanarg[jj , i, j, 0])
                b = int(nanarg[jj , i, j, 1])
                dconv2[jj,i+a,j+b] = dpool[jj,i//2,j//2]

    dconv2 = dconv2 * drelu(conv2)

    for jj in range(0,l2):
        for x in range(0,w2):
            for y in range(0,w2):
                dfilt2[jj]+=dconv2[jj,x,y]*conv1[:,x:x+f,y:y+f]
                dconv1[:,x:x+f,y:y+f]+=dconv2[jj,x,y]*filt2[jj]
        dbias2[jj] = np.sum(dconv2[jj])

    dconv1 = dconv1 * drelu(conv1)

    for jj in range(0,l1):
        for x in range(0,w1):
            for y in range(0,w1):
                dfilt1[jj]+=dconv1[jj,x,y]*image[:,x:x+f,y:y+f]

        dbias1[jj] = np.sum(dconv1[jj])

    return dfilt1, dfilt2, dbias1, dbias2, dtheta3, dbias3



def init_grads_velocity(filt1, filt2, bias1, bias2, theta3, bias3):
    dfilt1 = np.zeros((len(filt1), filt1[0].shape[0], filt1[0].shape[1], filt1[0].shape[2]))
    dbias1 = np.zeros((len(filt1),))
    v1 = np.zeros((len(filt1), filt1[0].shape[0], filt1[0].shape[1], filt1[0].shape[2]))
    bv1 = np.zeros((len(filt1),))

    dfilt2 = np.zeros((len(filt2), filt2[0].shape[0], filt2[0].shape[1], filt2[0].shape[2]))
    dbias2 = np.zeros((len(filt2),))
    v2 = np.zeros((len(filt2), filt2[0].shape[0], filt2[0].shape[1], filt2[0].shape[2]))
    bv2 = np.zeros((len(filt2),))

    dtheta3 = np.zeros(theta3.shape)
    dbias3 = np.zeros(bias3.shape)
    v3 = np.zeros(theta3.shape)
    bv3 = np.zeros(bias3.shape)

    return dfilt1, dfilt2, dtheta3, dbias1, dbias2, dbias3, v1, v2, v3, bv1, bv2, bv3




def batches(X, y, size):
    batches = []
    
    for idx in np.array_split(np.arange(len(X)), len(X) / size):
        X_batch = X[idx]
        y_batch = y[idx]
        batches.append((X_batch, y_batch))
    
    return batches




