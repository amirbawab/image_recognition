#!/usr/bin/python3
import numpy as np 
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#implementation of logistic regression for one vs rest 

def augment(x):
    return np.insert(x,0,1,axis=1)
     
def weight_gen(f,c):
    return np.zeros( [f, c] )

def sigma(w,x):
    z = np.dot(x,w)
    if z.any()<0 :
        z = np.exp(z)
        return z / (1 + z)
    else:
        return 1/(1+np.exp(-z))

def cross_entropy(y, sigma):
    return -np.sum (np.dot(y, np.log(sigma)) + np.dot( (1-y), np.log(1 - sigma)))

#params
#x: input (n, 4096), sigma:(n,1) y, one hot encoded: (n, c), i: class i
def gradient(x, w, y, lam):
    s = sigma(w,x)
    x = np.transpose(x)
    return np.sum( np.dot( x, (y - s) ), axis=0 ) + lam*w
 

def mini_batch(x, y, batchsize):
    for i in range(0,x.shape[0]-batchsize+1,batchsize):
	    ind=slice(i,i+batchsize)
	    yield x[ind], y[ind]

def mini_batch_gradient_descent(x,w,y,batchsize,epoch, alpha, lam):
    for i in range(epoch):
        for batch in mini_batch(x, y, batchsize):
            xi,yi=batch
            grad=gradient(xi, w, yi, lam)
            w = w + alpha * grad
    return w

def predict_validate(x,w,y):
    prob=sigma(w,x)
    pred=np.argmax(prob,axis=1)
    pred=np.asarray(pred)
    pred=pred.flatten()
    accuracy = np.sum(pred == y) / y.shape[0] * 100
    return pred, accuracy


def main():
    datax = 'x.csv'
    datay = 'y.csv'
    x = np.genfromtxt(datax,delimiter=',')
    y = np.genfromtxt(datay,delimiter=',')
    y = y.reshape(y.shape[0],1)
   
    #Normalize x:
    x = x/255

    print(np.max(x))
    #augment x by 1 more column to facilatate matrix manipulation
    x = augment(x)
    
    #one-hot Encode y
    enc = OneHotEncoder()
    y_enc = enc.fit_transform(y)
    
    c = y_enc.shape[1]
    f = x.shape[1]
    
    #generate w:
    w = weight_gen(f,c) 

    w = mini_batch_gradient_descent(x, w, y_enc, 128, 10000, 0.01, 0.01)

    pred, accuracy = predict_validate(x, w, y)
    print("The accuracy is", accuracy)

    #Write to prediction file
    pred = pd.DataFrame(data=pred)
    pred.to_csv("Prediction_logisticOVR.csv", sep=',', index=False, encoding='utf-8')

if __name__ == "__main__":
    main()
