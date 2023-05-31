
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import math
import random
import time
from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# p = [[1,2,3,4,5,6],[1,2,3,4,5,6],[1,2,3,4,5,6]]
# f = np.array([[1,2,3],[3,4,5]])
# p=np.array(p)
# zw = p[0].reshape((6,1))
# print(np.dot(zw,zw.T))
# print(copy([1,2,3,4,5],3).shape)

# train_path = "COL774_fmnist"
# test_path= "COL774_fmnist"
#train_path = str(sys.argv[1])
#test_path = str(sys.argv[2])

x_in = pd.read_csv('fmnist_train.csv',header=None)

testx = pd.read_csv("fmnist_test.csv",header=None)
testx = testx.to_numpy()


x=x_in.to_numpy()
np.random.shuffle(x)
X = x[:,0:784]
X=X/255
Y = x[:,784]

np.random.shuffle(testx)
testX = testx[:,0:784]
testX=testX/255
testY = testx[:,784]

def predict(theta,b,X,Y,l): # returns the accuracy and confusion matrix
    o = X
    for j in range(len(l)-1):
        o=copy(b[j],X.shape[0])+np.dot(o,theta[j]) # o shape is (m,l)
        if(j==len(l)-2):
            o=softmax(o)
        else:
            o=sig(o)
    cf = []
    r = l[len(l)-1]
    for i in range(r):
        temp=[]
        for  j in range(r):
            temp.append(0)
        cf.append(temp)
    acc=0
            
    for i in range(X.shape[0]):
        if(Y[i]==maxin(o[i])):
            acc=acc+1
        cf[Y[i]][maxin(o[i])]=cf[Y[i]][maxin(o[i])]+1
    return [acc/X.shape[0],cf]

def predict_r(theta,b,X,Y,l): # returns the accuracy and confusion matrix
    o = X
    for j in range(len(l)-1):
        o=copy(b[j],X.shape[0])+np.dot(o,theta[j]) # o shape is (m,l)
        if(j==len(l)-2):
            o=sig(o)
        else:
            o=relu(o)
    cf = []
    r = l[len(l)-1]
    for i in range(r):
        temp=[]
        for  j in range(r):
            temp.append(0)
        cf.append(temp)
    acc=0
            
    for i in range(X.shape[0]):
        if(Y[i]==maxin(o[i])):
            acc=acc+1
        cf[Y[i]][maxin(o[i])]=cf[Y[i]][maxin(o[i])]+1
    return [acc/X.shape[0],cf]

def maxin(y):#checked
    ind=0
    n=len(y)
    max=0
    for i in range(n):
        if(y[i]>max):
            max=y[i]
            ind=i
    return ind

def copy(g,l):#checked
    t=np.zeros((l,len(g)))
    for i in range(l):
        t[i]=g.reshape(len(g),)
    return t

# def f_p(theta,b,X,Y,start,length,l,n): # returns the cost function for i batch
#     er=0
#     oT = []
#     for i in range(start,start+length):
#         o1=[]
#         o=X[i].reshape((n,1))
#         for j in range(len(l)-1):
#             o = copy(b[j],m)+np.dot(theta[j],o)
#             o=sig(o)
#             o1.append(o)
#         oT.append(o1)
#         er=er+er_en(o,Y[i])
#     return [er/(2*length),oT]
#      #changes needed


def alt(t):#checked
  n = t.shape[0]
  m = t.shape[1]
  for i in range(n):
    for j in range(m):
      t[i,j]=random.random()*0.001
  return t

def f_p(theta,b,X,Y,start,length,l):#checked
    oT=[]
    o=X[start:start+length]
    for j in range(len(l)-1):
        o=copy(b[j],length)+np.dot(o,theta[j]) # o shape is (m,l)
        if(j==len(l)-2):
            o=softmax(o)
        else:
            o=sig(o)
        oT.append(o)
    er=0
    for j in range(length):
        er = er + er_en(oT[len(oT)-1][j],Y[start+j])
    return [er/(2*length),oT]

def f_pr(theta,b,X,Y,start,length,l):#checked
    oT=[]
    o=X[start:start+length]
    for j in range(len(l)-1):
        o=copy(b[j],length)+np.dot(o,theta[j]) # o shape is (m,l)
        if(j==len(l)-2):
            o=sig(o)
        else:
            o=relu(o)
        oT.append(o)
    er=0
    for j in range(length):
        er = er + er_en(oT[len(oT)-1][j],Y[start+j])
    return [er/(2*length),oT]

def er_en(o,y):# here o is a list of length r   checked....
    s = 0
    for i in range(len(o)):
        if(i==int(y)):
            s= s+(o[i]-1)*(o[i]-1)
        else:
            s=s+(o[i]-0)*(o[i]-0)
            
    return s

def sig(f):#checked
    return 1/(1+np.exp(-f))

def relu(f):
  return np.maximum(f,0)

def softmax(f):#checked
    n = f.shape[0]
    for i in range(n):
        t=f[i]
        ex = np.exp(t)
        s=np.sum(ex)
        ex =ex/s
        f[i]=ex
    return f


def b_p(theta,b,X,Y,l,n,eta,oT,m,start):
    i = len(oT)-1
    r= l[len(l)-1]
    o = oT[i] # shape is (m,r)
    #theta shape is (layer-1,layer)
    y = np.zeros((m,r))
    for j in range(m):
        y[j] = en_co(Y[start+j],r).reshape((r,))
    don = np.multiply(o,1-o) # shape is (m,r)
    dJo = y-o # shape is (m,r)
    dJn = -1*(np.multiply(dJo,don)) # shape is (m,r)
    dnt = oT[i-1] # shape is (m,layer-1)
    dJt = np.zeros((l[i],l[i+1]))
    for j in range(m):
        a1 = np.array(dnt[j]).reshape((len(dnt[j]),1))
        a2 = np.array(dJn[j]).reshape((len(dJn[j]),1))
        dJt = dJt+np.dot(a1,a2.T)
    dJt = dJt/m
    theta[i] = theta[i] -eta*dJt
    b[i] = b[i] - np.mean(dJn,axis=0).reshape((len(b[i]),1))
    i=i-1
    while(i>-1):
        #theta[i] shape is (layer-1,layer)
        t1 = theta[i+1] # shape(layer,layer+1)
        #dJn(old) shape is (m,layer+1)
        o = oT[i]
        don = np.multiply(o,1-o) # shape is (m,layer)
        dJo = np.dot(dJn,t1.T) # shape is (m,layer)
        dJn = np.multiply(dJo,don) # shape is (m,layer)
        if(i==0):
            dnt = X[start:start+m]
        else:
            dnt = oT[i-1] # shape is (m,layer-1)
        dJt = np.zeros((l[i],l[i+1]))
        for j in range(m):
            a1 = np.array(dnt[j]).reshape((len(dnt[j]),1))
            a2 = np.array(dJn[j]).reshape((len(dJn[j]),1))
            dJt = dJt+np.dot(a1,a2.T)
        dJt = dJt/m
        theta[i] = theta[i]-eta*dJt
        b[i] = b[i]-np.mean(dJn,axis=0).reshape((len(b[i]),1))
        i=i-1
    return [theta,b]

def b_p_r(theta,b,X,Y,l,n,eta,oT,m,start):
    i = len(oT)-1
    r= l[len(l)-1]
    o = oT[i] # shape is (m,r)
    #theta shape is (layer-1,layer)
    y = np.zeros((m,r))
    for j in range(m):
        y[j] = en_co(Y[start+j],r).reshape((r,))
    don = 1*(o>0)#np.multiply(o,1-o)  #shape is (m,r)
    dJo = y-o # shape is (m,r)
    dJn = -1*(np.multiply(dJo,don))  # shape is (m,r)
    dnt = oT[i-1] # shape is (m,layer-1)
    dJt = np.zeros((l[i],l[i+1]))
    for j in range(m):
        a1 = np.array(dnt[j]).reshape((len(dnt[j]),1))
        a2 = np.array(dJn[j]).reshape((len(dJn[j]),1))
        dJt = dJt+np.dot(a1,a2.T)
    dJt = dJt/m
    theta[i] = theta[i] -eta*dJt
    b[i] = b[i] - np.mean(dJn,axis=0).reshape((len(b[i]),1))
    i=i-1
    while(i>-1):
        #theta[i] shape is (layer-1,layer)
        t1 = theta[i+1] # shape(layer,layer+1)
        #dJn(old) shape is (m,layer+1)
        o = oT[i]
        don = 1*(o>0) # np.multiply(o,1-o) #shape is (m,layer)
        dJo = np.dot(dJn,t1.T) # shape is (m,layer)
        dJn = -1*(np.multiply(dJo,don)) # shape is (m,layer)
        if(i==0):
            dnt = X[start:start+m]
        else:
            dnt = oT[i-1] # shape is (m,layer-1)
        dJt = np.zeros((l[i],l[i+1]))
        for j in range(m):
            a1 = np.array(dnt[j]).reshape((len(dnt[j]),1))
            a2 = np.array(dJn[j]).reshape((len(dJn[j]),1))
            dJt = dJt+np.dot(a1,a2.T)
        dJt = dJt/m
        theta[i] = theta[i]-eta*dJt
        b[i] = b[i]-np.mean(dJn,axis=0).reshape((len(b[i]),1))
        i=i-1
    return [theta,b]


def en_co(y,r):#checked
    t=np.zeros((r,1))
    t[int(y),0]=1
    return t

def NN(m,n,r,h,l,X,Y,testX,testY):# m = batch size, n = no of features, r = no of target classes, h = no of hidden layers, 
    #l=list of units per hidden layer , hidden means excluding  output layer
    l.insert(0,n)
    l.append(r)
    theta = []
    b=[]
    for i in range(1,len(l)):
        t=np.ones((l[i-1],l[i]))
        t=alt(t)
        theta.append(t)
    for i in range(1,len(l)):
        t=np.ones((l[i],1))
        t=alt(t)
        b.append(t)
    er = 0
    er1 = 0.01
    epoch = 0
    eta=1
    start_time = time.time()
    while(epoch<100):
        i=0
        o=[]
        batch = 0
        er1=er
        while(i+m<X.shape[0]):
            sss= f_p(theta,b,X,Y,i,m,l)
            oT = sss[1]
            er = er+sss[0]
            ddd = b_p(theta,b,X,Y,l,n,eta,oT,m,i)
            theta = ddd[0]
            b = ddd[1]
            batch = batch+1
            i=i+m
            
        er = er/batch
        print(er)
#         er1=er
#         sss= f_p1(theta,b,X,Y,i,X.shape[0]-i,l)
#         oT = sss[1]
#         er1 = er
#         er = sss[0]
#         i=i+m
#         ddd = b_p(theta,b,X,Y,l,n,eta,oT,m,i)
#         theta = ddd[0]
#         b = ddd[1]
        epoch = epoch+1
        print("Epochs done "+str(epoch))
    #print(theta)
#     print("b = ")
#     print(b)
    end_time = time.time()
    print("time taken = "+str(end_time-start_time))
    return [predict(theta,b,X,Y,l),predict(theta,b,testX,testY,l)]

def NN_adp_lr(m,n,r,h,l,X,Y,testX,testY):# m = batch size, n = no of features, r = no of target classes, h = no of hidden layers, 
    #l=list of units per hidden layer , hidden means excluding  output layer
    l.insert(0,n)
    l.append(r)
    theta = []
    b=[]
    for i in range(1,len(l)):
        t=np.ones((l[i-1],l[i]))
        t=alt(t)
        theta.append(t)
    for i in range(1,len(l)):
        t=np.ones((l[i],1))
        t=alt(t)
        b.append(t)
    er = 0
    er1 = 0.01
    epoch = 0
    eta=1
    start_time = time.time()
    while(epoch<100):
        i=0
        o=[]
        batch = 0
        er1=er
        eta = eta/math.sqrt(epoch+1)
        while(i+m<X.shape[0]):
            sss= f_p(theta,b,X,Y,i,m,l)
            oT = sss[1]
            er = er+sss[0]
            
            ddd = b_p(theta,b,X,Y,l,n,eta,oT,m,i)
            theta = ddd[0]
            b = ddd[1]
            batch = batch+1
            i=i+m
            
        er = er/batch
        print(er)
#         er1=er
#         sss= f_p1(theta,b,X,Y,i,X.shape[0]-i,l)
#         oT = sss[1]
#         er1 = er
#         er = sss[0]
#         i=i+m
#         ddd = b_p(theta,b,X,Y,l,n,eta,oT,m,i)
#         theta = ddd[0]
#         b = ddd[1]
        epoch = epoch+1
        print("Epochs done "+str(epoch))
    #print(theta)
#     print("b = ")
#     print(b)
    end_time = time.time()
    print("time taken = "+str(end_time-start_time))
    return [predict(theta,b,X,Y,l),predict(theta,b,testX,testY,l)]


def NN_relu(m,n,r,h,l,X,Y,testX,testY):# m = batch size, n = no of features, r = no of target classes, h = no of hidden layers, 
    #l=list of units per hidden layer , hidden means excluding  output layer
    l.insert(0,n)
    l.append(r)
    theta = []
    b=[]
    for i in range(1,len(l)):
        t=np.ones((l[i-1],l[i]))
        t=alt(t)
        theta.append(t)
    for i in range(1,len(l)):
        t=np.ones((l[i],1))
        t=alt(t)
        b.append(t)
    er = 0
    er1 = 0.01
    epoch = 0
    eta=1
    start_time = time.time()
    while(abs(er-er1)>0.0002):
        i=0
        o=[]
        batch = 0
        er1=er
        while(i+m<=X.shape[0]):
            sss= f_pr(theta,b,X,Y,i,m,l)
            oT = sss[1]
            er = er+sss[0]
            ddd = b_p_r(theta,b,X,Y,l,n,eta,oT,m,i)
            theta = ddd[0]
            b = ddd[1]
            batch = batch+1
            i=i+m
            
        er = er/batch
        print(er)
#         er1=er
#         sss= f_p1(theta,b,X,Y,i,X.shape[0]-i,l)
#         oT = sss[1]
#         er1 = er
#         er = sss[0]
#         i=i+m
#         ddd = b_p(theta,b,X,Y,l,n,eta,oT,m,i)
#         theta = ddd[0]
#         b = ddd[1]
        epoch = epoch+1
        print("Epochs done "+str(epoch))
    #print(theta)
#     print("b = ")
#     print(b)
    end_time = time.time()
    print("time taken = "+str(end_time-start_time))
    return [predict_r(theta,b,X,Y,l),predict_r(theta,b,testX,testY,l)]

def NN_scikit(X,Y,testX,testY):
    clf = MLPClassifier( activation="logistic",max_iter=300,hidden_layer_sizes={35,}).fit(X,Y)
    print(clf.score(X,Y))
    print(clf.score(testX,testY))
    clf = MLPClassifier( activation="relu",max_iter=300,hidden_layer_sizes={35,}).fit(X,Y)
    print(clf.score(X,Y))
    print(clf.score(testX,testY))
    return


# print(part_c(100,X.shape[1],10,1,[5],X,Y,testX,testY))
# print(part_c(100,X.shape[1],10,1,[10],X,Y,testX,testY))
# print(part_c(100,X.shape[1],10,1,[15],X,Y,testX,testY))
# print(part_c(100,X.shape[1],10,1,[20],X,Y,testX,testY))

#print(part_d(100,X.shape[1],10,1,[25],X,Y,testX,testY))


# theta = []
# b=[]
# l=[784,50,10]
# for i in range(1,len(l)):
#     t=np.ones((l[i-1],l[i]))
#     t=alt(t)
#     theta.append(t)
# for i in range(1,len(l)):
#     t=np.ones((l[i],1))
#     t=alt(t)
#     b.append(t)
# print(predict(theta,b,X,Y,l))

# test = np.ones((3,5))
# print(softmax(test))

def check_b(b):
    for j in range(len(b)):
        if(b[j].shape[1]!=1):
            return False
    return True

