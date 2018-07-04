import numpy as np
import random
import scipy.io as sio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker,cm
from matplotlib.ticker import LinearLocator,FormatStrFormatter
import matplotlib.mlab as mlab
from scipy.optimize import minimize, rosen, rosen_der

def sigmoid(z):
    g=np.matrix(np.zeros(np.shape(z)))
    g=1/(1+np.exp(-1*z))
    return g

def predict(Theta1, Theta2, X):
    (m,n)=np.shape(X)
    p=np.matrix(np.zeros((m,1)))
    a1=np.matrix(np.hstack((np.ones((m,1)),X)))
    a2=sigmoid(np.matmul(Theta1,a1.T))
    (m1,n1)=np.shape(a2)
    a2=np.matrix(np.vstack((np.ones((1,n1)),a2)))
    a3=sigmoid(np.matmul(Theta2,a2))
    p=np.argmax(a3.T,axis=1)
    p=p+1
    return p



input_layer_size  = 400
hidden_layer_size = 25
num_labels = 10     


file_name="ex3data1.mat"
data_content=sio.loadmat(file_name)
X=np.matrix(data_content['X'])
y=np.matrix(data_content['y'])
(m,n)=np.shape(X)
file_name="ex3weights.mat"
param_content=sio.loadmat(file_name)
Theta1=np.matrix(param_content['Theta1'])
Theta2=np.matrix(param_content['Theta2'])
pr=predict(Theta1,Theta2,X)
print("Train Accuracy:")
print(np.mean(pr==y)*100)
sel_ind=random.sample(range(0,m),m)

for i in sel_ind:
    p=predict(Theta1,Theta2,np.matrix(X[i,:]))
    print("Prediction from neural network:")
    print(p)
    print("Actual value: ")
    print(y[i])
    s=input("Press Enter to continue and q to quit : ")
    if s=='q':
       break

     
