import numpy as np
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker,cm
from matplotlib.ticker import LinearLocator,FormatStrFormatter
import matplotlib.mlab as mlab
from scipy.optimize import minimize, rosen, rosen_der
import scipy.io as sio


def sigmoid(z):
    g=np.matrix(np.zeros(np.shape(z)))
    g=1/(1+np.exp(-1*z))
    return g

def lrCostFunction(theta, X, y,lamda):
    theta=np.matrix(theta)
    if (np.shape(theta)[0]==1):
         theta=theta.T
    m=np.shape(y)[0]
    J=0
    h=sigmoid(np.matmul(X,theta))
    temp_theta=np.multiply(theta,theta)
    temp_theta[0,0]=0
    J=np.sum((-1*np.multiply(y,np.log(h)))+(-1*np.multiply(1-y,np.log(1-h))))/m
    J=J+((lamda/(2*m))*np.sum(temp_theta))
    return J


def lrGradFunction(theta,X,y,lamda):
    theta=np.matrix(theta)
    if (np.shape(theta)[0]==1):
         theta=theta.T   
    m=np.shape(y)[0]
    grad=np.matrix(np.zeros(np.shape(theta)))
    h=sigmoid(np.matmul(X,theta))
    temp_theta=np.matrix(np.zeros(np.shape(theta)))+theta
    temp_theta[0,0]=0
    grad=np.matmul((h-y).T,X).T/m
    grad=grad+((lamda/m)*temp_theta)
    return np.squeeze(np.asarray(grad))


def oneVsAll(X, y, num_labels, lamda):
    (m,n)=np.shape(X)
    all_theta=np.matrix(np.zeros((num_labels,n+1)))
    X=np.matrix(np.hstack((np.ones((np.shape(X)[0],1)),X)))
    for i in range(num_labels):
        initial_theta=np.matrix(np.zeros((n+1,1))).T
        temp_y=np.matrix(np.array((y==(i+1)))*1)
        res = minimize(fun=lrCostFunction, x0=initial_theta, args=(X,temp_y,lamda), method='BFGS', jac=lrGradFunction,options={'gtol': 1e-6, 'disp': True})
        all_theta[i,:]=np.matrix(res.x)
    return all_theta

def predictOneVsAll(all_theta,X):
    (m,n)=np.shape(X)
    num_labels = np.shape(all_theta)[0]
    p=np.matrix(np.zeros((m,1)))
    X=np.matrix(np.hstack((np.ones((np.shape(X)[0],1)),X)))
    p=np.argmax(sigmoid(np.matmul(X,all_theta.T)),axis=1)
    p=p+1
    return p

num_labels = 10
file_name="ex3data1.mat"
data_content=sio.loadmat(file_name)
X=np.matrix(data_content['X'])
y=np.matrix(data_content['y'])
(m,n)=np.shape(X)
#sel_ind=random.sample(range(0,m),100)
#sel=X[sel_ind,:]

y_t=np.matrix([[1],[0],[1],[0],[1]])
X_t=np.matrix(((np.arange(15)+1)/10).reshape((3,5))).T
X_t=np.matrix(np.hstack((np.ones((np.shape(X_t)[0],1)),X_t)))
lambda_t = 3

theta_t = np.matrix([[-2],[-1],[1],[2]])
cost = lrCostFunction(theta_t, X_t, y_t, lambda_t)
grad = lrGradFunction(theta_t, X_t, y_t, lambda_t)


print("Cost at intial theta:")
print(cost)
print("Gradient at initial theta (zeros):")
print(grad)

l=0.1
at=oneVsAll(X,y,num_labels,l)

pred = predictOneVsAll(at, X)
print("Train Accuracy:")
print(np.mean(pred==y)*100)

