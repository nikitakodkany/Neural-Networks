#libraries
import numpy as np
from sklearn import datasets
from utility.functions import *
import matplotlib.pyplot as plt
from progressbar import ProgressBar
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


#load dataset
cancer = datasets.load_breast_cancer()
x = cancer.data
y = cancer.target
y = y.reshape(569,1)

#plot the dataset
colors = ['orange', 'blue']
for i in range(x.shape[0]):
    plt.scatter(x[i,0], x[i,1], s=7, color = colors[int(y[i])])
plt.show()

#splitting the datasets
xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size=0.8, test_size=0.2)
ytrain = ytrain.reshape(455,1)

#feature scaling
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)



class network():
    """doctring for Network"""

    def __init__(self, xtrain, ytrain):
        #layer size
        n_x = xtrain.shape[1]
        n_1 = 4
        n_2 = 3
        n_y = ytrain.shape[1]

        np.random.seed(2)
        self.cache = {}
        self.cache['x'] = xtrain
        self.cache['y'] = ytrain
        self.cache['w1'] = np.random.randn(n_x, n_1)
        self.cache['w2'] = np.random.randn(n_1, n_2)
        self.cache['w3'] = np.random.randn(n_2, n_y)

    def forward(self, x = None):
        if x is None:
            x = self.cache['x']

        w1, w2, w3 = self.get('w1', 'w2', 'w3')

        z1 = np.dot(x, w1)
        a1 = sigmoid(z1)
        dz1 = sigmoid_prime(z1)

        z2 = np.dot(a1, w2)
        a2 = sigmoid(z2)
        dz2 = sigmoid_prime(z2)

        z3 = np.dot(a2, w3)
        a3 = relu(z3)
        dz3 = relu_prime(z3)

        self.put(dz1=dz1,dz2=dz2,dz3=dz3,a1=a1,a2=a2, a3=a3)
        return a3

    def cost(self, ypred, y= None):
        if y is None:
            y = self.cache['y']

        # mean square error
        cost = 1/y.shape[0] * np.sum((ypred-y)**2)

        return cost


    def backward(self):
        a1,a2,a3,y,w1,w2,w3,dz1,dz2,dz3,x = self.get('a1','a2','a3','y','w1','w2','w3','dz1','dz2','dz3','x')

        t3 = (a3 - y) * dz3
        t2 = np.multiply(dz2, np.dot(t3, w3.T))
        t1 = np.multiply(dz1, np.dot(t2, w2.T))

        dw3 = np.dot(a2.T, t3)
        dw2 = np.dot(a1.T, t2)
        dw1 = np.dot(x.T, t1)

        self.put(dw2 = dw2,dw1=dw1)


    def update(self, rate = 0.00001):
        w1,w2,dw1,dw2 = self.get('w1','w2','dw1','dw2')
        w1 = w1 - rate * dw1
        w2 = w2 - rate * dw2

        self.put(w1=w1, w2=w2)


    def put(self, **kwargs):
        for key, value in kwargs.items():
            self.cache[key] = value


    def get(self, *args):
        x = tuple(map(lambda x: self.cache[x], args))
        return x


def main():

    bar = ProgressBar()
    costs = []
    epoch = 100000
    net = network(xtrain, ytrain)

    for iteration in bar(range(epoch)):
        ypred = net.forward()
        costs.append(net.cost(ypred))
        net.backward()
        net.update()


    yhat = net.forward(xtest)
    print("Classification report")
    print(classification_report(ytest,yhat.round()))


    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()
