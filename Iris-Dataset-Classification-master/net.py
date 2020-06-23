#libraries
import numpy as np
from sklearn import datasets
from functions.functions import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

#load datasets
iris = datasets.load_iris()
x = iris.data
y = iris.target
y = y.reshape(150,1)

#OneHotEncoder
list = []
for i in y:
    c = np.zeros(3)
    c[int(i)] = 1
    list.append(c)
y = np.array(list)

#train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size = 0.8, test_size = 0.2)

ytrain = ytrain.T.reshape(3,120)
xtrain = xtrain.T

ytest = ytest.T.reshape(3,30)
xtest = xtest.T

#feature scaling
sc = StandardScaler()
xtrain = sc.fit_transform(xtrain)
xtest = sc.fit_transform(xtest)

#initialization
m = xtrain.shape[1]
lambd = 0.1


class net():
    """doctring for net"""

    def __init__(self, xtrain, ytrain):
        #layer size
        n_x = xtrain.shape[0]
        n_1 = 3
        n_2 = 2
        n_y = ytrain.shape[0]

        np.random.seed(2)
        self.cache = {}

        self.cache['x'] = xtrain
        self.cache['y'] = ytrain
        #HE initialization
        self.cache['w1'] = np.random.randn(n_1, n_x) * (np.sqrt(2.0 / n_x))
        self.cache['w2'] = np.random.randn(n_2, n_1) * (np.sqrt(2.0 / n_1))
        self.cache['w3'] = np.random.randn(n_y, n_2) * (np.sqrt(2.0 / n_2))
        #bias initialization
        self.cache['b1'] = np.zeros((n_1, 1))
        self.cache['b2'] = np.zeros((n_2, 1))
        self.cache['b3'] = np.zeros((n_y, 1))

    def forward(self, x = None):
        #forward propagation with bias

        if x is None:
            x = self.cache['x']

        w1, w2, w3, b1, b2, b3 = self.get('w1', 'w2', 'w3', 'b1', 'b2', 'b3')

        z1 = np.dot(w1, x) + b1
        a1 = sigmoid(z1)
        dz1 = sigmoid_prime(z1)

        z2 = np.dot(w2, a1) + b2
        a2 = sigmoid(z2)
        dz2 = sigmoid_prime(z2)

        z3 = np.dot(w3, a2) + b3
        a3 = softmax(z3)

        self.put(dz1 = dz1, dz2 = dz2, a1 = a1, a2 = a2, a3 = a3)
        return a3

    def cost(self, ypred, y= None):
        #cost with l2 regularization

        if y is None:
            y = self.cache['y']

        w1, w2, w3 = self.get('w1', 'w2', 'w3')

        # logprobs = np.multiply(-np.log(ypred),y) + np.multiply(-np.log(1-ypred),1-y)
        # cost_entropy = 1./m * np.sum(logprobs)

        cost_entropy = cost = -np.mean(y * np.log(ypred + 1e-8))
        l2_regularization = lambd / (2*m) * (np.sum(np.square(w1)) + np.sum(np.square(w2)) + np.sum(np.square(w3)))

        cost = cost_entropy + l2_regularization
        return cost

    def backward(self):
        #back propagation with bias and regularization
        a1, a2, a3, y, w1, w2, w3, dz1, dz2, x = self.get('a1', 'a2', 'a3', 'y', 'w1', 'w2', 'w3', 'dz1', 'dz2', 'x')

        t3 = a3 - y
        dw3 = 1./m * (np.dot(t3, a2.T) + ((lambd/m) * w3))
        db3 = 1./m * (np.sum(t3, axis=1, keepdims = True))

        t2 = np.multiply(dz2, np.dot(w3.T, t3))
        dw2 = 1./m * (np.dot(t2, a1.T) + ((lambd/m) * w2))
        db2 = 1./m * (np.sum(t2, axis=1, keepdims = True))

        t1 = np.multiply(dz1, np.dot(w2.T, t2))
        dw1 = 1./m * (np.dot(t1, x.T) + ((lambd/m) * w1))
        db1 = 1./m * (np.sum(t1, axis=1, keepdims = True))

        self.put(dw3 = dw3, dw2 = dw2, dw1 = dw1, db3 = db3, db2 = db2, db1 = db1)

    def update(self, rate = 0.05):
        w1, w2, w3, dw1, dw2, dw3, b1, b2, b3, db1, db2, db3 = self.get('w1', 'w2', 'w3', 'dw1', 'dw2', 'dw3', 'b1', 'b2', 'b3', 'db1', 'db2', 'db3')

        w1 -= rate * dw1
        b1 -= rate * db1

        w2 -= rate * dw2
        b2 -= rate * db2

        w3 -= rate * dw3
        b3 -= rate * db3

        self.put(w1 = w1, w2 = w2, w3 = w3, b1 = b1, b2 = b2, b3 = b3)

    def put(self, **kwargs):
        for key, value in kwargs.items():
            self.cache[key] = value


    def get(self, *args):
        x = tuple(map(lambda x: self.cache[x], args))
        return x

def main():

    costs = []
    epoch = 5000
    n = net(xtrain, ytrain)

    for i in range(epoch):
        ypred = n.forward()
        costs.append(n.cost(ypred))
        n.backward()
        n.update()

    yhat = n.forward(xtest)

    yhat_rounded = np.argmax(yhat, axis = 0)
    ytest_rounded = np.argmax(ytest, axis = 0)

    #classification report on test dataset
    print(classification_report(ytest_rounded, yhat_rounded))

    #cost vs epochs graph
    plt.plot(costs)
    plt.show()

if __name__ == '__main__':
    main()
