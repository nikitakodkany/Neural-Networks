import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

iris = datasets.load_iris()
x = iris.data
y = iris.target
y.reshape(150,1)

for i in y:
    if i == 0 or i == 1:
        i = 0
    else:
        i = 1

xtrain, xtest, ytrain, ytest = train_test_split(x,y,train_size = 0.8, test_size = 0.2) #random_state = 0)

def initialization(n_x, n_h, n_y):
    np.random.seed(2)
    Wh = np.random.randn(n_h, n_x) * np.sqrt(2/n_x)
    Wo = np.random.randn(n_h, n_y) * np.sqrt(2/n_h)

    parameters = {"Wh": Wh,"Wo": Wo}
    return parameters

def layer_sizes(xtrain, ytrain):
    n_x = xtrain.shape[1]
    n_h = 6
    ytrain = ytrain.reshape(120,1)
    n_y = ytrain.shape[1]
    return (n_x, n_h, n_y)

def relu(Z):
     return np.maximum(0, Z)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def forward_propagation(xtrain, parameters):
    Wh = parameters['Wh']
    Wo = parameters['Wo']

    Zh = np.dot(Wh, xtrain.T)
    Ah = sigmoid(Zh)
    Zo = np.dot(Wo.T, Ah)
    Ao = sigmoid(Zo)

    cache = {"Zh": Zh,"Ah": Ah,"Zo": Zo,"Ao": Ao}
    return Ao, cache

def costfunction(Ao, ytain, parameters):
    m = ytrain.shape[0]
    Wh = parameters['Wh']
    Wo = parameters['Wo']
    #mean squared error
    cost = 1/m * np.sum((Ao-ytrain)**2)
    return cost

def relu_prime(Z):
    Z[Z < 0] = 0
    Z[Z > 0] = 1
    return Z

def sigmoid_prime(z):
  return sigmoid(z) * (1-sigmoid(z))

def backward_propagation(parameters, cache, xrain, ytrain):
    m = xtrain.shape[0]
    Wh = parameters['Wh']
    Wo = parameters['Wo']
    Zo = cache['Zo']
    Zh = cache['Zh']
    Ah = cache['Ah']
    Ao = cache['Ao']

    dZo = sigmoid_prime(Zo)
    dZh = sigmoid_prime(Zh)

    #backward propagation
    To = (Ao - ytrain) * dZo
    Th = np.multiply(dZh.T,np.dot(To.T,Wo.T))

    dWo = np.dot(Ah,To.T)
    dWh = np.dot(xtrain.T,Th)

    grads = {"dWh": dWh, "dWo": dWo}
    return grads

def updation(parameters, grads, lr=0.05):
    Wh = parameters['Wh']
    Wo = parameters['Wo']
    dWh = grads['dWh']
    dWo = grads['dWo']

    #updation
    Wh = Wh - lr * dWh.T
    Wo = Wo - lr * dWo

    parameters = {"Wh": Wh, "Wo": Wo}
    return parameters

costlist = []

def model(xtrain, train, n_h, num_iterations=10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(xtrain, ytrain)[0]
    n_y = layer_sizes(xtrain, ytrain)[2]

    parameters = initialization(n_x, n_h, n_y)
    Wh = parameters['Wh']
    Wo = parameters['Wo']

    for i in range(0, num_iterations):
        Ao, cache = forward_propagation(xtrain, parameters)
        cost = costfunction(Ao, ytrain, parameters)
        grads = backward_propagation(parameters, cache, xtrain, ytrain)
        parameters = updation(parameters, grads)

        if print_cost and i % 1000 == 0 and i!=0:
            costlist.append(cost)
            print ("Cost after iteration %i: %f" % (i, cost))
    print("Training Complete")

    for i in range(len(ytest)):
        if ytest[i] == 0 or ytest[1] == 1:
            ytest[i] = 0
        else:
            ytest[i] = 1

    yhat, cache2 = forward_propagation(xtest,parameters)
    for elem in yhat:
        if elem[0] >= 0.5:
            elem[0] = 1
        else:
            elem[0] = 0

    counter = 0
    yhat = np.array([int(item) for sublist in yhat for item in sublist])

    for i in range(30):
        if yhat[i] == ytest[i]:
            counter+= 1
    print("Model accuracy is "+str((counter/30*100)))
    return parameters,n_h

parameters = model(xtrain, ytrain , n_h = 6, num_iterations=10000, print_cost=True)

plt.plot(costlist)
plt.show()
