import numpy as np
class BPNetwork():
    def __init__(self,layers):
        self.layers = layers
        self.activation = self.sigmoid
        self.weight = [np.random.randn(layers[i],layers[i+1]) for i in range(len(layers)-1)]
        self.activationDerivative = self.sigmoidDerivative()
    def sigmoid(self,x):
        return 1/(1+np.exp(-x))
    def sigmoidDerivative(self,x):
        return self.sigmoid(x)*(1-self.sigmoid(x))
    def feedForword(self):

    def backPropagation(self):
    def train(self):
    def predict(self):