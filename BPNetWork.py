import numpy as np

class NeuralNetwork:
    def __init__(self, layers, activation='sigmoid'):
        self.layers = layers
        self.activation = activation
        self.weights = [np.random.randn(layers[i], layers[i + 1]) for i in range(len(layers) - 1)]
        self.biases = [np.random.randn(1, layers[i + 1]) for i in range(len(layers) - 1)]

        if activation == 'sigmoid':
            self.activation_function = self.sigmoid
            self.activation_derivative = self.sigmoid_derivative
        elif activation == 'tanh':
            self.activation_function = self.tanh
            self.activation_derivative = self.tanh_derivative
        elif activation == 'relu':
            self.activation_function = self.relu
            self.activation_derivative = self.relu_derivative

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x <= 0, 0, 1)

    def feedforward(self, x):
        a = x
        for i in range(len(self.layers) - 1):
            z = np.dot(a, self.weights[i]) + self.biases[i]
            a = self.activation_function(z)
        return a

    def backpropagation(self, x, y, learning_rate):
        # Feedforward
        a = [x]
        z = []
        for i in range(len(self.layers) - 1):
            z.append(np.dot(a[i], self.weights[i]) + self.biases[i])
            a.append(self.activation_function(z[i]))

        # Backpropagation
        error = a[-1] - y
        delta = [error * self.activation_derivative(z[-1])]
        for i in range(len(self.layers) - 2, 0, -1):
            delta.insert(0, np.dot(delta[0], self.weights[i].T) * self.activation_derivative(z[i - 1]))

        # Gradient descent
        for i in range(len(self.layers) - 1):
            self.weights[i] -= learning_rate * np.dot(a[i].T, delta[i])
            self.biases[i] -= learning_rate * np.sum(delta[i], axis=0)

    def train(self, X, y, epochs, learning_rate):
        for i in range(epochs):
            for j in range(len(X)):
                self.backpropagation(X[j], y[j], learning_rate)

    def predict(self, X):
        y_pred = []
        for i in range(len(X)):
            y_pred.append(self.feedforward(X[i]))
        return np.array(y_pred)
if __name__=="__main__":
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    nn = NeuralNetwork(layers=[2, 4, 1], activation='sigmoid')
    nn.train(X, y, epochs=100, learning_rate=0.1)

    print(nn.predict(X))
