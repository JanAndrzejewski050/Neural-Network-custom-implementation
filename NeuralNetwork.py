import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Model
class NN:
    def __init__(self, layers):
        self.layers = layers
        self.n = len(layers)
        self.A = [np.zeros((x, 1)) for x in layers] 
        self.B = [np.zeros((x, 1)) for x in layers]
        self.W = [np.random.normal(0, 0.1, size=(layers[i], layers[i+1])) for i in range(self.n - 1)]

    def forward(self, X):
        self.A[0] = X
        
        for i in range(1, self.n):
            self.A[i] = sigmoid(self.W[i-1].T @ self.A[i-1] + self.B[i])

        return self.A[-1]  
    

# Optimizer
class GD:
    def __init__(self, lr):
        self.lr = lr

    def step(self, model, X, y):
        y_pred = model.forward(X)
        Error = y - y_pred

        for i in range(model.n-1, 0, -1):
            d_sigmoid = model.A[i] * (1 - model.A[i])
            d_Error = -2 * Error

            d_B = d_Error * d_sigmoid
            d_W = model.A[i-1] @ (d_sigmoid * d_Error).T
            d_A = model.W[i-1] @ (d_Error * d_sigmoid)

            model.B[i] -= self.lr * d_B
            model.W[i-1] -= self.lr * d_W
            Error = d_A #* self.lr
