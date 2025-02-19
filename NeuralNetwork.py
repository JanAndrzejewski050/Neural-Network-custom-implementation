import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Model
class NN_Sigmoid:
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
class GD_Sigmoid:
    def __init__(self, lr):
        self.lr = lr

    def step(self, model, X, y):
        y_pred = model.forward(X)
        Error = y - y_pred

        d_Error = -2 * Error

        for i in range(model.n-1, 0, -1):
            d_sigmoid = model.A[i] * (1 - model.A[i])
            
            d_B = d_Error * d_sigmoid
            d_W = model.A[i-1] @ (d_sigmoid * d_Error).T
            d_A = model.W[i-1] @ (d_sigmoid * d_Error)

            model.B[i] -= self.lr * d_B
            model.W[i-1] -= self.lr * d_W
            d_Error = d_A #* self.lr



def Leaky_Relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

def d_Leaky_Relu(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)  # Gradient is not to equal 0


# Model
class NN_Leaky_ReLu:
    def __init__(self, layers):
        self.layers = layers
        self.n = len(layers)
        self.A = [np.zeros((x, 1)) for x in layers] 
        self.B = [np.zeros((layers[i+1], 1)) for i in range(self.n - 1)]
        #self.B = [np.random.normal(0, np.sqrt(2 / layers[i+1]), size=(layers[i+1], 1)) for i in range(self.n - 1)]
        self.W = [np.random.randn(layers[i], layers[i+1]) * np.sqrt(2 / layers[i]) for i in range(self.n - 1)]

    def forward(self, X):
        self.A[0] = X
        for i in range(self.n - 1):
            #print(self.A[i+1].shape, self.A[i].shape,  self.W[i].shape, self.B[i].shape)
            self.A[i+1] = Leaky_Relu(self.W[i].T @ self.A[i] + self.B[i])
            
        return self.A[-1] 
    

# Optimizer
class GD_Leaky_ReLu:
    def __init__(self, lr):
        self.lr = lr

    def step(self, model, X, y):
        y_pred = model.forward(X)
        Error = y - y_pred

        d_Error = -2 * Error

        for i in range(model.n-1, 0, -1):
            d_leaky_relu = d_Leaky_Relu(model.A[i])
            d_B = d_Error * d_leaky_relu
            d_W = model.A[i-1] @ (d_leaky_relu * d_Error).T
            d_A = model.W[i-1] @ (d_leaky_relu * d_Error)

            #print(d_B.shape, model.B[i].shape)

            model.B[i-1] -= self.lr * d_B
            model.W[i-1] -= self.lr * d_W

            d_Error = d_A
