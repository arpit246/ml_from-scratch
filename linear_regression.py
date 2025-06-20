#y=wx +b, find the best fit line ,  gradient descent 
# to minimize mse
import numpy as np

class LinearRegression:
    def __init__(self, lr= 0.001 , n_iters=1000):
        self.lr= lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None

    def fit(self, X,y):
        n_samples, n_features = X.shape
        self.weights= np.zeros(n_features)
        self.bias=0
        for _ in range(self.n_iters):
    # if we run this part without loop it will be run just once and not optimally fir the a=data
    # to fit we have to put it in loop
    # dot of x with weights  and add with bias will give y_pred
            y_pred =np.dot(X , self.weights) +self.bias
            # this dot product definition
            #  of numpy automatically calculates and give us the sum value 
            dw=(1/n_samples) * np.dot(X.T, (y_pred-y)) # we have to do transpose then only it will be able to multiply correctly
            db=(1/n_samples) * np.sum(y_pred-y)
            # these updates has been written assuming mse as the error
            self.weights= self.weights - self.lr* dw
            self.bias= self.bias - self.lr * db

    def predict(self, X):
        y_pred= np.dot( X , self.weights) + self.bias
        return y_pred
