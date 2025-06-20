# we use cross entropy for the case of logidstic 
# again gradient descent  used to optimize
import numpy as np
def sigmoid(x):
    return 1/(1+np.exp(-x))
 
class LogisticRegression():
    def __init__(self, lr=0.001 ,n_iters=1000):
        self.lr=lr
        self.n_iters=n_iters
        self.weights=None
        self.bias=None
    def fit(self,X,y):
        n_sample, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias=0
        
        for _ in range(self.n_iters):
            # predict the result
            linear_pred = np.dot(X , self.weights) +self.bias 
            # this y_pred itself contains many values(n_sample values , is an numpy array)
            predictions=sigmoid(linear_pred)

            dw=(1/n_sample) * np.dot(X.T,(predictions-y))
            db=(1/n_sample) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.bias = self.bias -self.lr*db
    def predict(self,X):
        linear_pred=np.dot(X , self.weights) +self.bias
        predictions=sigmoid(linear_pred)
        final_pred=[0 if y<=0.5 else 1 for y in predictions]
        return final_pred