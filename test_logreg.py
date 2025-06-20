import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import datasets
import matplotlib.pyplot as plt
from logistic_regression import LogisticRegression
bc=datasets.load_breast_cancer()
X,y= bc.data , bc.target
X_train , X_test , y_train ,y_test=train_test_split(X,y, test_size=0.2, random_state=367)

print(X.shape)
clf=LogisticRegression()
clf.fit(X_train , y_train)
y_pred=clf.predict(X_test)
print(y_pred)

def acc(y_pred, y):
    return np.sum(y_pred==y)/y.shape[0]

accuracy=acc(y_pred,y_test)
print(accuracy)