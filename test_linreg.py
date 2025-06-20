import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

X,y= datasets.make_regression(n_samples=100 , n_features=1 ,noise=20 , random_state=4)
X_train , X_test , y_train ,y_test=train_test_split(X,y, test_size=0.2, random_state=367)


fig=plt.figure(figsize=(8,6))
# to see how tha data looks like
plt.scatter(X[:,0] , y, color = "b" , marker ="o" , s=30)
plt.show()

reg=LinearRegression(lr=0.01)
reg.fit(X_train, y_train)
predictions=reg.predict(X_test)
print(predictions)

# to get the mse 
def mse(y_test, pred):
    return np.mean((y_test-pred)**2)

ms_e=mse(y_test,predictions)
print(ms_e)
# to see how the best fit line fits the data
plt.scatter(X_train,y_train, color='red', s=10)
plt.scatter(X_test,y_test, color='blue', s=10)
plt.plot(X_test, predictions, color= 'black' ,linewidth = 2 , label ='Prediction')
plt.show()