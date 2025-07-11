import numpy as np 
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from knn_scr import KNN

iris=datasets.load_iris()
X,y= iris.data , iris.target
X_train , X_test , y_train ,y_test=train_test_split(X,y, test_size=0.2, random_state=367)

plt.figure()
plt.scatter(X[:,2], X[:,3], c=y , edgecolors='k', s=20)
plt.show()

clf=KNN(k=5)
clf.fit(X_train,y_train)
predictions=clf.predict(X_test)
print(predictions)