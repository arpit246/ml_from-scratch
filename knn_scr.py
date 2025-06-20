# calculate distance from all other points , get the closest k points
#regression-avg, classification -label with majority value
import numpy as np
from collections import Counter
def euclidean_dist(x1,x2):
    dist = np.sqrt(np.sum((x1-x2)**2))
    return dist

class KNN:
    def __init__(self,k=3):
        self.k = k


    def fit(self, X,y):
        self.X_train = X
        self.y_train = y


    def predict(self, X):
        predictions = [self.help_predict(x) for x in X]
        return predictions
    

    def help_predict(self,x):
        # compute the distance
        distances=[euclidean_dist(x, x2) for x2 in self.X_train]
        # get tyhe closest k 
        #np.argsort give the indices of the smallest numbers in distances array
        k_indices= np.argsort(distances)[:self.k]
        k_nearest_labels=[self.y_train[i] for i in k_indices]
        # majority
        # unique, counts= np.unique(k_nearest_labels , return_counts=True)
        # most_common=unique[np.argmax(counts)]
        # return most_common
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]        
