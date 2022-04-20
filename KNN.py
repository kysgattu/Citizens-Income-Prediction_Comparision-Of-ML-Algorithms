#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import Counter


# In[2]:


from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum(((x1)-(x2))**2))


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self._predict(x) for x in X]
        return np.array(predicted_labels)

    def _predict(self, x):
        # compute distances
        distances = [euclidean_distance(x, xtrain) for xtrain in self.X_train]

        # get k nearest samples, lables
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority vote
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# In[3]:


from DataPreprocessing import *


# In[4]:


from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn import metrics 


# In[5]:


model = KNN(n_neighbors = 3)


# In[6]:


model.fit(x_train,y_train)


# In[7]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[8]:


knn_pred = model.predict(x_test)


# In[11]:


accuracy_score(knn_pred,y_test)


# In[ ]:





# In[ ]:




