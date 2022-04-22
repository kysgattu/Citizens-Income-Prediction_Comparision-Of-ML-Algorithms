#!/usr/bin/env python
# coding: utf-8

# **Import Libraries**

# In[1]:


from numpy import log, dot, e
from numpy.random import rand


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


class LogisticRegression:
    
    def sigmoid(self, z): return 1 / (1 + e**(-z)) #define sigmoid function
    
    #define costfunction based on log of sigmoid
    def cost_function(self, X, y, weights):                 
        z = dot(X, weights)
        predict_1 = y * log(self.sigmoid(z))
        predict_0 = (1 - y) * log(1 - self.sigmoid(z))
        return -sum(predict_1 + predict_0) / len(X)
    #fitting the model based on the running gradient descent when the accuracy reached to best
    def fit(self, X, y, iter=25, lr=0.05):        
        loss = []
        weights = rand(X.shape[1])
        N = len(X)
                 
        for _ in range(iter):        
            # Gradient Descent
            y_hat = self.sigmoid(dot(X, weights))
            weights -= lr * dot(X.T,  y_hat - y) / N            
            # Saving Progress
            loss.append(self.cost_function(X, y, weights)) 
            
        self.weights = weights
        self.loss = loss
    
    def predict(self, X):        
        # Predicting with sigmoid function
        z = dot(X, self.weights)
        # Returning binary result
        return [1 if i > 0.5 else 0 for i in self.sigmoid(z)]


# In[4]:


from DataPreprocessing import * 


# **training and testing the model based on above**

# In[5]:


log_reg_model = LogisticRegression()


# In[6]:


log_reg_model.fit(x_train,y_train,iter = 1000, lr = 0.5)


# In[7]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[8]:


log_reg_y_pred = log_reg_model.predict(x_test)


# In[10]:


accuracy_score(log_reg_y_pred,y_test)


# In[ ]:





# In[ ]:




