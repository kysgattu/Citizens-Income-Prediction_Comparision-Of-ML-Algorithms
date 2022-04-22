#!/usr/bin/env python
# coding: utf-8

# ## Defining Naive Bayes Algorithm

# ##### import libraries

# In[1]:


import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Define logic for Naive Bayes Classifier
class NaiveBayesClassifier():

    def Prior_Probability(self, features, target):
        #Calculate the prior probability
        #get the count of respective class and apply a function to count.
        self.prior_prob = (features.groupby(target).apply(lambda x: len(x)) / self.rows).to_numpy() 
        return self.prior_prob
    
    def statistics(self, features, target):
        # Calculating mean and variance of each column and parse the data to an array. 
        
        self.mean = features.groupby(target).apply(np.mean).to_numpy()           #Calculate mean of the class
        self.variance = features.groupby(target).apply(np.var).to_numpy()        #Calculate variance of the class
    
        return self.mean, self.variance
    
    def Probability(self, label, x):     
        
        # calculating probability from gaussian density function assuming probability of specific target value of given specific class is normally distributed 
        mean = self.mean[label]
        variance = self.variance[label]
        probability = (np.exp((-1/2)*((x-mean)**2) / (2 * variance)))/(np.sqrt(2 * np.pi * variance))
        return probability
    
    def Posterior_Probability(self, x):
        # calculate the posterior probability 
        posteriors = []
        for i in range(self.count):
            prior = np.log(self.prior_prob[i])                             
            conditional = np.sum(np.log(self.Probability(i, x)))     
            posteriors.append( prior + conditional)
        return self.classes[np.argmax(posteriors)]                   # return class with highest posterior probability
     

    def fit(self, features, target):
        self.classes = np.unique(target)
        self.count = len(self.classes)
        self.rows = features.shape[0]
        self.statistics(features, target)
        self.Prior_Probability(features, target)
        
    def predict(self, test_data):
        predictions = [self.Posterior_Probability(f) for f in test_data.to_numpy()]
        return predictions


# In[3]:


from DataPreprocessing import *


# #### train and test model based on above algorithm

# In[4]:


nb_model = NaiveBayesClassifier()
nb_model.fit(x_train,y_train)


# In[5]:


nb_predictions = nb_model.predict(x_test)


# In[6]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[8]:


accuracy_score(nb_predictions,y_test)


# In[ ]:




