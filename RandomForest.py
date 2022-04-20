#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from DataPreprocessing import * 
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier as RFC


# crit = ['gini', 'entropy']
# max_features = ['auto' , 'sqrt' , 'log2']

# print(' Criterion ' + ' max_features '+' Train Score ' + ' Test Score')
# for i in crit:
#     for j in max_features:
#         temp_rf = RFC(n_estimators = 100, criterion = i, max_features = j)
#         temp_rf.fit(x_train,y_train)
#         temp_y_pred = temp_rf.predict(x_test)
#         train_score=temp_rf.score(x_train,y_train)
#         test_score=temp_rf.score(x_test,y_test)
#         print(i,j,train_score,test_score)

# > Accuracy is best with Gini Index and log2 max_features

# In[6]:


gini_rd_frst = RFC(n_estimators = 100, criterion = 'gini', max_features = 'log2')


# In[7]:


gini_rd_frst.fit(x_train, y_train)


# In[11]:


gini_rd_frst_y_pred = gini_rd_frst.predict(x_test)


# In[9]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[14]:


gini_rd_frst_ac = accuracy_score(gini_rd_frst_y_pred,y_test)


# In[16]:


gini_rd_frst_ac


# In[ ]:




