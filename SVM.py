#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from DataPreprocessing import * 
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
from sklearn.svm import SVC


# kernels = ['linear', 'poly', 'rbf', 'sigmoid']

# print(' Kernel ' + ' Train Score ' + ' Test Score')
# for k in kernels:
#     temp_svm = SVC(kernel = k)
#     temp_svm.fit(x_train,y_train)
#     temp_y_pred = temp_svm.predict(x_test)
#     train_score=temp_svm.score(x_train,y_train)
#     test_score=temp_svm.score(x_test,y_test)
#     print(k,train_score,test_score)

# > Polynomial Kernel has the best Scores, Hence we decide to use Polynomial Kernel for training the model

# In[4]:


svm_poly_model = SVC(kernel = 'poly')


# In[5]:


svm_poly_model.fit(x_train, y_train)


# In[6]:


svm_poly_pred = svm_poly_model.predict(x_test)


# In[7]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[10]:


svm_ac = accuracy_score(svm_poly_pred,y_test)


# In[11]:


svm_ac


# In[ ]:




