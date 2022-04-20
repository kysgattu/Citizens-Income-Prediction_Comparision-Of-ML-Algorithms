#!/usr/bin/env python
# coding: utf-8

# In[14]:


import numpy as np
import pandas as pd
from DataPreprocessing import * 
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree as plot

#crit = ['gini', 'entropy']
#max_features = ['auto' , 'sqrt' , 'log2']

#'''
#print(' Criterion ' + ' max_features '+' Train Score ' + ' Test Score')
#for i in crit:
#    for j in max_features:
#        temp_dt = DTC(splitter = 'best', criterion = i, max_features = j)
#        temp_dt.fit(x_train,y_train)
#        temp_y_pred = temp_dt.predict(x_test)
#        train_score=temp_dt.score(x_train,y_train)
#        test_score=temp_dt.score(x_test,y_test)
#        print(i,j,train_score,test_score)
#'''
# > Using Information Gain and sqrt max_features gives slightly better accuracy

# In[5]:


gini_d_tree_model = DTC(criterion = 'gini' , splitter = 'best', max_features = 'sqrt')


# In[6]:


gini_d_tree_model.fit(x_train,y_train)


# In[7]:


gini_d_tree_y_pred = gini_d_tree_model.predict(x_test)


# In[8]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[9]:


gini_d_tree_ac = accuracy_score(gini_d_tree_y_pred,y_test)


# In[11]:


gini_d_tree_ac


# In[12]:

#The below line prints the decision tree in a python output dialog box and taking some time to loas hence commented
#import graphviz


# In[ ]:


#plot(gini_d_tree_model)


# In[ ]:


#dot_data = tree.export_graphviz(gini_d_tree_model, out_file=None) 
#graph = graphviz.Source(dot_data)
#graph.render("dt")

#graph


# In[ ]:




