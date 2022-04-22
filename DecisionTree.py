#!/usr/bin/env python
# coding: utf-8

# ### import libraries

# In[1]:


import numpy as np
import pandas as pd
from DataPreprocessing import * 
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import plot_tree as plot


# **Creating and testing temporary models with available criteria in scikit learn and finalizing the model which has best test and train scores**

# In[23]:


#crit = ['gini', 'entropy']
#max_features = ['auto' , 'sqrt' , 'log2']


#print(' Criterion ' + ' max_features '+' Train Score ' + ' Test Score')
#for i in crit:
#    for j in max_features:
#        temp_dt = DTC(splitter = 'best', criterion = i, max_features = j)
#        temp_dt.fit(x_train,y_train)
#        temp_y_pred = temp_dt.predict(x_test)
#        train_score=temp_dt.score(x_train,y_train)
#        test_score=temp_dt.score(x_test,y_test)
#        print(i,j,train_score,test_score)


# > Using Information Gain and log2 max_features gives slightly better accuracy

# #### Train the model with Gini index with maximum feautures as log2 and test the model

# In[24]:


gini_d_tree_model = DTC(criterion = 'gini' , splitter = 'best', max_features = 'log2')


# In[25]:


gini_d_tree_model.fit(x_train,y_train)


# In[26]:


gini_d_tree_y_pred = gini_d_tree_model.predict(x_test)


# In[27]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score


# In[28]:


gini_d_tree_ac = accuracy_score(gini_d_tree_y_pred,y_test)


# In[29]:


gini_d_tree_ac


# #### ploting the decision tree
# > I have commented this steps as it takes longer time to run and is out of scope of the project and only for visualization

# In[30]:


#import graphviz


# In[31]:


#plot(gini_d_tree_model)


# In[32]:


#dot_data = tree.export_graphviz(gini_d_tree_model, out_file=None) 
#graph = graphviz.Source(dot_data)
#graph.render("dt")
#graph


# In[33]:


#graph.render("dt")


# In[34]:


#gini_d_tree_model_plot = DTC(criterion = 'gini' , splitter = 'best', max_features = 'sqrt', max_depth = 3)
#gini_d_tree_model_plot.fit(x_train,y_train)


# In[35]:


#graph_plot = graphviz.Source(tree.export_graphviz(gini_d_tree_model_plot, out_file=None))
#graph_plot.render("dt_3")


# In[ ]:




