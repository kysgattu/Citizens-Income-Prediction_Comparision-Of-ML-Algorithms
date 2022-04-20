#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


from DataPreprocessing import * 


# In[3]:


from LogisticRegression import log_reg_y_pred  as lr
from KNN import knn_pred as knn
from NaiveBayes import nb_predictions as nb
from SVM import svm_poly_pred as svm
from DecisionTree import gini_d_tree_y_pred as dt
from RandomForest import gini_rd_frst_y_pred as rdf


# In[4]:


from sklearn.metrics import *


# In[5]:


cm_log = confusion_matrix(y_test,lr)
cm_knn = confusion_matrix(y_test,knn)
cm_nb = confusion_matrix(y_test,nb)
cm_svm = confusion_matrix(y_test,svm)
cm_dt = confusion_matrix(y_test,dt)
cm_rf = confusion_matrix(y_test,rdf)


# In[6]:


fig=plt.figure(figsize=(18,18))

plt.subplot(3,2,1)
sns.heatmap(cm_log,annot=True, fmt=".1f", cmap='summer')
plt.title('Logistic Regression')

plt.subplot(3,2,2)
sns.heatmap(cm_knn,annot=True, fmt=".1f", cmap='summer')
plt.title('K Nearest Neighbor ')

plt.subplot(3,2,3)
sns.heatmap(cm_nb,annot=True, fmt=".1f", cmap='summer')
plt.title('Naive Bayes ')

plt.subplot(3,2,4)
sns.heatmap(cm_svm,annot=True, fmt=".1f", cmap='summer')
plt.title('Support Vector Machine')

plt.subplot(3,2,5)
sns.heatmap(cm_dt,annot=True, fmt=".1f", cmap='summer' )
plt.title('Decision Tree')

plt.subplot(3,2,6)
sns.heatmap(cm_rf,annot=True, fmt=".1f", cmap='summer')
plt.title('Random Forest Tree')
plt.show()


# In[7]:


ac_log = accuracy_score(y_test,lr)
ac_knn = accuracy_score(y_test,knn)
ac_nb = accuracy_score(y_test,nb)
ac_svm = accuracy_score(y_test,svm)
ac_dt = accuracy_score(y_test,dt)
ac_rf = accuracy_score(y_test,rdf)


# In[8]:


algorithms = ['Logistic Regression' ,
              'K-Nearest Neighbors',
              'Naive Bayes',
              'Support Vector Machine',
             'Decision Tree',
             'Random Forest Classifier']

scores = [ac_log, ac_knn, ac_nb, ac_svm, ac_dt, ac_rf]


# In[9]:


max_y_lim = max(scores) + 0.05
min_y_lim = min(scores) - 0.05


# In[10]:


fig=plt.figure(figsize=(12,8))
plt.ylim(min_y_lim, max_y_lim)
bars =plt.bar(algorithms, scores)
plt.bar_label(bars)
plt.xlabel("Algorithms")
plt.ylabel('Accuracy score')
plt.title('Categories Bar Plot')
plt.show()


# In[11]:


cr_log = classification_report(y_test,lr)
cr_knn = classification_report(y_test,knn)
cr_nb = classification_report(y_test,nb)
cr_svm = classification_report(y_test,svm)
cr_dt = classification_report(y_test,dt)
cr_rf = classification_report(y_test,rdf)


# In[12]:


print("*"*20+'Logistic Regression'+"*"*20)
print(cr_log)

print("*"*20+'K Nearest Neighbor'+"*"*20)
print(cr_knn)

print("*"*20+'Naive Bayes'+"*"*20)
print(cr_nb)

print("*"*20+'Support Vector Machine'+"*"*20)
print(cr_svm)

print("*"*20+'Decision tree'+"*"*20)
print(cr_dt)

print("*"*20+'Random Forest'+"*"*20)
print(cr_rf)

