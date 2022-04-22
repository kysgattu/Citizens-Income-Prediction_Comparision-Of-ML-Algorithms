#!/usr/bin/env python
# coding: utf-8

# # Visualizing the Data Median Points using Seaborn BoxPLot

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn


# In[2]:


from DataPreprocessing import *


# In[3]:


data


# In[4]:


sns.boxplot(x = 'age', data = data, palette="vlag")
plt.title('Age')
plt.show()


# In[5]:


sns.boxplot(x = 'education-num', data = data,palette="vlag")
plt.title('Education-Number')
plt.show()


# In[6]:


sns.boxplot(x= 'sex', data = data, palette="vlag")
plt.title('Sex')
plt.show()


# In[7]:


sns.boxplot(x = 'hours-per-week', data = data, palette="vlag")
plt.title('Hours per Week')
plt.show()


# In[ ]:




