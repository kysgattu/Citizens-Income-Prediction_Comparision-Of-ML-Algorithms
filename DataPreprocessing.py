#!/usr/bin/env python
# coding: utf-8

# ## Import Libraries

# In[1]:


import pandas as pd
import numpy as np


# ## Define Column Names

# In[2]:


names = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race','sex', 'capital-gain','capital-loss','hours-per-week','native-country','income']


# ## Import adult.data

# In[3]:


tr_data = pd.read_csv("data/adult.data",
                     names = names
                     )


# ## Import adult.data

# In[4]:


ts_data = pd.read_csv("data/adult.test",
                     names = names
                     )


# ## Combining Two datasets to 1 single dataset

# In[5]:


data = pd.concat([tr_data,ts_data])


# In[6]:


data


# ## Removing Missing Values:
# ### Missing values are present only in workclass, occupation and native-country and are denoted with '?' hence dropping rows 

# In[7]:


i = data[data['workclass'] == ' ?'].index
data = data.drop(i)
i = data[data['occupation'] == ' ?'].index
data = data.drop(i)
i = data[data['native-country'] == ' ?'].index
data = data.drop(i)


# ## fnlgwt doesnt have any effect on the income; education is just the description of education_num; relationship almost infers the marital-status;capital-gain and capital-loss have a lot of zero values which doesnt effect the prediction,,hence removing the columns

# In[9]:


data = data.drop(['fnlwgt','education','relationship','capital-gain','capital-loss'], axis = 1)


# In[10]:


data


# ## resetting indices for uniformity in visualtization

# In[11]:


data = data.reset_index(drop=True)
#data = data.drop(['index'], axis = 1)


# In[12]:


data


# ## the income values have extra . at the end making the data inconsistent, hence modifying it

# In[13]:


error = data['income'] == ' >50K.'
data.loc[error,'income'] = ' >50K'
error2 = data['income'] == ' <=50K.'
data.loc[error2,'income'] = ' <=50K'

data


# ## checking whether above function worked

# In[14]:


data.income.unique()


# ## transforming the data into continuous numeric values

# In[15]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()


# In[16]:


data['workclass']=le.fit_transform(data['workclass'])
data['marital-status']=le.fit_transform(data['marital-status'])
data['occupation']=le.fit_transform(data['occupation'])
data['race']=le.fit_transform(data['race'])
data['sex']=le.fit_transform(data['sex'])
data['native-country']=le.fit_transform(data['native-country'])
data['income']=le.fit_transform(data['income'])


# In[17]:


data


# ## Summarizing statistics of the data features

# In[18]:


data.describe()


# ## visualizing the correlation between features

# In[19]:


import matplotlib.pyplot as plt
import seaborn as sns


'''# In[20]:


corr=data.corr()
plt.figure(figsize=(15,8))
sns.heatmap(corr,annot=True,cmap='summer')
plt.title("Correlation Between the Features")
plt.show()

'''
# In[21]:


from sklearn.model_selection import train_test_split
data_train,data_test = train_test_split(data,test_size = 0.3, random_state = np.random)


# In[22]:


x_train=data_train.iloc[:,:-1] # Features
y_train=data_train.iloc[:,-1] # Target
x_train.shape,y_train.shape


# In[23]:


x_test=data_test.iloc[:,:-1] # Features
y_test=data_test.iloc[:,-1] # Target
x_test.shape,y_test.shape


# In[ ]:




