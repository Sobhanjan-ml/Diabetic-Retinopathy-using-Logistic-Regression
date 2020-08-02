#!/usr/bin/env python
# coding: utf-8

# # Importing Liabraries

# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# # Importing Dataset

# In[39]:


dataset = pd.read_csv('Diabetic-Retinopathy.csv')


# In[40]:


dataset.head()

status-
0 = no
1 = yes
# # Checking for null-values and Data types

# In[41]:


dataset.isnull().any()


# In[42]:


dataset.dtypes


# # Checking Information 

# In[43]:


dataset.info()


# In[44]:


dataset.describe()


# # Importing Label Encoder to treat Categorical values

# In[45]:


from sklearn.preprocessing import LabelEncoder


# In[46]:


lb = LabelEncoder()


# In[47]:


dataset['eye'].value_counts()


# In[48]:


dataset['eye'] = lb.fit_transform(dataset['eye'])


# In[49]:


dataset['eye'].value_counts()

right = 1
left = 0
# In[50]:


dataset['laser'].value_counts()


# In[51]:


dataset['laser'] = lb.fit_transform(dataset['laser'])


# In[52]:


dataset['laser'].value_counts()

xenon = 1
argon = 0
# In[53]:


dataset


# # Checking Correlation

# In[54]:


dataset.corr()


# In[55]:


sns.heatmap(dataset.corr(),annot=True)


# # Data Visualization

# Count-Plot of Status 

# In[77]:


sns.set_style('whitegrid')
sns.countplot(dataset['status'])


# Count-Plot of Status with respect to risk

# In[78]:


sns.countplot(x='status',hue='risk',data=dataset)


# Count-Plot of Status with respect to eye

# In[79]:


sns.countplot(x='status',hue='eye',data=dataset)


# Count-Plot of Status with respect to laser

# In[80]:


sns.countplot(x='status',hue='laser',data=dataset)


# Count-Plot of Status with respect to trt

# In[81]:


sns.countplot(x='status',hue='trt',data=dataset)


# Age variations

# In[88]:


sns.set()

plt.hist(x='age',data=dataset)
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# In[56]:


x = dataset.iloc[:,2:8].values


# In[57]:


x


# In[58]:


y = dataset.iloc[:,8].values


# In[59]:


y


# Training and Testing Splits

# In[60]:


from sklearn.model_selection import train_test_split


# In[61]:


x_train , x_test , y_train , y_test = train_test_split(x,y,test_size=0.25,random_state=0)


# In[62]:


x_train


# In[63]:


x_test


# In[64]:


y_train


# In[65]:


y_test


# Importing Algorithm

# In[66]:


from sklearn.linear_model import LogisticRegression


# In[67]:


lr = LogisticRegression()


# In[68]:


lr.fit(x_train,y_train)


# In[69]:


y_pred = lr.predict(x_test)
y_pred


# Checking Accuracy of our model

# In[70]:


from sklearn.metrics import accuracy_score


# In[71]:


accuracy_score(y_test,y_pred)*100


# Testing Model

# In[72]:


lr.predict([[1,13,0,0,10,0.30]])


# In[73]:


lr.predict([[1,22,1,8,0,46.23]])

