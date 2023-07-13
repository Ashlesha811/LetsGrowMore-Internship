#!/usr/bin/env python
# coding: utf-8

# #                                     LETSGROWMORE INTERNSHIP 
# ##                                                              TASK 1
# ###                                                        IRIS FLOWER CLASSIFICATION
# 

# In[1]:


# Import Packages
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df = pd.read_csv("iris_csv.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.describe()


# In[5]:


df['class'].value_counts()


# In[6]:


df.shape


# In[7]:


data=df.replace(to_replace={'class':{'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2}})
data.head()


# In[8]:


data.isnull().sum()


# In[9]:


data['class'] = data['class'].astype('category').cat.codes


# In[10]:


sns.countplot(y=data['class'],data=data)
plt.ylabel('target classes')
plt.xlabel('count of each Target class')
plt.show()


# In[11]:


data['sepallength'].hist()


# In[12]:


df['sepalwidth'].hist()


# In[13]:


df['petallength'].hist()


# In[14]:


data['petalwidth'].hist()


# In[15]:


df.corr()


# In[16]:


corr = df.corr()
fig, ax = plt.subplots(figsize = (9,11))
sns.heatmap(corr, cmap="Greens", annot = True, ax = ax)


# In[17]:


sns.pairplot(df.iloc[:,:],hue='class')


# In[18]:


X = data.drop(['class'], axis=1)
Y = data['class']


# In[19]:


from sklearn.model_selection import train_test_split
from sklearn import metrics


# In[20]:


X_train, X_test,Y_train, Y_test = train_test_split(X ,Y ,test_size = 0.30, random_state=0)
print("Train Shape",X_train.shape)
print("Test Shape",X_test.shape)


# In[21]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()


# In[22]:


model.fit(X_train, Y_train)


# In[23]:


print("Accuracy: ",model.score(X_test, Y_test)*100)


# In[24]:


predicted=model.predict(X_test)
predicted


# In[25]:


print('The accuracy of the model is',metrics.accuracy_score(predicted,Y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




