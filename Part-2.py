#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# In[2]:


data=pd.read_csv('D:/netflix_titles.csv')
data.head()


# In[3]:


data.dropna(axis=0,how='any', inplace=True)
data.shape


# In[4]:


data.isnull().sum()


# In[5]:


data['new_date']=pd.to_datetime(data.date_added)
data.head()


# ## Linear Regression 

# In[6]:


movie=data[data.type=='Movie']
movie.head()


# In[7]:


movie.shape


# In[8]:


movie.sort_values(by='new_date',ascending=False)
movie.head()


# In[9]:


df=[[movie.release_year.unique()],[movie.release_year.value_counts()]]


# In[10]:


df


# In[15]:


plt.scatter(x=[movie.release_year.unique()],y=[movie.release_year.value_counts()])
plt.xlabel('Year')
plt.ylabel('Number of productions in each year')
plt.show()


# In[16]:


x=movie.release_year.unique()
y=movie.release_year.value_counts()
x_train, x_test, y_train, y_test= train_test_split(x, y,train_size=0.50,test_size=0.50,random_state=0)


# In[17]:


plt.scatter(x_train,y_train, label='Training Data',color='r')
plt.scatter(x_test,y_test, label='Testing Data',color='b')
plt.legend()
plt.show()


# In[18]:


reg=LinearRegression()
reg.fit(x_train.reshape(-1,1),y_train)


# In[19]:


predict=reg.predict(x_test.reshape(-1,1))
plt.plot(x_test,predict,label='LR',color='g')
plt.scatter(x_test,y_test,label='Actual data',color='b')
plt.legend()
plt.show()


# In[22]:


reg.predict(np.array([[2023]]))[0]


# In[ ]:




