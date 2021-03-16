#!/usr/bin/env python
# coding: utf-8

# # Author : Renuka Chadalawada

# # Task1: Prediction using Supervised ML
#     Predict the percentage of an student based on the no. of study hours.

# ### Import Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# ### Read Dataset in Pandas dataframe

# In[2]:


dataset=pd.read_csv(r"D:\Data sciece SP project\scores.csv")


# In[3]:


dataset


# ### Check the if null values

# In[4]:


dataset.isnull().sum()


# ### Undersatnding the data and data types

# In[5]:


dataset.info()


# In[6]:


dataset.shape


# In[7]:


dataset.describe()


# ### Visualization on outliers if any

# In[8]:


sns.boxplot(dataset["Scores"])


# In[9]:


sns.boxplot(dataset["Hours"])


# ### Finding the relationship between the variables

# In[10]:


dataset.corr()


# In[11]:


sns.heatmap(dataset.corr(),annot=True)


# In[12]:


sns.scatterplot(dataset["Hours"],dataset["Scores"])


# ### Observations
# 
# We can say that there is strongly positive linear relationship between the variables

# ## Split data into dependent and independent Variables

# In[13]:


x=dataset.iloc[:,0:1]# Hours


# In[14]:


y=dataset.iloc[:,1]# Scores


# ### Split data into train and test

# In[15]:


from sklearn.model_selection import train_test_split


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=12)


# In[17]:


x_train.shape


# In[18]:


x_test.shape


# ### Build the Model

# #### Train

# In[19]:


from sklearn.linear_model import LinearRegression


# In[20]:


# creating an empty model or instance
score_model=LinearRegression()


# In[21]:


# fit --trains a model
score_model.fit(x_train,y_train)


# ### Test your Model

# In[22]:


y_pred=score_model.predict(x_test)


# In[23]:


y_pred


# In[24]:


y_test


# In[25]:


score_model.intercept_


# In[26]:


score_model.coef_


# ### Observations
# 
# 1. A fresher can expect a salary about 25k
# 2. On an average an employee can expect 9k increment every year

# ### Plot Best Fit Line

# In[27]:


y_pred_train=score_model.predict(x_train)


# In[28]:


y_pred_train


# In[29]:


plt.scatter(x_train,y_train, label="Actual values")
plt.plot(x_train,y_pred_train,label="Best fit line",color="red")
plt.xlabel("Hours of Study")
plt.ylabel("scores obtained")
plt.title("Actual vs Predicted")
plt.legend()
plt.show()


# In[30]:


plt.scatter(x_test,y_test, label="Actual values")
plt.plot(x_test,y_pred,label="Best fit line",color="red")
plt.xlabel("Hours of Study")
plt.ylabel("scores obtained")
plt.title("Test Actual vs Predicted")
plt.legend()
plt.show()


# ### Evaluation

# In[31]:


from sklearn.metrics import mean_squared_error


# In[32]:


mse=mean_squared_error(y_test,y_pred)


# In[33]:


mse


# In[34]:


rmse=np.sqrt(mse)


# In[35]:


rmse


# In[36]:


from sklearn.metrics import r2_score


# In[37]:


r2_score(y_test,y_pred)


# ### Real time predictions

# ### What will be predicted score if a student studies for 9.25 hrs/ day?

# In[38]:


hrs=9.25


# In[39]:


hrs=np.array([[hrs]])


# In[40]:


score_model.predict(hrs)


# # Conclusion

# ###### predicted score of a student who studies for 9.25 hrs/ day will be eequal to 92.2%
