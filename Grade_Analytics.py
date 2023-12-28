#!/usr/bin/env python
# coding: utf-8

# # Grade Analytics - Machine Learning Project

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ## **Data Preprocessing**

# In[2]:


df=pd.read_csv("Grades.csv")
df


# In[3]:


df.columns


# In[7]:


drop_column=df.columns[:-1][8:]
print(drop_column)
df.drop(columns=drop_column,inplace=True)
seats = df['Seat No.'].to_dict()
df.drop(columns=['Seat No.'], inplace=True)


# In[8]:


df.shape


# In[9]:


df.head()


# In[10]:


df.tail()


# In[11]:


df.describe(include='all')


# In[12]:


df.info()


# In[13]:


df.isnull().sum()


# In[14]:


for column in df.columns:
  df[column].fillna(df[column].mode()[0], inplace=True)
df.isnull().sum()


# ## **Exploratory Data Analysis**

# In[15]:


grade_list = ['A+', 'A', 'A-', 'B+', 'B', 'B-', 'C+', 'C', 'C-', 'D+', 'D', 'F', 'WU']


# ### **Countplot**

# In[16]:


for i in df.columns[:-1]:
    sns.countplot(x=i, data=df, palette='Set2', legend=False, hue=i, order=grade_list)
    plt.title(f'Countplot for {i}')
    plt.show()


# ### **Scatter/Strip Plot**

# In[17]:


sns.stripplot(data=df, x=df['PH-121'], y=df['HS-101'], color='green', order=grade_list)
plt.xlabel('PH-121')
plt.ylabel('HS-101')
plt.title('Strip Plot of PH-121 vs HS-101')
plt.show()


# In[18]:


sns.stripplot(data=df, x=df['HS-101'], y=df['CGPA'], color='green', order=grade_list)
plt.show()
sns.stripplot(data=df, x=df['CS-105'], y=df['CGPA'], color='green', order=grade_list)
plt.show()
sns.stripplot(data=df, x=df['CS-106'], y=df['CGPA'], color='green', order=grade_list)
plt.show()


# ### **Student Performance**

# In[70]:


roll = input("Enter roll number: ")
roll = [i for i in seats if seats[i]==roll]

if len(roll)==1:
    roll = roll[0]
    rolldf = df2.loc[roll][:-1]
    ax = rolldf.plot(kind='bar', color=['blue']*7+['green'])
    bx = ax.bar_label(ax.containers[0], labels=df.loc[roll][:-1].tolist())
    #cx = df2.loc[roll][-2:-1].plot(kind='bar')
else:
    print("Roll number not found.")


# ### **Box and Violin Plots**

# In[19]:


df2 = df.copy()
df2


# In[20]:


print(grade_list)
grade_mapping = {}
for i in range(len(grade_list)):
    grade_mapping[grade_list[i]] = len(grade_list)-i-1
print(grade_mapping)


# In[21]:


for col in df2.columns[:-1]:
  df2[col] = df2[col].map(grade_mapping)
df2


# In[22]:


for i in df2.columns:
    sns.boxplot(x=i, data=df2, orient='h')
    plt.show()


# In[23]:


for i in df2.columns:
    sns.violinplot(x=i, data=df2)
    plt.show()


# ### **Correlation Matrix**

# In[24]:


df2.corr()


# In[25]:


plt.figure(figsize=(6,5))
sns.heatmap(df2.corr(), annot=True, cmap='coolwarm')


# ## Clustering

# In[26]:


from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

X = df2.drop(['CGPA'], axis=1)
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

num_clusters = 3

kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init='auto')
df2['Cluster'] = kmeans.fit_predict(X)

plt.figure(figsize=(10, 6))
plt.scatter(df2['CGPA'], df2['CS-106'], c=df2['Cluster'], cmap='viridis')
plt.title('K-Means Clustering of Students')
plt.xlabel('CGPA')
plt.ylabel('CS-106')
plt.show()


# ## CGPA Prediction (Using 4 Subjects)

# In[27]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from random import shuffle

cols = df2.columns[:-2].tolist()
print(f'Available Subjects: \t\t\t {cols}')
shuffle(cols)
cols = cols[0:4]
print(f'Selected Subjects: \t\t\t {cols}')

X = df2[cols]
y = df2['CGPA']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
score = model.score(X_test, y_test)

print(f'Mean Squared Error: \t\t\t {mse}')
print(f'Coefficient of determination (R^2): \t {score}\n')

output = pd.DataFrame({'Actual CGPA': y_test, 'Predicted CGPA': y_pred})
plt.figure(figsize=(10, 6))
sns.regplot(x='Actual CGPA', y='Predicted CGPA', data=output, ci=95)
plt.title('Actual vs Predicted CGPA\n')
plt.xlabel('Actual CGPA')
plt.ylabel('Predicted CGPA')
plt.show()


# In[71]:


get_ipython().system('quarto render Grade_Analytics.ipynb')


# In[ ]:




