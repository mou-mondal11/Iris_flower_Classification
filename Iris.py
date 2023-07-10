#!/usr/bin/env python
# coding: utf-8

# #  Iris Flowers Classification 
# * Predict the different species of flowers on the length of there petals and sepals.

# In[53]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore")
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix


# In[3]:


data = pd.read_csv(r'C:\Users\DELL\Desktop\Data\Iris.csv')
data


# In[4]:


data.describe()


# In[5]:


data.info()


# In[8]:


data['Species'].value_counts()


# # Data Visualization

# In[14]:


Data2 = data.drop('Id', axis=1)
fig = sns.pairplot(Data2,hue='Species',markers = '*')
plt.show()


# * Here, Iris-setosa is completly different from Other two species

# In[22]:


fig2 = sns.violinplot(y='Species',x='SepalLengthCm',data=data)
plt.show()


# In[23]:


fig3 = sns.violinplot(y='Species',x='SepalWidthCm',data=data)
plt.show()


# In[24]:


fig4 = sns.violinplot(y='Species',x='PetalLengthCm',data=data)
plt.show()


# In[25]:


fig5 = sns.violinplot(y='Species',x='PetalWidthCm',data=data)
plt.show()


# # *Modelling*

# In[64]:


#splitting into Dependent and Independent Variable
x= data.iloc[:,[1,3]].values
y= data.iloc[:,-1].values


# In[67]:


encoder= LabelEncoder()
y = encoder.fit_transform(y)


# In[68]:


x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state= 0)


# In[69]:


LogisticReg = LogisticRegression()
Model_Logistic=LogisticReg.fit(x_train,y_train)
y_pred1= Model_Logistic.predict(x_test)
print('Logistic Regression Confusion Matrix:\n ',confusion_matrix(y_test,y_pred1))
print('Logistic Regression Accuracy score: ',accuracy_score(y_test,y_pred1)*100)


# In[70]:


DecisionTree = DecisionTreeClassifier()
Model_DecisionTree = DecisionTree.fit(x_train,y_train)
y_pred2 = Model_DecisionTree.predict(x_test)
print('Decision Tree Classifier Confusion Matrix:\n ',confusion_matrix(y_test,y_pred2))
print('Decision Tree Classifier Accuracy score: ',accuracy_score(y_test,y_pred2)*100)


# In[71]:


modelRandomForestClassifier = RandomForestClassifier(criterion='gini', max_depth=2)
modelRandomForestClassifier.fit(x_train, y_train)
y_pred3 = modelRandomForestClassifier.predict(x_test)
print('Random Forest Classifier Confusion Matrix:\n ',confusion_matrix(y_test,y_pred3))
print('Random Forest Classifier Accuracy score: ',accuracy_score(y_test,y_pred3)*100)


# In[72]:


modelSVC = SVC(kernel='sigmoid')
modelSVC.fit(x_train, y_train)
y_pred4 = modelSVC.predict(x_test)
print('SVC Confusion Matrix:\n ',confusion_matrix(y_test,y_pred4))
print('SVC Accuracy score: ',accuracy_score(y_test,y_pred4)*100)


# # Accuracy Score of all the models:
# * Logistic Regression: 96.67%
# * Decision Tree Classifier: 93.33%
# * Random Forest Classifier: 90%
# * Support Vector Classifier: 20%

# # Conclusion

# * The Support Vector Classifer is having an accuracy of 20% so there's a scenario of under-fitting
# * The Logistic Regression,Decision Tree Classifer and The Random Forest Classifier performs the best

# In[79]:


encoder.inverse_transform(Model_Logistic.predict([[6, 4]]))


# In[80]:


encoder.inverse_transform(Model_DecisionTree.predict([[6, 4]]))


# In[81]:


encoder.inverse_transform(modelRandomForestClassifier.predict([[6, 4]]))


# In[82]:


encoder.inverse_transform(Model_Logistic.predict([[1, 5]]))


# In[83]:


encoder.inverse_transform(Model_DecisionTree.predict([[1,5]]))


# In[84]:


encoder.inverse_transform(Model_DecisionTree.predict([[1,5]]))


# # --------------------------------------------------------------------------------------------------

# In[ ]:




