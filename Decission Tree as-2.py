#!/usr/bin/env python
# coding: utf-8

# In[102]:


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 
get_ipython().run_line_magic('matplotlib', 'inline')
from warnings import filterwarnings
filterwarnings('ignore')


# # ANS 1

# In[103]:


df=pd.read_csv('diabetes.csv')


# In[ ]:





# In[104]:


df.head()


# In[105]:


# check the missing value 
df.isnull().sum()


# In[106]:


# check the duplicates value 
df.duplicated().sum()


# In[23]:


# there are no duplicates value 


# In[107]:


# check datatype 
df.info()


# In[108]:


# checking the number of uniques value of each value 
df.nunique()


# In[167]:


## check the statitics of the dataset
df.describe()


# In[29]:


df[['Age','Outcome']]=df[['Age','Outcome']].astype(int)


# In[110]:


df.info()


# In[111]:


df.head()


# In[33]:


### Visualization


# In[112]:


sns.heatmap(df.corr(),annot=True)


# In[36]:


# Insights 
# pregnancies has positve realtioship with age
# preganancies has negativ reattionsip with skinthickness,insulin,bmi,diabatiespregurefucntion


# In[43]:


plt.style.use('seaborn')
df.hist(bins=50,ec='b',figsize=(20,15))
plt.show()


# In[44]:


df.head()


# In[48]:


sns.barplot(x='Pregnancies',y='Age',data=df,)


# In[50]:


#insights
#highest number of pregnant women are between age 40 ane 50


# In[52]:


percentage=df.Outcome.value_counts(normalize=True)*100
percentage


# In[55]:


classlabel=['non-diabatic','diabatic']
plt.figure(figsize=(10,7))
plt.pie(percentage,labels=classlabel,autopct='%1.1f%%')
plt.title('pie of chart of outcome',fontsize=15)
plt.show()


# In[76]:


# inshights 
# non daibatic patients are less compared to diabatic


# In[60]:


int_feature=[feature for feature in df.columns if df[feature].dtype=='int64']


# In[61]:


numerical_feature=df[]


# In[68]:


numerical_feture=[feature for feature in df.columns if df[feature].astype!='O'] 
    


# In[71]:


numerical_feature=numerical_feture[0:-1]


# In[74]:


for feature in numerical_feature:
    sns.histplot(data=df,x=feature,hue='Outcome')
    plt.title(feature)
    plt.show()


# In[75]:


# diabatic people have a glucose between 150 and 200
# high number of pregnant women are not diabatic
# diabatic people have a diabticesPedigreeFunction between 0.5 and 0.0


# # ANS 2

# In[168]:


dftemp=df.drop(['Outcome'],axis=1)
plt.figure(figsize=(12,6))
dftemp.boxplot()
plt.show()


# In[82]:


# Insights there are many outliers in Insulin feauture


# In[125]:


minima,Q1,median,Q3,maxima=np.quantile(df['Insulin'],[0,0.25,0.50,0.75,1.0])


# In[126]:


IQR=Q3-Q1
lower_fence=Q1-1.5*(IQR)
higher_fence=Q3+1.5*(IQR)


# In[121]:


l=[]

for i in df['Insulin']:
    if i>=lower_fence and i<=higher_fence:
        l.append(i)
    else:
        df['Insulin']=df['Insulin'].replace(i,df['Insulin'].mean())
        


# In[123]:


l=[]
l2=[]
for i in df['Insulin']:
    if i>=lower_fence and i<=higher_fence:
        l.append(i)
    else:
        l2.append(i)


# In[ ]:





# In[128]:


# insights we replace outlier with mean value 


# # ANS 3

# In[129]:


from sklearn.model_selection import train_test_split


# In[133]:


X=df.iloc[:,0:8]
y=df.iloc[:,-1]


# In[135]:


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.33,random_state=42)


# # ANS 4

# In[137]:


from sklearn.tree import DecisionTreeClassifier


# In[138]:


parameter={
 'criterion':['gini','entropy','log_loss'],
  'splitter':['best','random'],
  'max_depth':[1,2,3,4,5],
  'max_features':['auto', 'sqrt', 'log2']
    
}


# In[139]:


from sklearn.model_selection import GridSearchCV


# In[140]:


treeclassifier=DecisionTreeClassifier()
clf=GridSearchCV(treeclassifier,param_grid=parameter,cv=5,scoring='accuracy')


# In[142]:


clf.fit(X_train,y_train)


# In[163]:


clf.best_estimator_


# # ANS 7

# In[143]:


y_pred=clf.predict(X_test)


# In[158]:


treeclassifier=DecisionTreeClassifier(criterion='gini',max_depth=2)


# In[159]:


treeclassifier.fit(X_train,y_train)


# In[160]:


y_pred1=treeclassifier.predict(X_test)


# In[ ]:





# In[ ]:





# In[ ]:





# # ANS 5

# In[146]:


from sklearn.metrics import accuracy_score,classification_report


# In[165]:


score=accuracy_score(y_pred1,y_test)
print(score)
print(classification_report(y_pred,y_test))


# # ANS 6

# In[171]:


from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(treeclassifier,filled=True)


# In[1]:


# Insights
#1) BMI,Glucose and age are the important faeture
#2) thrushold value for glucose,BMI,age is 154.5,28.7,30.5
#3) there are 3 branch node 
#4) there are 4 leaves 
#5) gini impurity is used for to check the impuruty of the data and 
# for the further divison


# In[ ]:




