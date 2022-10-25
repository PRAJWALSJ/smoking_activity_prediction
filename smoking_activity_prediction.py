#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Importing required libraries and modules
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns


# In[2]:


#Loading the dataset
data = pd.read_csv('/Users/prajwalsj/Downloads/smoking.csv')


# In[3]:


data.head()


# In[4]:


#Checking the total number of rows and columns of the dataset 
print('There are',data.shape[0] ,' rows and', data.shape[1],' columns')


# In[5]:


#We can check if the each variable is either numerical or categorical
data.info()


# In[6]:


#Checking for null values
data.isnull().sum().any()


# In[7]:


#Checking for duplicated values
data.duplicated(subset=['ID']).sum()


# In[8]:


#Dropping the ID as it is irrelevent
data.drop("ID", axis = 1, inplace = True)


# In[9]:


data.head()


# In[10]:


data.describe()


# # Data Analysis/EDA

# In[11]:


#Plotting the count graph for total number of smokers vs total number of non-smokers
#Note: The term smoker indicates that the smoking activity has been observed in the body irrespective of active smoking or passive smoking
plt.figure(figsize=(8,6))
chart = sns.countplot(data['smoking'])
plt.show()


# In[12]:


#Plotting pie chart for percentage of smokers vs non smokers
data['smoking'].value_counts().plot(kind='pie',autopct='%1.2f%%')
plt.title("Data Distribution Pie Chart")
plt.show()


# In[13]:


#Plotting the count graph for total number of males vs females
plt.figure(figsize=(8,6))
chart = sns.countplot(data['gender'])
plt.show()


# In[14]:


#Plotting the count graph for number of female smokers and female non-smokers vs number of male smokers and male non-smoker
plt.figure(figsize = (8,6))
sns.countplot(x="gender", hue="smoking", data = data)
plt.show()


# In[15]:


#Plotting distribution plot for age to check the density
sns.set_style('whitegrid')
plt.figure(figsize = (10,6))
sns.distplot(data['age'],bins=10)
plt.show()


# In[16]:


#Checking the min, max and mean of the age
data['age'].describe()


# In[17]:


#Plotting the count graph of smokers vs non smokers with respect to age group
plt.figure(figsize = (10,6))
sns.countplot(x="age", hue="smoking", data = data)
plt.show()


# In[18]:


#Plotting the distribution graph for height
plt.figure(figsize = (10,6))
sns.distplot(data['height(cm)'],bins=10)
plt.show()


# In[19]:


#Checking the min, max and mean of the height
data['height(cm)'].describe()


# In[20]:


#Plotting the count graph of smokers vs non smokers with respect to height group
plt.figure(figsize = (10,6))
sns.countplot(x="height(cm)", hue="smoking", data = data)
plt.show()


# In[21]:


#Plotting the distribution graph for weights
plt.figure(figsize = (10,6))
sns.distplot(data['weight(kg)'],bins=10)
plt.show()


# In[22]:


#Checking the min, max and mean of the weight
data['weight(kg)'].describe()


# In[23]:


#Plotting the count graph of smokers vs non smokers with respect to weight group
plt.figure(figsize = (10,6))
sns.countplot(x="weight(kg)", hue="smoking", data = data)
plt.show()


# In[24]:


#Plotting the distribution graph for systolic
#Systolic blood pressure measures the pressure in your arteries when your heart beats
#Normal systolic blood pressure being less than 120mm Hg
plt.figure(figsize = (10,6))
sns.distplot(data['systolic'],bins=10)
plt.show()


# In[25]:


#Checking the min, max and mean of the systolic
data['systolic'].describe()


# In[26]:


#checking the number of smokers with systolic blood pressure greater than 120
x = data.loc[
    data.systolic.gt(120) & data.smoking.eq(1)
]
print(len(x))


# In[27]:


#checking the number of non smokers with systolic blood pressure greater than 120
y = data.loc[
    data.systolic.gt(120) & data.smoking.eq(0)
]
print(len(y))


# In[28]:


#Plotting the distribution graph for relaxation variable
plt.figure(figsize = (10,6))
sns.distplot(data['relaxation'],bins=10)
plt.show()


# In[29]:


#Checking the min, max and mean of the parameter relaxation
data['relaxation'].describe()


# In[30]:


#Plotting the distribution graph for fasting blood sugar parameter/feature
plt.figure(figsize = (10,6))
sns.distplot(data['fasting blood sugar'],bins=10)
plt.show()


# In[31]:


#Checking the min, max and mean of the parameter fasting blood sugar
data['fasting blood sugar'].describe()


# In[32]:


#Plotting the distribution graph for cholesterol variable
plt.figure(figsize = (10,6))
sns.distplot(data['Cholesterol'],bins=10)
plt.show()


# In[33]:


#Checking the min, max and mean of the cholesterol
data['Cholesterol'].describe()


# In[34]:


#Plotting the distribution graph for hemoglobin variable
plt.figure(figsize = (10,6))
sns.distplot(data['hemoglobin'],bins=10)
plt.show()


# In[35]:


#Checking the min, max and mean of the hemoglobin
data['hemoglobin'].describe()


# In[36]:


#plot to check how hemoglobin levels changes for smokers vs non smokers
plt.figure(figsize = (10,6))
sns.countplot(x="hemoglobin", hue="smoking", data = data)
plt.show()


# In[37]:


#count plot for the parameter oral
plt.figure(figsize=(8,6))
chart = sns.countplot(data['oral'])
plt.show()


# In[38]:


#checking the unique values of parameter oral
data['oral'].unique()


# In[39]:


#Since category oral has one one value for all the rows, it is relevent to drop this feature
data.drop("oral", axis = 1, inplace = True)


# In[40]:


#Plotting the countplot for tartar
plt.figure(figsize=(8,6))
chart = sns.countplot(data['tartar'])
plt.show()


# In[41]:


#Plotting the count graph of smokers vs non smokers with respect to weight group
plt.figure(figsize = (8,6))
sns.countplot(x="tartar", hue="smoking", data = data)
plt.show()


# In[42]:


#plotting the heatmap of correlation matrix which describes the correlation trends among different attributes
plt.figure(figsize = (18,15))
plt.title('correlation matrix')
sns.heatmap(data.corr(), annot=True)
plt.show()


# In[43]:


# Dropping the highly correlated features from the dataset
data = data.drop(['LDL', 'weight(kg)', 'hemoglobin', 'ALT'], axis = 1)


# In[44]:


#Plotting pairplot to check if the data points are overlapped
sns.pairplot(data.sample(1000), hue='smoking')


# In[45]:


data.head()


# In[46]:


#Function to remove the outliers using IQR method
from collections import Counter
def outlier_detection(data, n, columns):
    rows = []
    will_drop_train = []
    for col in columns:
        Q1 = np.nanpercentile(data[col], 25)
        Q3 = np.nanpercentile(data[col], 75)
        IQR = Q3 - Q1
        outlier_point = 1.5 * IQR
        rows.extend(data[(data[col] < Q1 - outlier_point)|(data[col] > Q3 + outlier_point)].index)
    for r, c in Counter(rows).items():
        if c >= n: will_drop_train.append(r)
    return will_drop_train

will_drop_train = outlier_detection(data, 5, data.select_dtypes(["float", "int"]).columns)
will_drop_train[0:5]


# In[47]:


#Dropping the outliers detected using IQR method
data.drop(will_drop_train, inplace = True, axis = 0)


# In[48]:


data


# # Observations after EDA

# Observations after doing EDA
# 
# NOTE: The dataset has two categories of target feature "smoking"(label 0- population where no smoking signals were found, label 1- population where smoking signals were found). We will simply refer the population of label 0 as non smokers and population of label 1 as smokers(irrespective of the reasons like active smoking or passive smoking etc). 
# 
# - In the dataset, 63.27% of population are non smokers and 36.73% are smokers.
# - Number of male smokers are more than the number of female smokers. A very small population of females has smoking   signals in their bodies.
# - The dataset has the majority of the population belonging to the age groups between 40-60. 
# - Most smokers were found in the age group of 40 and the age 35 has the highest percentage of smokers
# - The population is well distributed across the people with a height between 150 and 180 cms.
# - People over the height of 170 cm, there is a high chance that people are more likely to have smoking signals in their bodies.
# - The population is well distributed across the people with a height between 40 and 90 kgs.
# - People over weight 75 kgs, there is a high chance that people are more likely to have smoking signals in their bodies.
# - Normal systolic blood pressure(Systolic blood pressure measures the pressure in your arteries when your heart beats) being less than 120mm Hg, it is oberved that the number of non smokers with abnormal systolic blood pressure(>120) i.e are higher than the number of smokers with abnormal systolic blood pressure. This unexpected observation may be due to randomness or might imply that smoking does not affect systolic blood pressure.
# - Hemoglobin levels are higher in smokers compared to that of non-smokers indicating a high hemoglobin count that occurs most commonly when your body requires an increased oxygen-carrying capacity.
# - The population of people with tartar is marginally greater than the population of people without tartar and most percentages of smoking signals were found in people with tartar.
# - The 4 features are correlated and hence dropped from the dataset.
# - Height, Weight and hemoglobin affect the class "smoking" the most.
# - From observing the sample pairplot, the datapoints are highly overlapped, thus linear classification algorithms like Logistic regression and Support Vector Machine cannot be used for model building. Hence we will use KNN algorithm based on the principle of euclidean distance or Decision Tree or Random Forest algorithms which works on the principle of non-linear classification.
# - Outliers are detected and removed using IQR(Inter-Quartile Range) method.
# 
# 
# 

# # Data Pre-Processing

# - Categorical variables are transformed to the numeric representation using label encoder that assigns the numeric label to each category in a column.
# - The balancing of the data was performed using an oversampling technique called SMOTE which increases the number of data points of minority class to match the number of data points in majority class. 
# - The dataset is split into a training dataset that has 75% of the whole data and the rest 25% were included in the test dataset.
# 

# In[49]:


#Checking the data types of all the features
data.dtypes


# In[50]:


#since features gender and tartar are categorical values, we will encode them using LaberEncoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data['gender'] = le.fit_transform(data['gender'])
data['tartar'] = le.fit_transform(data['tartar'])


# In[51]:


data.head()


# In[52]:


#we will load all the features into f and target variable to c
f = data.drop(columns = 'smoking', axis = 1)
c = data['smoking']


# In[53]:


#Since the dataset target variable is highly imbalanced(63.27% non smokers vs 36.73 smokers), we will balance the target variable using resampling technique SMOTE
from imblearn.over_sampling import SMOTE
smote  = SMOTE()
f, c = smote.fit_resample(f, c)


# In[54]:


#we can observe that total number of smokers and non smokers are equal and now the dataset is balanced
c.value_counts()


# In[55]:


#shape of the dataset after sampling
f.shape


# In[56]:


#splitting the dataset into train and test dataset using sklearn
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(f, c, test_size=0.25)


# In[57]:


#shape of train dataset
x_train.shape


# In[58]:


#shape of test dataset
x_test.shape


# # Machine Learning Algorithms

# In[59]:


#importing the required libraries/modules
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix


# # K Nearest Neighbors(KNN)

# In[60]:


#We will use a for loop to check the ideal value of k for knn algorithm 
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

k_range = range(1,26)
scores = {}
scores_list = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    scores[k] = metrics.accuracy_score(y_test, y_pred)
    scores_list.append(metrics.accuracy_score(y_test, y_pred))


# In[61]:


#plotting a graph for values of k vs test accuracy, so that we can choose ideal k value
plt.plot(k_range, scores_list)
plt.xlabel('value of k for KNN')
plt.ylabel('testing accuracy')


# In[62]:


#We will train the algorithm using train data and check the accuracy for test data
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
acc = accuracy_score(y_test, pred)
print('Test accuracy for n_neighbors = 5 is %f%%' % (acc))


# In[63]:


#Printing the accuracy, precision, recall and f1-score
print(classification_report(y_test, pred))


# In[64]:


#Plotting the confusion matrix to check number of TP,TF,FP and FN
plot_confusion_matrix(knn, x_test, y_test)
plt.grid(False)
plt.show()


# # Random Forest

# In[65]:


#importing random forest classifier from sklearn and checking the ideal value of n by iterating for multiple values
from sklearn.ensemble import RandomForestClassifier

n_estimators = [5, 10, 20, 50, 100]

for i in n_estimators:
    rf = RandomForestClassifier(n_estimators=i)
    rf.fit(x_train, y_train)
    pred = rf.predict(x_train)
    acc = accuracy_score(y_train, pred)
    print('\nTrain accuracy for n_estimators = %f is %f%%' % (i, acc))


# In[66]:


#We will train the algorithm using train data and check the accuracy for test data
rf = RandomForestClassifier(n_estimators=100)
rf.fit(x_train, y_train)
pred = rf.predict(x_test)
acc = accuracy_score(y_test, pred)
print('Test accuracy for n_estimators = 100 is %f%%' % (acc))


# In[67]:


#Printing the accuracy, precision, recall and f1-score
print(classification_report(y_test, pred))


# In[68]:


#Plotting the confusion matrix to check number of TP,TF,FP and FN
plot_confusion_matrix(rf, x_test, y_test)
plt.grid(False)
plt.show()


# # Gradient Boosted Decision Tree

# In[69]:


#importing gradient boosted decision tree from sklearn and checking the ideal value of n by iterating for multiple values
from sklearn.ensemble import GradientBoostingClassifier

n_estimators = [5, 10, 20, 50, 100]

for i in n_estimators:
    gb = GradientBoostingClassifier(n_estimators=i)
    gb.fit(x_train, y_train)
    pred = gb.predict(x_train)
    acc = accuracy_score(y_train, pred)
    print('\nTrain accuracy for n_estimators = %f is %f%%' % (i, acc))


# In[70]:


#We will train the algorithm using train data and check the accuracy for test data
gb = GradientBoostingClassifier(n_estimators=100)
gb.fit(x_train, y_train)
pred = gb.predict(x_test)
acc = accuracy_score(y_test, pred)
print('Test accuracy for n_estimators = 100 is %f%%' % (acc))


# In[71]:


#Printing the accuracy, precision, recall and f1-score
print(classification_report(y_test, pred))


# In[72]:


#Plotting the confusion matrix to check number of TP,TF,FP and FN
plot_confusion_matrix(gb, x_test, y_test)
plt.grid(False)
plt.show()


# # XGBoost

# In[73]:


#importing XGBoost from sklearn and checking the ideal value of n by iterating for multiple values
from xgboost import XGBClassifier

n_estimators = [5, 10, 20, 50, 100]

for i in n_estimators:
    xgb = XGBClassifier(n_estimators=i, n_jobs= -1, eval_metric='logloss')
    xgb.fit(x_train, y_train)
    pred = xgb.predict(x_train)
    acc = accuracy_score(y_train, pred)
    print('\nTrain accuracy for n_estimators = %f is %f%%' % (i, acc))


# In[74]:


#We will train the algorithm using train data and check the accuracy for test data
xgb = XGBClassifier(n_estimators=100, n_jobs= -1, eval_metric='logloss')
xgb.fit(x_train, y_train)
pred = xgb.predict(x_test)
acc = accuracy_score(y_test, pred)
print('Test accuracy for n_estimators = 100 is %f%%' % (acc))


# In[75]:


#Printing the accuracy, precision, recall and f1-score
print(classification_report(y_test, pred))


# In[76]:


#Plotting the confusion matrix to check number of TP,TF,FP and FN
plot_confusion_matrix(xgb, x_test, y_test)
plt.grid(False)
plt.show()


# # Conclusion

# - Machine Learning algorithms like KNN, Random Forest, Gradient Boosted Decision Tree and XGBoost were trained on the training dataset and then evaluated on the test dataset. 
# - Hyperparameter tuning is performed for each algorithms to identify the best parameter value.
# - Confusion matrix is used for each algoritms to understand the number of misclassifications on the test dataset.
# - For KNN algorithm, the test accuracy is 76.8% with 4,067 misclassications out of total 17,542 datapoints.
# - For Random Forest algorithm, the test accuracy is 86.4% with 2,383 misclassications out of total 17,542 datapoints.
# - For Gradient Boosted Decision Tree algorithm, the test accuracy is 79.3% with 3,615 misclassications out of total 17,542 datapoints.
# - For XGBoost algorithm, the test accuracy is 82.4% with 3,074 misclassications out of total 17,542 datapoints.
# - Hence it is concluded that the Random Forest is the most suitable algorithm in determining smoking activity through body signals with the accuracy of 86.4%.
# 

# In[ ]:




