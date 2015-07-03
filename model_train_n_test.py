
# coding: utf-8

# <hr>
# # PYTHON FOR DATA SCIENCE
# ### Exploring a Real World Example
# 
# #Agenda
# 
# - <p style="color: red">Define the problem and the approach</p>
# - Data basics: loading data, looking at your data, basic commands
# - Handling missing values
# - Intro to scikit-learn
# - Grouping and aggregating data
# - Feature selection
# - Fitting and evaluating a model
# - Deploying your work
# 

# ##In this notebook you will
# 
# - Determine if the problem is worth solving
# - Define an approach
# - Develop a workflow outline

# <hr>
# ## Revisiting Kaggle's Give Me Some Credit Competition
# 
# ### Intro
# 
# #### We'll be exploring an age-old prediction problem--predicting risk on consumer loans.
# 
# #### We've selected this topic as a case study because the problem is well-defined and familiar to most people. Additionally, this is a binary classification problem, so the strategies we apply should be relatively extensible to other problems you may encounter totally unrelated to credit and finance.
# 
# ### About the data
# 
# The data is made available to us by Kaggle and was used in a competition in 2011.
# 
# [http://www.kaggle.com/c/GiveMeSomeCredit](http://www.kaggle.com/c/GiveMeSomeCredit)
# 
# ### About the prediction problem
# 
# Predict the probability that somebody will experience financial distress in the next two years.

# ## Developing an understanding of the data and the problem
# 
# #### Key questions we'll need to keep in mind
# 
# - How do losses occur?
# - What are the characteristics that constitute credit default?
# - How often do people "go bad?"
# - How might we improve loss rates?

# In[2]:

import pandas as pd
df = pd.read_csv("./data/credit-training.csv")


# 
# Column Types?
print df.dtypes

# Missing Value?
# find columns that have null values
print df.isnull().any(axis=0)
# looks like MonthlyIncome has a lot of missing value
print df.shape
print "# null values in MonthlyIncome: %i" % df['MonthlyIncome'].isnull().sum()
print "# null values in NumberOfDependents: %i" % sum(df['NumberOfDependents'].isnull() == True)
df[df.isnull().any(axis=1)]

# Summary?
df.describe()
df.info() # info() is a faster way to spot null values


# In[3]:

# inspect unique and unique counts
print df['NumberOfDependents'].nunique()
print df['NumberOfDependents'].unique()

for col in df.columns:
    print "column %s" % col
    print df[col].value_counts(sort=True).head(5)


# In[3]:

# crosstab is powerful
print df['SeriousDlqin2yrs'].value_counts()
print pd.crosstab(df['NumberOfDependents'],df['SeriousDlqin2yrs'])


# In[4]:

# Basic cleaning
import re
def camel_to_snake(column_name):
    """
    converts a string that is camelCase into snake_case
    Example:
        print camel_to_snake("javaLovesCamelCase")
        > java_loves_camel_case
    See Also:
        http://stackoverflow.com/questions/1175208/elegant-python-function-to-convert-camelcase-to-camel-case
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', column_name)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()

df.columns = [camel_to_snake(col) for col in df.columns]
df.columns.tolist()


# In[66]:

# Basic filtering
#mask = (df.age >= 35) & (df.serious_dlqin2yrs==0) & (df.number_of_open_credit_lines_and_loans < 10)




# In[7]:

# handle number_of_dependents
df.number_of_dependents.value_counts()
# we naively fill the missing value with the mode 
df.number_of_dependents = df.number_of_dependents.fillna(0)
df.info()


# In[8]:

# we need to take care of monthly_income more sophisticatedly..

# the naive method is to simply fill with median or mean
df.monthly_income.describe()

import numpy as np
# use imputation: we try to infer the value from other columns
# split data into train and test
np.random.seed(713)
is_test = np.random.uniform(0, 1, len(df)) > 0.75
train = df[is_test==False]
test = df[is_test==True]
print len(train), len(test)
# remember to apply the imputer to the test data


# In[9]:

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

rows_train = train.monthly_income.notnull()
rows_test = test.monthly_income.notnull()

# get the correlation pairs
train[rows_train].corr().ix[:, 5]


# find the most correlated variables
cols_x = ['number_of_open_credit_lines_and_loans', 'number_of_dependents', 'number_real_estate_loans_or_lines']

train_not_null = train[cols_x][rows_train]
test_not_null = test[cols_x][rows_test]


lr = LinearRegression()
lr.fit(train_not_null, train['monthly_income'][rows_train])
print lr.score(test_not_null, test['monthly_income'][rows_test])
# score: 0.0478125755667
knn = KNeighborsRegressor(n_neighbors=120)
knn.fit(train_not_null, train['monthly_income'][rows_train])
print knn.score(test_not_null, test['monthly_income'][rows_test])
# score: 0.00680687486842

# use linear regression model as imputer


# In[10]:

train[rows_train].corr().ix[:, 5]


# In[11]:

train_null = train[cols_x][~rows_train]
test_null = test[cols_x][~rows_test]

new_values_train = lr.predict(train_null)
train.loc[~rows_train, 'monthly_income'] = new_values_train

new_values_test = lr.predict(test_null)
test.loc[~rows_test, 'monthly_income'] = new_values_test

# ways to deal with the (false positive) warning
#train.loc[~rows_train, 'monthly_income']


# In[12]:

#train.describe()
#test.describe()
#df.describe()
#train.to_csv("./data/credit-data-trainingset-hx.csv", index=False)
#test.to_csv("./data/credit-data-testset-hx.csv", index=False)
test.info()
train.info()


#

# In[15]:

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(max_features = "sqrt", bootstrap = True, oob_score = True, 
                            random_state = np.random.RandomState(713))
features = np.array(['age',
            'revolving_utilization_of_unsecured_lines',
            'number_of_time30-59_days_past_due_not_worse',
            'capped_debt_ratio',
            'log_monthly_income',
            'number_of_open_credit_lines_and_loans',
            'number_of_times90_days_late',
            'number_real_estate_loans_or_lines',
            'number_of_time60-89_days_past_due_not_worse',
            'number_of_dependents'])

rf.fit(train[features], train['serious_dlqin2yrs'])


# In[16]:

importances = rf.feature_importances_
#zip(features, importances)
s_index = np.argsort(importances) # np.argsort returns index instead of sorted array like np.sort

y_pos = np.arange(len(importances))
plt.barh(y_pos, importances[s_index])
plt.yticks(y_pos, features[s_index])
plt.xlabel("Relative Importance")
plt.title("Variable Importance")


# In[24]:

#rf.score(test[features], test['serious_dlqin2yrs'])
rf.oob_score_
rf.predict(train[features])


# In[221]:

# Various ways of feature engineering

### Bucketize monthly_income
# Result: not helpful


#pd.cut(train.monthly_income, bins=15, labels=False)
#train['bin_monthly_income'] = pd.cut(train.monthly_income.apply(lambda x: cap(x, 15000)), bins=15, labels=False)

train[['bin_monthly_income', 'serious_dlqin2yrs']].groupby('bin_monthly_income').mean().plot()


# In[287]:

#train[['age', 'serious_dlqin2yrs']].groupby('age').mean().plot()
#train.age.describe()

### Bucketize age
# Result: not helpful

mybins = [-1] + range(20, 90, 5) + [110]
train['bin_age'] = pd.cut(train.age, bins=mybins, labels=False)
# if labels=True
#train.bin_age = pd.factorize(train.bin_age)[0]
#train.bin_age.head()

train[['bin_age', 'serious_dlqin2yrs']].groupby('bin_age').mean().plot()


# In[328]:

#########################################################
################ Test Zone ##############################
#########################################################
hasattr(classifiers[1], "decision_function")


# In[242]:

train[['normalized_monthly_income', 'serious_dlqin2yrs']].groupby('normalized_monthly_income').mean().reset_index().plot(kind='scatter', x='normalized_monthly_income', y='serious_dlqin2yrs', alpha=0.2, color=)


# In[64]:


from sklearn.cross_validation import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.lda import LDA
from sklearn.qda import QDA

from sklearn.metrics import roc_auc_score, accuracy_score, classification_report, confusion_matrix



X, y = train[features], train.serious_dlqin2yrs

names = ["Logistic Regression", 
         #"Nearest Neighbors", 
         #"Linear SVM", 
         "Random Forest", 
         "AdaBoost", 
         "Gradient Boosting", 
         "Naive Bayes", 
         "LDA", 
         "QDA"]
classifiers = [
    LogisticRegression(),
    #KNeighborsClassifier(n_neighbors=50),
    #SVC(kernel='linear'),
    RandomForestClassifier(max_features = "sqrt", bootstrap = True,
                            random_state = 713),
    AdaBoostClassifier(),
    GradientBoostingClassifier(),
    GaussianNB(),
    LDA(),
    QDA()]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1234)

sclr = StandardScaler().fit(X_train)
X_train = sclr.transform(X_train)
X_test = sclr.transform(X_test)


    
    


# In[65]:


import time
for name, clf in zip(names, classifiers):
    print '*'*55
    print name
    print '*'*55
    p1 = time.clock()
    clf.fit(X_train, y_train)
    p2 = time.clock()
    print "Training time : %.3f" % (p2 - p1)

    if hasattr(clf, "decision_function"):
        y_score = clf.decision_function(X_test)
    else:
        y_score = clf.predict_proba(X_test)[:, 1]

    #y_pred = clf.predict(X_test)
    # predict lable using probability manually to save time 
    y_pred0 = np.array(map(lambda x: 1 if x > 0.5 else 0, y_score))

    #print y_pred[np.not_equal(y_pred, y_pred0)]
    #print y_score[np.not_equal(y_pred, y_pred0)]
    #print np.where(y_pred != y_pred0)

    p3 = time.clock()
    print "Predicting time : %.3f" % (p3 - p2)

    #print clf.classes_
    print "Accuracy score: %.3f" % (accuracy_score(y_test, y_pred0))
    print "AUC score: %.3f" % (roc_auc_score(y_test, y_score))
    print classification_report(y_test, y_pred0)



    
    
    


# In[67]:

the_clf = classifiers[-1]
the_clf.fit(X, y)
y_pred = the_clf.predict(test[features])
print classification_report(test.serious_dlqin2yrs, y_pred)







