#!/usr/bin/env python
# coding: utf-8

# The dataset for this assignment is file **whitewine.csv** which is provided with this notebook.

# ## Dataset
# 
# The dataset was adapted from the Wine Quality Dataset (https://archive.ics.uci.edu/ml/datasets/Wine+Quality)
# 
# ### Attribute Information:
# 
# For more information, read [Cortez et al., 2009: http://dx.doi.org/10.1016/j.dss.2009.05.016].
# 
# Input variables (based on physicochemical tests):
# 
#     1 - fixed acidity 
#     2 - volatile acidity 
#     3 - citric acid 
#     4 - residual sugar 
#     5 - chlorides 
#     6 - free sulfur dioxide 
#     7 - total sulfur dioxide 
#     8 - density 
#     9 - pH 
#     10 - sulphates 
#     11 - alcohol 
# Output variable (based on sensory data):
# 
#     12 - quality (0: normal wine, 1: good wine)
#     
# ## Problem statement
# Predict the quality of a wine given its input variables. Use AUC (area under the receiver operating characteristic curve) as the evaluation metric.

# First, let's load and explore the dataset.

# In[1]:


import numpy as np
import pandas as pd

np.random.seed = 42


# In[2]:


data = pd.read_csv("C:/Users/OEM/Desktop/DATA201/A4/whitewine.csv")
data.head()


# In[3]:


data.info()


# In[4]:


data["quality"].value_counts()


# Please note that this dataset is unbalanced.

# ## Questions and Code

# **[1]. Split the given data using stratify sampling into 2 subsets: training (80%) and test (20%) sets. Use random_state = 42. **

# In[5]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(data, train_size=0.8, test_size=0.2, random_state = 42)
print(len(train_set), ' Training Set +', len(test_set), ' Testing Set')


# **[2]. Use ``GridSearchCV`` and ``Pipeline`` to tune hyper-parameters for 3 different classifiers including ``KNeighborsClassifier``, ``LogisticRegression`` and ``svm.SVC`` and report the corresponding AUC values on the training and test sets. Note that a scaler may need to be inserted into each pipeline.**

# Hint: You may want to use `kernel='rbf'` and tune `C` and `gamma` for `svm.SVC`. Find out how to enable probability estimates (for Question 3).
# 
# Document: https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html#sklearn.svm.SVC

# In[6]:


#Perform hyper parameter optimization
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


#Same for train set(Target is purchase)
X_train = train_set.drop("quality", axis=1) # drop labels for training set
y_train = train_set["quality"].copy()

#same for test set
X_test = test_set.drop("quality", axis=1)
y_test = test_set["quality"].copy()


# In[7]:


#------KNeighborsClassifier-----
pipelineone = Pipeline([('knn', KNeighborsClassifier())])
param_grid = {'knn__n_neighbors':[1,2,3,4,5,6,7,8,9,10],
              'knn__leaf_size': [10,20,30,40,50,60,70,80,90,100]
             }

#Gridsearch takes in param_grid, and pipeline.
gridone = GridSearchCV(pipelineone, param_grid, cv =5, scoring = 'roc_auc')
gridone.fit(X_train, y_train)

testscore = gridone.score(X_test, y_test)
trainscore = gridone.score(X_train, y_train)
print('----AUC Values from KNN train and test Set----')
print('test accuracy: ', testscore)
print('train test accuracy: ', trainscore)


#on the testing data
y_testpred = gridone.predict(X_test)

#on the training data
y_trainpred = gridone.predict(X_train)

# summarize and present ROC score
#roc takes in y_true and y_score
KNNAUCtest = roc_auc_score(y_test, y_testpred)
KNNAUCtrain = roc_auc_score(y_train, y_trainpred)
print('Best parameters: ', gridone.best_params_)
print('KNN AUC value from test set: ', (KNNAUCtest))
print('KNN AUC value from train set: ', (KNNAUCtrain))


# In[8]:


#-------LogisticRegression-----
#[0.001, 0.01, 0.1, 1, 10, 100, 1000]
pipelinetwo = Pipeline([('logisticregression', LogisticRegression())])
param_grid = {'logisticregression__penalty' : ['l2'],
              'logisticregression__C' : [10],
              'logisticregression__solver' : ['liblinear']}
#Gridsearch takes in param_grid, and pipeline.
gridtwo = GridSearchCV(pipelinetwo, param_grid, cv =5, scoring = 'roc_auc')
gridtwo.fit(X_train, y_train)

testscore = gridtwo.score(X_test, y_test)
trainscore = gridtwo.score(X_train, y_train)
print('----AUC Values from Logistic Regression train and test Set----')
print('test accuracy: ', testscore)
print('train test accuracy: ', trainscore)


#on the testing data
y_testpred = gridtwo.predict(X_test)

#on the training data
y_trainpred = gridtwo.predict(X_train)

# summarize and present ROC score
#roc takes in y_true and y_score
logAUCtest = roc_auc_score(y_test, y_testpred)
logAUCtrain = roc_auc_score(y_train, y_trainpred)
print('Best parameters: ', gridtwo.best_params_)
print('Logistic Regress AUC value from test set: ', (logAUCtest))
print('Logistic Regress AUC value from train set: ', (logAUCtrain))


# In[9]:


#-----------svm.SVC------------
pipelinethree = Pipeline([('svc', SVC(kernel='rbf', probability = True))])

param_grid = {
        'svc__C': [1],
        'svc__gamma': [1]}

#Gridsearch takes in param_grid, and pipeline.
gridthree = GridSearchCV(pipelinethree, param_grid, cv =5, scoring = 'roc_auc')
gridthree.fit(X_train, y_train)

testscore = gridthree.score(X_test, y_test)
trainscore = gridthree.score(X_train, y_train)
print('----AUC Values from SVM train and test Set----')
print('test accuracy: ', testscore)
print('train test accuracy: ', trainscore)

#on the testing data
y_testpred = gridthree.predict(X_test)

#on the training data
y_trainpred = gridthree.predict(X_train)

# summarize and present ROC score
#roc takes in y_true and y_score
SVMAUCtest = roc_auc_score(y_test, y_testpred)
SVMAUCtrain = roc_auc_score(y_train, y_trainpred)
print('Best parameters: ', gridthree.best_params_)
print('SVC AUC value from test set: ', (SVMAUCtest))
print('SVC AUC value from train set: ', (SVMAUCtrain))


# **[3]. Train a soft ``VotingClassifier`` with the estimators are the three tuned pipelines obtained from [2]. Report the AUC values on the training and test sets. Comment on the performance of the ensemble model.**
# 
# Hint: consider the voting method.
# 
# Document: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html#sklearn.ensemble.VotingClassifier

# In[10]:


from sklearn.ensemble import VotingClassifier 
from sklearn.metrics import accuracy_score 

# Voting Classifier with soft voting 

estim = [('knn', gridone), ('lr', gridtwo),('svc', gridthree)]

softvote = VotingClassifier(estimators = estim, voting ='soft') 
softvote.fit(X_train, y_train) 

#on the testing data
y_testproba = softvote.predict_proba(X_test)
#on the training data
y_trainproba = softvote.predict_proba(X_train)

#ROC
print('AUC test score with softvote: ', roc_auc_score(y_test,y_testproba[:, 1]))
print('AUC train score with softvote: ', roc_auc_score(y_train,y_trainproba[:, 1]))


# COMMENT ON PERFORMANCE OF THE ENSEMBLE MODEL:
# 
# Predictions for the training set improved drastically. The ensemble model has a training AUC value that is pretty much correct 100% and is outstanding. The AUC values for test score have improved overtime with a model that is 86.9% correct. The voting classifier helped improve the test score overall.

# **[4]. Redo [3] with a sensible set of ``weights`` for the estimators. Comment on the performance of the ensemble model in this case. **

# In[11]:


from sklearn.ensemble import VotingClassifier 
from sklearn.metrics import accuracy_score 

# Voting Classifier with soft voting 
estim = [('knn', gridone), ('lr', gridtwo),('svc', gridthree)]

#Sensible set of weights:
#KNN = 0
#LR = 0
#SVC = 1 
softvote = VotingClassifier(estimators = estim, voting ='soft',  weights = [0,0,1] ) 
softvote.fit(X_train, y_train) 

#on the testing data
y_testpred = softvote.predict_proba(X_test)
#on the training data
y_trainpred = softvote.predict_proba(X_train)

# summarize and present ROC score
#roc takes in y_true and y_score
VotingAUCtest = roc_auc_score(y_test, y_testpred[:,1])
VotingAUCtrain = roc_auc_score(y_train, y_trainpred[:,1])
print('SOFT AUC value from test set(With Weights): ', (VotingAUCtest))
print('SOFT AUC value from train set(With Weights): ', (VotingAUCtrain))


# COMMENT ON THE PERFORMANCE OF THE ENSEMBLE MODEL:
# 
# It does not really improve. When you add weights, you are looping and checking which is the best combination by weighting the occurrences of the predicted labels. This can somtimes improve the model although, at a cost of longer time. The prev result already has a training set that was 100%, but the test set ended up downgrading.

# **[5]. Use the ``VotingClassifier`` with ``GridSearchCV`` to tune the hyper-parameters of the individual estimators. The parameter grid should be a combination of those in [2]. Report the AUC values on the training and test sets. Comment on the performance of the ensemble model. **
# 
# Note that it may take a long time to run your code for this question.
# 
# Document: https://scikit-learn.org/stable/modules/ensemble.html#using-the-votingclassifier-with-gridsearchcv

# In[12]:


from sklearn.ensemble import VotingClassifier 
from sklearn.metrics import accuracy_score 

classifier1 = KNeighborsClassifier()
classifier2 = LogisticRegression()
classifier3 = SVC(kernel='rbf', probability = True)

#combine to make a voting pipeline
voteclassifier = VotingClassifier(estimators = [('knn', classifier1), ('logisticregression', classifier2),
                                                ('svc', classifier3)], voting ='soft')

#params for combination of all 3's best params
params = {'logisticregression__penalty' : ['l2'],
              'logisticregression__C' : [10],
              'logisticregression__solver' : ['liblinear'],
              'knn__n_neighbors': [5],
              'knn__leaf_size': [10],
              'svc__C': [1],
              'svc__gamma': [1]
             }
#combine voting classifier with gridsearch
#Gridsearch takes in param_grid, and estimator.
gridmerge = GridSearchCV(voteclassifier, param_grid=params, cv =5, scoring = 'roc_auc')
gridmerge = gridmerge.fit(X_train, y_train)
#on the testing data
y_testpred = gridmerge.predict_proba(X_test)
#on the training data
y_trainpred = gridmerge.predict_proba(X_train)

# summarize and present ROC score
#roc takes in y_true and y_score
COMBINEAUCtest = roc_auc_score(y_test, y_testpred[:,1])
COMBINEAUCtrain = roc_auc_score(y_train, y_trainpred[:,1])
print('Combine AUC value from test set: ', (COMBINEAUCtest))
print('Combine AUC value from train set: ', (COMBINEAUCtrain))


# COMMENT ON PERFORMANCE OF ENSEMBLE MODEL
# 
# Supposedly takes a very very long time as you are putting multiple classifiers into gridsearch. Test Set has improved again to be almost equivalent to what q 3 had. Although there is slight differences being that the one outputted here is 0.0001 of a difference. Which is very insignificant. Training set is still 100%

# In[ ]:




