#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 21 02:27:14 2018

@author: thiru
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn import metrics,cross_validation
from matplotlib import pyplot as plt
import itertools
import numpy as np
from sklearn.cross_validation import StratifiedShuffleSplit,train_test_split

wd = '/storage/git/mckinsey_hackathon/to-be-renamed/'
filepath = wd + 'train.csv'

def xgboostCV(clf, dataset,target ,useTrainCV=True, cv_folds=5, early_stopping_rounds=25):
    print('Running XGBOOST cross validation')
    if useTrainCV:
        xgb_param = clf.get_xgb_params()
        xgtrain = xgb.DMatrix(dataset.values, label=target.values)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=clf.get_params()['n_estimators'], nfold=cv_folds,
            metrics=['auc'], early_stopping_rounds=early_stopping_rounds)
        CV_ROUNDS = cvresult.shape[0]
        print('Optimal Rounds: '+str(CV_ROUNDS))
        clf.set_params(n_estimators=CV_ROUNDS)
    return clf

def splitTrainTest(dataset,target,test_size=0.20):
    trainX,trainY,testX,testY = train_test_split(dataset,target, 
                                                 test_size=test_size, 
                                                 random_state=123,stratify=target)
    return trainX,trainY,testX,testY

def accuracyChecker(target,predicted):
    accuracy = metrics.accuracy_score(target,predicted)
    confMat = metrics.confusion_matrix(target,predicted)
    print('Cross val accuracy: '+str(accuracy))
    print('Confusion Matrix:')
    print(confMat)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

def encoding(df):
    sc = pd.get_dummies(df['sourcing_channel'])
    rat = pd.get_dummies(df['residence_area_type'])
    df = pd.concat([df, sc], axis=1)
    df = pd.concat([df, rat], axis=1)
    
    del df['sourcing_channel']
    del df['residence_area_type']
    return df

    
#==============================================================================
# modelling
#==============================================================================
df = pd.read_csv(filepath)
df = df.drop('id',axis=1)
df = encoding(df)


df['age_in_days'] = df['age_in_days'] / 365 #doesnt matter for tree based methods

clf = xgb.XGBClassifier(max_depth = 8,n_estimators=63,nthread=8,seed=123,
                            silent=1,objective= 'binary:logistic',learning_rate=0.08)

dataset = df.drop('renewal',axis=1)
target = df['renewal']
clf_cv = xgboostCV(clf,dataset,target)

 
 
trainX,testX,trainY,testY = splitTrainTest(dataset,target)
clf_cv.fit(trainX,trainY,eval_metric='auc')

## manual probability assignment
predicted_proba = clf.predict_proba(testX)[:,1]
threshold = 0.9
predicted = [1 if x >= threshold else 0 for x in predicted_proba]

accuracyChecker(testY,predicted)
roc_auc = metrics.roc_auc_score(testY,predicted)
print("ROC-AUC: {}".format(roc_auc))

class_names = [0,1]
cnf_matrix = metrics.confusion_matrix(testY, predicted)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

#==============================================================================
# revenue calculation
#==============================================================================

test = pd.read_csv(wd + "test.csv")
id_ = test['id']
test.drop('id',inplace=True,axis=1)
test = encoding(test)

default_incentive = 460

incentives = np.array([default_incentive for x in range(len(test))])

effort = 10*(1-np.exp(-incentives/400)) #assumed to be true
increase_renewal_prob = 20*(1-np.exp(-effort/5)) #assumed to be true

test_pred_proba = clf_cv.predict_proba(test)[:,1]

delta_p = increase_renewal_prob*test_pred_proba
net_revenue = sum((test_pred_proba + delta_p)*test['premium'] - incentives)

w1 = 0.7
w2 = 0.3
l = 1
#overall_score = w1*roc_auc + w2*(net_revenue)*l

#==============================================================================
# submission
#==============================================================================

submit_df = pd.DataFrame({"id":id_,
                          "renewal":test_pred_proba,
                          "incentives":incentives})
submit_df.to_csv(wd + "submission2.csv",index=False) 


#==============================================================================
# notes
#==============================================================================
# current max on leaderboard is 0.72
# using incentive 1000 - score - 0.6798737995. 
# using incentive 460 - score - 0.6875179509832214. 