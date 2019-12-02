import random
import time

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import GridSearchCV, cross_val_predict
from sklearn.tree import DecisionTreeClassifier

from Processing import BalanceData
from Processing.util import eval_with_kfold

mat = scipy.io.loadmat('../data750.mat')
org_dat = mat['OriginalData']
stand_dat = mat['Scaled_Standardization']
minmax_dat = mat['Scaled_Min_Max']
label = mat['label'][0]

best_sc = 0
best_x = []
best_y = []
best_es = None
initial_start_time = time.time()

for i in range(10):
    random.seed(i)
    X, y = BalanceData.balance_dt(stand_dat, label, seed=i)

    parameters = {'criterion': ['gini', 'entropy'],
                  'min_samples_leaf': np.arange(1, 10),
                  'min_samples_split': np.arange(2, 10), 'random_state': [i], }

    clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=-1, cv=10, verbose=1, scoring='f1')
    clf.fit(X, y)

    if clf.best_score_ > best_sc:
        best_sc = clf.best_score_
        best_es = clf.best_estimator_
        best_x = X
        best_y = y

print("-----------------Results--------------------")
print("Best score: ", best_sc)
print(best_es)
print("Total --- %s seconds ---" % (time.time() - initial_start_time))

aa, bb = BalanceData.balance_dt(stand_dat, label)

eval_with_kfold(best_es, best_x, best_y, aa, bb)
