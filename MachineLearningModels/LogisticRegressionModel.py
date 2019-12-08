import random
import time

import numpy as np
import scipy.io
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

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
    X, y = BalanceData.balance_dt(minmax_dat, label, seed=i)

    parameters = {'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
                  'random_state': [i], 'max_iter': [100, 300, 500, 1000]}

    clf = GridSearchCV(LogisticRegression(), parameters, n_jobs=-1, cv=10, verbose=1, scoring='recall')
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

eval_with_kfold(best_es, best_x, best_y, minmax_dat, label)
