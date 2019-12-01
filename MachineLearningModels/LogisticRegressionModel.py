import time

import numpy as np
import scipy.io
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

mat = scipy.io.loadmat('../data750.mat')
org_dat = mat['OriginalData']
stand_dat = mat['Scaled_Standardization']
minmax_dat = mat['Scaled_Min_Max']
label = mat['label'][0]

data = minmax_dat

parameters = {'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
              'random_state': np.arange(0, 10), 'max_iter': [100, 300, 500, 1000]}

initial_start_time = time.time()

clf = GridSearchCV(LogisticRegression(), parameters, n_jobs=-1, cv=10, verbose=1)
clf.fit(data, label)

print("-----------------Results--------------------")
print("Best score: ", clf.best_score_)
print("Using the following parameters:")
print(clf.best_params_)
print("------------------------------------------------------")


clf = GridSearchCV(LogisticRegression(), parameters, n_jobs=-1, cv=10, verbose=1)
clf.fit(stand_dat, label)
print("-----------------Results--------------------")
print("Best score: ", clf.best_score_)
print("Using the following parameters:")
print(clf.best_params_)
print("------------------------------------------------------")

print("Total --- %s seconds ---" % (time.time() - initial_start_time))

# 0.9543243243243245 liblinear
