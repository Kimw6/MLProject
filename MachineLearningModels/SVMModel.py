import random
import time

import scipy.io
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

from Processing import BalanceData
from Processing.util import eval_with_kfold
import numpy as np
import scipy.io
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt

mat = scipy.io.loadmat('../data750_full.mat')
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

    parameters = {'kernel': ['linear', 'poly', 'rbf'], 'random_state': [i], 'C': [1, 2, 3, 4, 5]}

    clf = GridSearchCV(SVC(probability=True), parameters, n_jobs=-1, cv=10, verbose=1, scoring='f1')
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

# mean_fpr = np.linspace(0, 1, 100)
# tprs = []
# aucs = []
# i = 1
# clf = best_es
# cv = StratifiedKFold(n_splits=10, shuffle=False)
# for train, test in cv.split(best_x, best_y):
#     prediction = clf.fit(best_x[train], best_y[train]).predict_proba(best_x[test])
#     fpr, tpr, t = roc_curve(best_y[test], prediction[:, 1])
#     tprs.append(interp(mean_fpr, fpr, tpr))
#     roc_auc = auc(fpr, tpr)
#     aucs.append(roc_auc)
#     plt.plot(fpr, tpr, lw=2, alpha=0.3, label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
#     i = i + 1
#
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('ROC Curves for SVM')
# plt.legend(loc="lower right")
# plt.show()
eval_with_kfold(best_es, best_x, best_y, minmax_dat, label)
