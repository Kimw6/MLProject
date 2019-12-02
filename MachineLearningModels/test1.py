import time

import matplotlib.pyplot as plt
import numpy as np
import scikitplot as skplt
import scipy.io
from sklearn import metrics
from sklearn.metrics import classification_report, roc_curve, auc, roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier

from Processing import BalanceData

mat = scipy.io.loadmat('../data750.mat')
org_dat = mat['OriginalData']
stand_dat = mat['Scaled_Standardization']
minmax_dat = mat['Scaled_Min_Max']
label = mat['label'][0]

print("1s", np.count_nonzero(label == 1))
print("0s", np.count_nonzero(label == 0))
# for i in range(len(label)):
#     if label[i] == 0:
#         label[i] = 1
#     else:
#         label[i] = 0

X, y = stand_dat, label
# X, y = BalanceData.balanceD(stand_dat, label, 3)

initial_start_time = time.time()

# parameters = {'criterion': ['gini', 'entropy'],
#               'random_state': np.arange(0, 10), 'min_samples_leaf': np.arange(1, 10),
#               'min_samples_split': np.arange(2, 10), 'min_impurity_split': 10.0 ** -np.arange(1, 10)}

parameters = {'criterion': ['gini'],
              'random_state': [2], 'min_samples_leaf': [2],
              'min_samples_split': [2]}

clf = GridSearchCV(DecisionTreeClassifier(), parameters, n_jobs=-1, cv=10, verbose=1)
clf.fit(X, y)

y_pred = clf.best_estimator_.predict(stand_dat)
print("Confusion:\n", metrics.confusion_matrix(label, y_pred))
fpr, tpr, thresholds = roc_curve(label, clf.best_estimator_.predict_proba(stand_dat)[:, 1])
#
y_score = clf.best_estimator_.predict_proba(stand_dat)
skplt.metrics.plot_roc(label, y_score)
plt.show()

print("AUC Prob:", auc(fpr, tpr))
print("AUC Pred", roc_auc_score(label, y_pred))
#
report = classification_report(label, y_pred)
print(report)
print("-----------------Results--------------------")
print("Best score: ", clf.best_score_)
print(clf.best_params_)

print("Total --- %s seconds ---" % (time.time() - initial_start_time))
# {'criterion': 'gini', 'min_impurity_split': 0.1, 'min_samples_leaf': 2, 'min_samples_split': 2, 'random_state': 2}
