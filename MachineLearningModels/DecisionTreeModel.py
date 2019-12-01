import time

import numpy as np
import scipy.io
from sklearn import tree
from sklearn.model_selection import KFold, cross_val_score

mat = scipy.io.loadmat('../data750.mat')
org_dat = mat['OriginalData']
stand_dat = mat['Scaled_Standardization']
minmax_dat = mat['Scaled_Min_Max']
label = mat['label'][0]

data = stand_dat

para1 = ['gini', 'entropy']
para2 = [1, 2, 3, 4, 5]
para3 = [2, 3, 4]

parameters = {'criterion': ['gini', 'entropy'],
              'random_state': np.arange(0, 10), 'min_samples_leaf': np.arange(1, 10),
              'min_samples_split': np.arange(2, 10), 'min_impurity_split': 10.0 ** -np.arange(1, 10)}

best_score = 0
best_para1 = 0
best_para2 = 0
best_para3 = 0

initial_start_time = time.time()

for i in range(len(para1)):
    for j in range(len(para2)):
        for k in range(len(para3)):
            scores = []
            clf = tree.DecisionTreeClassifier(criterion=para1[i], min_samples_leaf=para2[j], min_samples_split=para3[k])
            cv = KFold(n_splits=10, random_state=None, shuffle=True)
            for train_index, test_index in cv.split(data):
                X_train, X_test, y_train, y_test = data[train_index], data[test_index], \
                                                   label[train_index], label[test_index]
                clf.fit(X_train, y_train)
                scores.append(clf.score(X_test, y_test))

            score_avg = np.mean(scores)
            if score_avg > best_score:
                best_score = score_avg
                best_para1 = para1[i]
                best_para2 = para2[j]
                best_para3 = para3[k]

print(best_score, best_para1, best_para2, best_para3)

print("Total --- %s seconds ---" % (time.time() - initial_start_time))

# 0.9427027027027028 entropy 5 4 stand data
clf = tree.DecisionTreeClassifier(criterion='entropy', min_samples_leaf=5, min_samples_split=4)
sss = cross_val_score(clf, data, label, cv=10)
print(sss, np.mean(sss))
