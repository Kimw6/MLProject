import time

import numpy as np
import scipy.io
from sklearn.model_selection import KFold
from sklearn.svm import SVC

mat = scipy.io.loadmat('../data750.mat')
org_dat = mat['OriginalData']
stand_dat = mat['Scaled_Standardization']
minmax_dat = mat['Scaled_Min_Max']
label = mat['label'][0]

para1 = ['linear', 'poly', 'rbf']
para2 = [1, 2, 3, 4, 5]

best_score = 0
best_para1 = 0
best_para2 = 0

initial_start_time = time.time()

for i in range(len(para1)):
    for j in range(len(para2)):
        scores = []
        clf = SVC(kernel=para1[i], C=para2[j], gamma='auto')
        cv = KFold(n_splits=10, random_state=None, shuffle=True)
        for train_index, test_index in cv.split(minmax_dat):
            X_train, X_test, y_train, y_test = minmax_dat[train_index], minmax_dat[test_index], \
                                               label[train_index], label[test_index]
            clf.fit(X_train, y_train)
            scores.append(clf.score(X_test, y_test))

        score_avg = np.mean(scores)
        if score_avg > best_score:
            best_score = score_avg
            best_para1 = para1[i]
            best_para2 = para2[j]

print(best_score, best_para1, best_para2)

print("Total --- %s seconds ---" % (time.time() - initial_start_time))

# 0.9540540540540541 rbf 2
