import time

import numpy as np
import scipy.io
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier

mat = scipy.io.loadmat('../data750.mat')
org_dat = mat['OriginalData']
stand_dat = mat['Scaled_Standardization']
minmax_dat = mat['Scaled_Min_Max']
label = mat['label'][0]

data = org_dat

para1 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
para2 = [1, 2]

best_score = 0
best_para1 = 0
best_para2 = 0

initial_start_time = time.time()

for i in range(len(para1)):
    for j in range(len(para2)):
        scores = []
        clf = KNeighborsClassifier(n_neighbors=para1[i], p=para2[j])
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

print(best_score, best_para1, best_para2)

print("Total --- %s seconds ---" % (time.time() - initial_start_time))

# p is Power parameter for the Minkowski metric.
# When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2.

# 0.9535135135135133 8 2
