import random
import time

import scipy.io
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

from Processing import BalanceData
from Processing.util import eval_with_kfold



# parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
#               'learning_rate': ['constant', 'invscaling', 'adaptive'],
#               'hidden_layer_sizes': [tuple([random.randint(1, 101) for n in range(i)]) for i in range(4)],
#               'max_iter': [500], 'random_state': [1, 2, 3]}

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

    parameters = {'activation': ['relu'], 'solver': ['sgd'],
                  'learning_rate': ['constant'],
                  'hidden_layer_sizes': (90,),
                  'max_iter': [200, 500, 1000], 'random_state': [i]}

    # parameters = {'activation': ['identity', 'logistic', 'tanh', 'relu'], 'solver': ['lbfgs', 'sgd', 'adam'],
    #               'learning_rate': ['constant', 'invscaling', 'adaptive'],
    #               'hidden_layer_sizes': [tuple([random.randint(1, 101) for n in range(i)]) for i in range(4)],
    #               'max_iter': [500], 'random_state': [1, 2, 3]}

    clf = GridSearchCV(MLPClassifier(), parameters, n_jobs=-1, cv=10, verbose=1, scoring='f1')
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
