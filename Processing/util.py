import numpy as np
from sklearn.model_selection import cross_val_score


def eval_with_kfold(best_clf, x, y, org_dt, org_lb):
    cros_res = cross_val_score(best_clf, x, y, cv=10, scoring='accuracy')
    print("cross_res accuracy", np.mean(cros_res))
    cros_res = cross_val_score(best_clf, org_dt, org_lb, cv=10, scoring='accuracy')
    print("cross_res accuracy", np.mean(cros_res))

    cros_res = cross_val_score(best_clf, x, y, cv=10, scoring='precision')
    print("cross_res precision", np.mean(cros_res))
    cros_res = cross_val_score(best_clf, org_dt, org_lb, cv=10, scoring='precision')
    print("cross_res precision", np.mean(cros_res))

    cros_res = cross_val_score(best_clf, x, y, cv=10, scoring='recall')
    print("cross_res recall", np.mean(cros_res))
    cros_res = cross_val_score(best_clf, org_dt, org_lb, cv=10, scoring='recall')
    print("cross_res recall", np.mean(cros_res))

    cros_res = cross_val_score(best_clf, x, y, cv=10, scoring='f1')
    print("cross_res f1", np.mean(cros_res))
    cros_res = cross_val_score(best_clf, org_dt, org_lb, cv=10, scoring='f1')
    print("cross_res f1", np.mean(cros_res))

    cros_res = cross_val_score(best_clf, x, y, cv=10, scoring='roc_auc')
    print("cross_res auc", np.mean(cros_res))
    cros_res = cross_val_score(best_clf, org_dt, org_lb, cv=10, scoring='roc_auc')
    print("cross_res auc", np.mean(cros_res))
