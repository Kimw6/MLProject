import numpy as np
from numpy import shape
from scipy.io import savemat
from sklearn import preprocessing

from PreProcessing.GraphClass import Graph

with open('data750/countLabelDict.txt') as f:
    count = eval(f.read())

print(count)

data = [[0 for i in range(4)] for j in range(3700)]
print(shape(data))

with open('data750/data750.txt_nemoProfile.txt') as f:
    for line in f:
        if line.startswith('N'):
            continue
        else:
            ll = list(map(int, line.split()))
            data[ll[0]] = ll[1:]
data = np.array(data)
org_label = {}
with open("label.txt") as f:
    for line in f:
        (key, val) = line.split()
        org_label[key.strip()] = val.strip()
print("label_dict length", len(org_label))

graph_part = Graph()
with open("networkdata.txt") as f:
    for line in f:
        v1, v2, w = line.split()
        if v1 in org_label and v2 in org_label and int(w) >= 750:
            graph_part.add_edge(v1, v2, int(w))

print("number of vertices in graph", graph_part.num_vertices)

trans_dict = {}
trans_dict_reverse = {}
idx = 0
for v in graph_part.vertices:
    trans_dict[idx] = v
    trans_dict_reverse[v] = idx
    idx += 1

label_list = []
for i in range(idx):
    item = 1 if (org_label[trans_dict[i]]) == "E" else 0
    label_list.append(item)

print("label_list length", shape(label_list))
scaler = preprocessing.StandardScaler().fit(data)
# print(scaler.mean_)
# print(scaler.scale_)
scaled_dt = scaler.transform(data)

# trainScoreMean, testScoreMean = feature_subset_score(scaled_dt, label_list)
# print(trainScoreMean, testScoreMean)
scaled_dt = np.array(scaled_dt)

min_max_scaler = preprocessing.MinMaxScaler()
scaled_minmax = min_max_scaler.fit_transform(data)

label_list = np.array(label_list)
savemat('data750.mat',
        {'label': label_list, 'OriginalData': data, 'Scaled_Standardization': scaled_dt,
         'Scaled_Min_Max': scaled_minmax})
