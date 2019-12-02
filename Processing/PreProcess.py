from Processing.GraphClass import Graph

org_label = {}
with open("../dataOther/label.txt") as f:
    for line in f:
        (key, val) = line.split()
        org_label[key.strip()] = val.strip()
print("label_dict", len(org_label))

graph_part = Graph()
with open("../dataOther/networkdata.txt") as f:
    for line in f:
        v1, v2, w = line.split()
        if v1 in org_label and v2 in org_label and int(w) >= 750:
            graph_part.add_edge(v1, v2, int(w))

print(graph_part.num_vertices)

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
print("Count of essentials", label_list.count(1))

num_edges = 0
for v in graph_part.get_vertices():
    num_edges += len(graph_part.get_vertex(v).get_neighbors())
print(num_edges)  # 1021786 correct

# with open('NetworkForMotif999.txt', 'w') as f:
#     for v in graph_part.get_vertices():
#         vwrite = trans_dict_reverse[v]
#         for t in graph_part.get_vertex(v).get_neighbors():
#             f.write("%d %d\n" % (vwrite, trans_dict_reverse[t.id]))
