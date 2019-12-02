from Processing.GraphClass import Graph

org_label = {}
with open("label.txt") as f:
    for line in f:
        (key, val) = line.split()
        org_label[key.strip()] = val.strip()
print("label_dict", len(org_label))

# graph = Graph()
# with open("networkdata.txt") as f:
#     for line in f:
#         v1, v2, w = line.split()
#         graph.add_edge(v1, v2, int(w))
#
# print("network vertices num", graph.num_vertices)
wlist = []
elist = []
numlist = []
for ww in range(1, 1000, 10):
    graph_part = Graph()
    with open("networkdata.txt") as f:
        for line in f:
            v1, v2, w = line.split()
            if v1 in org_label and v2 in org_label and int(w) >= ww:
                graph_part.add_edge(v1, v2, int(w))

    trans_dict = {}
    idx = 0
    for v in graph_part.vertices:
        trans_dict[idx] = v
        idx += 1

    label_list = []
    for i in range(idx):
        item = 1 if (org_label[trans_dict[i]]) == "E" else 0
        label_list.append(item)
    # print("Count of essentials", label_list.count(1))

    num_edges = 0
    for v in graph_part.get_vertices():
        num_edges += len(graph_part.get_vertex(v).get_neighbors())
    numlist.append(num_edges)
    elist.append(label_list.count(1))
    wlist.append(ww)


graph_part = Graph()
with open("networkdata.txt") as f:
    for line in f:
        v1, v2, w = line.split()
        if v1 in org_label and v2 in org_label and int(w) >= 999:
            graph_part.add_edge(v1, v2, int(w))

trans_dict = {}
idx = 0
for v in graph_part.vertices:
    trans_dict[idx] = v
    idx += 1

label_list = []
for i in range(idx):
    item = 1 if (org_label[trans_dict[i]]) == "E" else 0
    label_list.append(item)
# print("Count of essentials", label_list.count(1))

num_edges = 0
for v in graph_part.get_vertices():
    num_edges += len(graph_part.get_vertex(v).get_neighbors())
numlist.append(num_edges)
elist.append(label_list.count(1))
wlist.append(999)


with open('elist.txt', 'w') as f:
    for item in elist:
        f.write("%s\n" % item)
with open('numlist.txt', 'w') as f:
    for item in numlist:
        f.write("%s\n" % item)
sum_essential = sum(1 for x in org_label.values() if x == 'E')
# print(sum_essential)
# num_edges = 0
# for v in graph_part.get_vertices():
#     num_edges += len(graph_part.get_vertex(v).get_neighbors())
# print(num_edges)  # 1021786 correct