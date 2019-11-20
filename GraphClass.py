class Vertex:
    def __init__(self, node):
        self.id = node
        self.neighbors = {}

    def __str__(self):
        return str(self.id) + ' adjacent: ' + str([x.id for x in self.neighbors])

    def add_neighbor(self, neighbor, weight=0):
        self.neighbors[neighbor] = weight

    def get_neighbors(self):
        return self.neighbors.keys()

    def get_id(self):
        return self.id

    def get_weight(self, neighbor):
        return self.neighbors[neighbor]


class Graph:
    def __init__(self):
        self.vertices = {}
        self.num_vertices = 0

    def add_vertex(self, node):
        self.num_vertices = self.num_vertices + 1
        new_vertex = Vertex(node)
        self.vertices[node] = new_vertex
        return new_vertex

    def get_vertex(self, node):
        if node in self.vertices:
            return self.vertices[node]
        else:
            return None

    def add_edge(self, frm, to, weight=0):
        if frm not in self.vertices:
            self.add_vertex(frm)
        if to not in self.vertices:
            self.add_vertex(to)

        self.vertices[frm].add_neighbor(self.vertices[to], weight)
        self.vertices[to].add_neighbor(self.vertices[frm], weight)

    def get_vertices(self):
        return self.vertices.keys()

    def to_file(self, trans_dict, file_name):
        with open(file_name, ".txt") as f:
            for idx in range(len(trans_dict)):
                if trans_dict[idx] in self.vertices:
                    neighs = self.vertices[trans_dict[idx]].neighbors
