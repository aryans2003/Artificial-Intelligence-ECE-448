# Utility functions for computing the Minimum Spanning Tree of a set of objectives, 
# which is a useful heuristic for the Traveling Salesman Problem.

# given a list/tuple of objectives and a distance function on those objectives
# return the weight of the Minimum Spanning Tree among those objectives
def compute_mst_cost(objectives, distance):
    mst = MST(objectives, distance)
    return mst.compute_mst_weight()

# TODO: this is not efficient because it forces us to recompute all the between goal distances each time we call MST...
class MST:
    def __init__(self, nodes, distance):
        self.elements = {key: None for key in nodes}
        self.distances   = {
                (i, j): distance(i, j)
                for i, j in self.cross(nodes)
            }
        
    # Prim's algorithm adds edges to the MST in sorted order as long as they don't create a cycle
    def compute_mst_weight(self):
        weight      = 0
        for distance, i, j in sorted((self.distances[(i, j)], i, j) for (i, j) in self.distances):
            if self.unify(i, j):
                weight += distance
        return weight

    # helper checks the root of a node, in the process flatten the path to the root
    def resolve(self, key):
        path = []
        root = key 
        while self.elements[root] is not None:
            path.append(root)
            root = self.elements[root]
        for key in path:
            self.elements[key] = root
        return root
    
    # helper checks if the two elements have the same root they are part of the same tree
    # otherwise set the root of one to the other, connecting the trees
    def unify(self, a, b):
        ra = self.resolve(a) 
        rb = self.resolve(b)
        if ra == rb:
            return False 
        else:
            self.elements[rb] = ra
            return True

    # helper that gets all pairs i,j for a list of keys
    def cross(self, keys):
        return (x for y in (((i, j) for j in keys if i < j) for i in keys) for x in y)