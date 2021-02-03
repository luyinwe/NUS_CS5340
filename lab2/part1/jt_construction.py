import numpy as np
import networkx as nx
from networkx.algorithms import tree
from factor import Factor
from factor_utils import factor_product
import itertools
from collections import defaultdict


""" ADD HELPER FUNCTIONS HERE (IF NEEDED) """
class Kruskal_Graph:
    def __init__(self, V):
        self.V = V
        self.graph = []

    def addEdge(self, weight, u, v):
        self.graph.append([weight, u, v])

    def find(self, parent, node_i):
        if parent[node_i] == node_i:
            return node_i
        return self.find(parent, parent[node_i])

    def union(self, parent, rank, node_x, node_y):
        xroot = self.find(parent, node_x)
        yroot = self.find(parent, node_y)

        if rank[xroot] < rank[yroot]:
            parent[xroot] = yroot
        elif rank[xroot] > rank[yroot]:
            parent[yroot] = xroot

        else:
            parent[yroot] = xroot
            rank[xroot] += 1

    def KruskalMST(self):
        result = []
        i = 0
        e = 0

        self.graph = sorted(self.graph, key=lambda item: item[0], reverse = True)

        parent = []
        rank = []

        for node in range(self.V):
            parent.append(node)
            rank.append(0)

        while e < self.V - 1:

            weight, u, v = self.graph[i]
            i = i + 1
            x = self.find(parent, u)
            y = self.find(parent, v)

            if x != y:
                e = e + 1
                result.append([u, v])
                self.union(parent, rank, x, y)

        return result

""" END ADD HELPER FUNCTIONS HERE """


def _get_clique_factors(jt_cliques, factors):
    """
    Assign node factors to cliques in the junction tree and derive the clique factors.

    Args:
        jt_cliques: list of junction tree maximal cliques e.g. [[x1, x2, x3], [x2, x3], ... ]
        factors: list of factors from the original graph

    Returns:
        list of clique factors where the factor(jt_cliques[i]) = clique_factors[i]
    """
    clique_factors = [Factor() for _ in jt_cliques]

    """ YOUR CODE HERE """
    for i in range(len(clique_factors)):
        jt_clique = jt_cliques[i]
        temp_factor = factors.copy()
        for factor in temp_factor:
            ret = list(set(jt_clique).intersection(list(factor.var)))
            if sorted(ret) == sorted(list(factor.var)):
                clique_factors[i] = factor_product(clique_factors[i],factor)
                factors.remove(factor)
    """ END YOUR CODE HERE """

    assert len(clique_factors) == len(jt_cliques), 'there should be equal number of cliques and clique factors'
    return clique_factors


def _get_jt_clique_and_edges(nodes, edges):
    """
    Construct the structure of the junction tree and return the list of cliques (nodes) in the junction tree and
    the list of edges between cliques in the junction tree. [i, j] in jt_edges means that cliques[j] is a neighbor
    of cliques[i] and vice versa. [j, i] should also be included in the numpy array of edges if [i, j] is present.
    You can use nx.Graph() and nx.find_cliques().

    Args:
        nodes: numpy array of nodes [x1, ..., xN]
        edges: numpy array of edges e.g. [x1, x2] implies that x1 and x2 are neighbors.

    Returns:
        list of junction tree cliques. each clique should be a maximal clique. e.g. [[X1, X2], ...]
        numpy array of junction tree edges e.g. [[0,1], ...], [i,j] means that cliques[i]
            and cliques[j] are neighbors.
    """
    jt_cliques = []
    jt_edges = np.array(edges)  # dummy value

    """ YOUR CODE HERE """
    G = nx.Graph()
    for i in nodes:
        G.add_node(i)
    G.add_edges_from(edges)
    jt_cliques = list(nx.find_cliques(G))

    ## maximum spanning tree
    K_G = Kruskal_Graph(len(jt_cliques))
    for i in range(len(jt_cliques)):
        for j in range(len(jt_cliques)):
            if i>=j:
                continue
            else:
                weight = len(list(set(jt_cliques[i]).intersection(jt_cliques[j])))
                if weight>=0:
                    K_G.addEdge(weight, i, j)
    jt_edges = K_G.KruskalMST()
    """ END YOUR CODE HERE """

    return jt_cliques, jt_edges


def construct_junction_tree(nodes, edges, factors):
    """
    Constructs the junction tree and returns its the cliques, edges and clique factors in the junction tree.
    DO NOT EDIT THIS FUNCTION.

    Args:
        nodes: numpy array of random variables e.g. [X1, X2, ..., Xv]
        edges: numpy array of edges e.g. [[X1,X2], [X2,X1], ...]
        factors: list of factors in the graph

    Returns:
        list of cliques e.g. [[X1, X2], ...]
        numpy array of edges e.g. [[0,1], ...], [i,j] means that cliques[i] and cliques[j] are neighbors.
        list of clique factors where jt_cliques[i] has factor jt_factors[i] where i is an index
    """
    jt_cliques, jt_edges = _get_jt_clique_and_edges(nodes=nodes, edges=edges)
    jt_factors = _get_clique_factors(jt_cliques=jt_cliques, factors=factors)
    return jt_cliques, jt_edges, jt_factors
