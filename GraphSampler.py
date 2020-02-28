from random import choice
import math
from collections import Counter
import numpy as np
import networkx as nx


class GraphSample:

    def list_sampling(self, graph, target_size, V2=False):
        # reference paper: https://content.iospress.com/articles/intelligent-data-analysis/ida163319

        sampled_nodes = [choice(list(graph.nodes))]
        sampled_edges = list()
        temp_store = [sampled_nodes[0]]
        CL = set()
        frequency = Counter()

        # node selection
        while (len(sampled_nodes) < target_size * len(graph.nodes)):
            # obtain all the neighbours
            for j in temp_store:
                neighborSet = set([n for n in graph[j]])
                frequency.update(neighborSet)
                CL = CL.union(neighborSet)
            # remove the ones we might have sample already
            CL = CL - set(sampled_nodes)
            # obtain number of nodes to sample
            k = math.ceil(math.sqrt(len(CL)))
            # reset the temporary store
            temp_store = []

            # if V2 - use more advanced probabilities to select nodes
            if (V2):
                p = list()
                for i in CL:
                    p_hat = 1 / graph.degree[i] + (1 - (1 - 1 / graph.degree[i]) ** frequency[i])
                    p.append(p_hat)
                p_total = sum(p)
                p = [x / p_total for x in p]
            else:
                p = [1 / len(CL)] * len(CL)
            # select k nodes
            random_nodes = np.random.choice(a=list(CL), size=k, replace=False, p=p)
            CL.discard(set(random_nodes))
            if (len(CL) == 0):
                CL = set()
            temp_store.extend(random_nodes)
            sampled_nodes.extend(random_nodes)

        # graph induction
        for edge in graph.edges:
            # if both nodes are in the sample, add the edge
            if (edge[0] in sampled_nodes and edge[1] in sampled_nodes):
                sampled_edges.append(edge)

        return sampled_nodes, sampled_edges

    def gmd(self, graph, target_size, C, return_epsilon=True):
        current_node = np.random.choice(a=list(graph), size=1)[0]
        i = 1
        sampled_nodes = []
        sampled_edges = list()
        epsilon = list()
        size_graph =len(graph.nodes)
        # the novelty of this algorithm the the choice of C - the rest has been covered, see reference paper
        # http://pike.psu.edu/classes/ku/latest/ref/random-walk-vldb-2000.pdf
        # and
        # https://ieeexplore.ieee.org/document/7113345
        while (len(sampled_nodes) < target_size * size_graph):

            v = np.random.choice(a=[n for n in graph[current_node]], size=1)[0]

            epsilon.append(np.random.geometric(graph.degree[current_node] /
                                               max([graph.degree[current_node], C])))
            sampled_nodes.append(current_node)
            sampled_edges.append((current_node, v))

            current_node = v
            i += 1
        if (return_epsilon):
            return sampled_nodes, sampled_edges, epsilon
        return sampled_nodes, sampled_edges

    def rcmhrw(self, graph, target_size, alpha=0):
        # reference paper: https://ieeexplore.ieee.org/document/7113345
        current_node = np.random.choice(a=list(graph), size=1)[0]
        sampled_nodes = []
        sampled_edges = list()

        while (len(sampled_nodes) < target_size * len(graph.nodes)):
            v = np.random.choice([n for n in graph[current_node]], size=1)[0]
            q = np.random.uniform(0, 1, size=1)
            if (q <= (graph.degree(current_node) / graph.degree(v)) ** alpha):
                sampled_nodes.append(current_node)
                sampled_edges.append((current_node, v))
                current_node = v

        return sampled_nodes, sampled_edges


    def avg_clustering_rcmh(self, graph, nodes, alpha):
        # produces the average clustering for all nodes, given level = alpha
        # returns adjusted average clustering
        top = 0
        bottom = 0
        for x in nodes:
            c = nx.algorithms.clustering(graph, nodes=x)
            deg = nx.degree(graph, x)
            top += c * deg ** (alpha - 1)
            bottom += deg ** (alpha - 1)
        return top/bottom

    def avg_clustering_gmd(self, graph, nodes, eps, C):
        # produces the average clustering for all nodes, given level = C and geometric solutions = eps
        # returns adjusted average clustering
        top = 0
        bottom = 0
        for x in range(len(nodes)):
            c = nx.algorithms.clustering(graph, nodes=nodes[x])
            deg = nx.degree(graph, nodes[x])
            top += c * eps[x] / np.maximum(deg, C)
            bottom += eps[x] / np.maximum(deg, C)
        return top/bottom