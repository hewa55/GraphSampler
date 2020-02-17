from random import choice
import math
from collections import Counter
import numpy as np
import networkx as nx


class GraphSample:

    def list_sampling(self, graph, target_size, V2=False):

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

    def sample(self, graph, method,n):
        if(method == "list1"):
            return self.list_sampling(graph,target_size=n,V2=False)
        if(method == "list2"):
            return self.list_sampling(graph,target_size=n,V2=True)



