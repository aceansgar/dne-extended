'''
only to define how much nodes to delete,
and change in-degree and out-degree because distribution in f_in and f_out only includes remains,
not delete graph nodes or edges because we need to get batch from delta edges in l1,l2.l3.
in method 2
do not change the graph
'''
import networkx as nx
import os
import sys


class GetNext(object):
    def __init__(self, params):
        self.f = open(params["input_file"], "r")
        self.is_directed = params["is_directed"]
        self.n = params["num_to_del"]
        self.num_at_least = params["num_at_least"]

    @staticmethod
    def dict_del(d, key, num_del):
        if key in d:
            d[key] -= num_del
        else:
            d[key] -= num_del

    def get_next(self, G):
        num_nodes_pre = G.number_of_nodes()
        num_to_del = self.n
        for num in xrange(num_to_del):

            line = self.f.readline()
            if not line:
                return num
            line = line.strip()
            if len(line) == 0:
                continue
            u, m = [int(i) for i in line.split()]

            for i in xrange(m):
                line = self.f.readline()
                line = line.strip()
                u, v = [int(i) for i in line.split()]
                GetNext.dict_del(G.node[u], 'out_degree', 1)
                GetNext.dict_del(G.node[v], 'in_degree', 1)
                GetNext.dict_del(G.graph, 'degree', 1)
                if not self.is_directed and u != v:
                    GetNext.dict_del(G.node[v], 'out_degree', 1)
                    GetNext.dict_del(G.node[u], 'in_degree', 1)
                    G.graph['degree'] -= 1

        return num_to_del

    def __del__(self):
        self.f.close()

