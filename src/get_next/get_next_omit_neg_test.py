'''
only to define how much nodes to delete,
and change in-degree and out-degree because distribution in f_in and f_out only includes remains,
not delete graph nodes or edges because we need to get batch from delta edges in l1,l2.l3.
in method 2
do not change the graph
consider skipped nodes
'''
import networkx as nx
import os
import sys


class GetNext(object):
    def __init__(self, params):

        print("get_next init begin")
        self.f = open(params["input_file"], "r")
        self.is_directed = params["is_directed"]
        self.n = params["num_to_del"]
        self.num_at_least = params["num_at_least"]
        print("get_next init done")

    @staticmethod
    def dict_del(d, key, num_del):
        # print("d:",d,"key:",key,"value:",d[key])
        if key in d:
            d[key] -= num_del
        else:
            d[key] -= num_del

    def get_next(self, G):
        print("get_next function begin")
        # print
        # for nd in G.nodes():
        #     node_attr = G.node[nd]
        #     print("node_id:",str(nd), node_attr)
        #     if 'in_degree' in node_attr:
        #         tmp_in_degree = node_attr['in_degree']
        #
        #         if tmp_in_degree < 0:
        #             print("nodeid:" + str(nd) + "in_degree is negative" + str(tmp_in_degree))
        #     else:
        #         print("nodeid:" + str(nd) + " in_degree not exist")
        #     if 'out_degree' in node_attr:
        #         tmp_out_degree = node_attr['out_degree']
        #
        #         if tmp_out_degree < 0:
        #             print("nodeid:" + str(nd) + "out_degree is negative" + str(tmp_out_degree))
        #     else:
        #         print("nodeid:" + str(nd) + " out_degree not exist")
        # print
        num_nodes_pre = G.number_of_nodes()
        num_to_del = self.n

        for num in xrange(num_to_del):
            print("num_to_del loop read file, num:",num)
            line = self.f.readline()
            if not line:
                return num
            line = line.strip()
            if len(line) == 0:
                line = self.f.readline()
                if not line:
                    return num
                line = line.strip()
            u, m = [int(i) for i in line.split()]

            for i in xrange(m):
                line = self.f.readline()
                line = line.strip()
                u, v = [int(i) for i in line.split()]
                # print("node id:" + str(u) + " its out_degree is")
                GetNext.dict_del(G.node[u], 'out_degree', 1)
                # print("node id:" + str(v) + " its in_degree is")
                GetNext.dict_del(G.node[v], 'in_degree', 1)
                G.graph['degree'] -= 1
                if not self.is_directed and u != v:
                    # print("node id:" + str(v) + " its out_degree is")
                    GetNext.dict_del(G.node[v], 'out_degree', 1)
                    # print("node id:" + str(u) + " its in_degree is")
                    GetNext.dict_del(G.node[u], 'in_degree', 1)
                    G.graph['degree'] -= 1
        print("get_next function done")
        return num_to_del

    def __del__(self):
        self.f.close()

