import networkx as nx
import os
import sys


class GetNext(object):
    def __init__(self, params):
        print("get_next init begin")
        self.f = open(params["input_file"], "r")
        self.is_directed = params["is_directed"]
        self.num_to_del = params["num_to_del"]
        self.num_at_least = params["num_at_least"]
        self.get_next_times=params["get_next_times"]
        print("get_next init done")

    @staticmethod
    def dict_del(d, key, num_del):
        # print("d:",d,"key:",key,"value:",d[key])
        if key in d:
            d[key] -= num_del
        else:
            d[key] -= num_del

    @staticmethod
    def dict_add(d, key, add):
        if key in d:
            d[key] += add
        else:
            d[key] = add

    def get_next(self, G):
        print("get_next function begin")
        # self.get_next_times-=1
        # # if self.get_next_times<0:
        #     return 0
        num_nodes_pre = G.number_of_nodes()
        num_to_del = self.num_to_del

        for num in xrange(num_to_del):
            print("num_to_del loop num:",num)
            line = self.f.readline()
            if not line:
                return num
            line = line.strip()
            if len(line) == 0:
                line = self.f.readline()
                if not line:
                    return num
                line = line.strip()
            node_to_del, connected_nodes_num = [int(i) for i in line.split()]

            for i in xrange(connected_nodes_num):
                line = self.f.readline()
                line = line.strip()
                u, v = [int(i) for i in line.split()]
                # print("node id:" + str(u) + " its out_degree is")
                GetNext.dict_del(G.node[u], 'out_degree', 1)
                # print("node id:" + str(v) + " its in_degree is")
                GetNext.dict_del(G.node[v], 'in_degree', 1)
                G.graph['degree'] -= 1
                #G.remove_edge(u,v)
                if not self.is_directed and u != v:
                    # print("node id:" + str(v) + " its out_degree is")
                    GetNext.dict_del(G.node[v], 'out_degree', 1)
                    # print("node id:" + str(u) + " its in_degree is")
                    GetNext.dict_del(G.node[u], 'in_degree', 1)
                    #G.remove_edge(v,u)
                    G.graph['degree'] -= 1
            G.remove_node(node_to_del)

        print("get_next function done")
        return num_to_del

    # def get_next(self, G):
    #     for num in xrange(self.n):
    #         while True:
    #             line = self.f.readline()
    #             if not line:
    #                 return num
    #             line = line.strip()
    #             if len(line) == 0:
    #                 continue
    #             u, m = [int(i) for i in line.split()]
    #             break
    #
    #         for i in xrange(m):
    #             line = self.f.readline()
    #             line = line.strip()
    #             u, v = [int(i) for i in line.split()]
    #             G.add_edge(u, v, weight=1)
    #             GetNext.dict_add(G.node[u], 'out_degree', 1)
    #             GetNext.dict_add(G.node[v], 'in_degree', 1)
    #             GetNext.dict_add(G.graph, 'degree', 1)
    #             if not self.is_directed and u != v:
    #                 G.add_edge(v, u, weight=1)
    #                 GetNext.dict_add(G.node[v], 'out_degree', 1)
    #                 GetNext.dict_add(G.node[u], 'in_degree', 1)
    #                 G.graph['degree'] += 1
    #
    #     return self.n

    def __del__(self):
        self.f.close()

