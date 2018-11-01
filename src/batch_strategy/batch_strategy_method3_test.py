'''
method 2
batch from delta edge
delta edge is about delta node
attention: edge of sample edges is "large_node-small_node"
'''

import sys
import os
import re
import networkx as nx
import random
import numpy as np

from alias_table_sampling import AliasTable as at

class BatchStrategy(object):
    # G is a DiGraph with edge weights
    def __init__(self, G, num_to_del, mapp, rmapp, num_modify, params = None):
        self.num_to_del = num_to_del
        self.edges = []
        self.G = G
        probs_in = []
        probs_out = []
        n = G.number_of_nodes()
        num_remain = n - num_to_del
        for i in xrange(num_modify):
            idx = len(rmapp) - i - 1
            u = rmapp[idx]
            for v in G[u]:
                probs_in.append(G[u][v]['weight'])
                probs_out.append(G[v][u]['weight'])
                if v >= len(mapp):
                    self.edges.append((idx, v))
                else:
                    self.edges.append((idx, mapp[v]))

        for u in xrange(n - num_to_del, n):
            for v in G[u]:
                probs_in.append(G[u][v]['weight'])
                probs_out.append(G[v][u]['weight'])
                if v >= len(mapp):
                    self.edges.append((u, v))
                else:
                    self.edges.append((u, mapp[v]))
        probs_in_pos = []
        probs_in_neg = []
        probs_out_pos = []
        probs_out_neg = []
        self.edges_pos = []
        self.edges_neg = []
        delta_edge_num = len(self.edges)
        for eid in range(delta_edge_num):
            tmp_edge = self.edges[eid]
            if tmp_edge[0] < num_remain and tmp_edge[1] < num_remain:
                self.edges_pos.append(tmp_edge)
                probs_in_pos.append(probs_in[eid])
                probs_out_pos.append(probs_out[eid])
            elif tmp_edge[0] >= num_remain and tmp_edge[1] >= num_remain:
                continue
            else:
                self.edges_neg.append(tmp_edge)
                probs_in_neg.append(probs_in[eid])
                probs_out_neg.append(probs_out[eid])

        self.sampling_handler_in_pos = at(probs_in_pos)
        self.sampling_handler_in_neg = at(probs_in_neg)
        self.sampling_handler_out_pos = at(probs_out_pos)
        self.sampling_handler_out_neg = at(probs_out_neg)
        # self.sampling_handler_in = at(probs_in)
        # self.sampling_handler_out = at(probs_out)

    def get_batch(self, batch_size):
        node_num = self.G.number_of_nodes()
        delta_edge_pos_total_num = len(self.edges_pos)
        delta_edge_neg_total_num = len(self.edges_neg)
        delta_edge_total_num = delta_edge_pos_total_num + delta_edge_neg_total_num
        batch_size_pos = batch_size * delta_edge_pos_total_num / delta_edge_total_num + 1
        batch_size_neg = batch_size * delta_edge_neg_total_num / delta_edge_total_num + 1
        batch_labels_in_pos = []
        batch_labels_out_pos = []
        batch_x_in_pos = []
        batch_x_out_pos = []
        batch_labels_in_neg= []
        batch_labels_out_neg = []
        batch_x_in_neg = []
        batch_x_out_neg = []
        for a in xrange(batch_size_pos):
            e_id = self.sampling_handler_in_pos.sample()
            batch_x_in_pos.append(self.edges_pos[e_id][0])
            batch_labels_in_pos.append([self.edges_pos[e_id][1]])
            e_id = self.sampling_handler_out_pos.sample()
            batch_x_out_pos.append(self.edges_pos[e_id][1])
            batch_labels_out_pos.append([self.edges_pos[e_id][0]])

        for a in xrange(batch_size_neg):
            e_id = self.sampling_handler_in_neg.sample()
            batch_x_in_neg.append(self.edges_neg[e_id][0])
            batch_labels_in_neg.append([self.edges_neg[e_id][1]])
            e_id = self.sampling_handler_out_neg.sample()
            batch_x_out_neg.append(self.edges_neg[e_id][1])
            batch_labels_out_neg.append([self.edges_neg[e_id][0]])

        # for a in xrange(batch_size):
        #     # print("get_batch loop"+str(a)+":")
        #     idx = self.sampling_handler_in.sample()
        #     # print("sampling_handler_in edge:", self.edges[idx])
        #     tmp_x_in=self.edges[idx][0]
        #     tmp_labels_in=[self.edges[idx][1]]
        #     if tmp_x_in >= node_num-self.num_to_del:
        #         batch_x_in_neg.append(tmp_x_in)
        #         batch_labels_in_neg.append(tmp_labels_in)
        #     else:
        #         batch_x_in.append(tmp_x_in)
        #         batch_labels_in.append(tmp_labels_in)
        #     idx = self.sampling_handler_out.sample()
        #     tmp_x_out=self.edges[idx][1]
        #     tmp_labels_out=[self.edges[idx][0]]
        #     if tmp_labels_out[0] >= node_num-self.num_to_del:
        #         batch_x_out_neg.append(tmp_x_out)
        #         batch_labels_out_neg.append(tmp_labels_out)
        #     else:
        #         batch_x_out.append(tmp_x_out)
        #         batch_labels_out.append(tmp_labels_out)

        return batch_x_in_pos, batch_x_out_pos, batch_labels_in_pos, batch_labels_out_pos,batch_x_in_neg, batch_x_out_neg, batch_labels_in_neg, batch_labels_out_neg

