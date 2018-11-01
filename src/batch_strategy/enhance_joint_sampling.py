'''
assume neighbor_size is 1,i.e each label length is 1
id of nodes here are results of map[]
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
    def __init__(self, G, num_neg_new, mapp, rmapp, num_modify, params = None):
        self.edges = []
        probs_in = []
        probs_out = []
        n = G.number_of_nodes()
        self.num_neg_new=num_neg_new
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

        for u in xrange(n - num_neg_new, n):
            for v in G[u]:
                probs_in.append(G[u][v]['weight'])
                probs_out.append(G[v][u]['weight'])
                if v >= len(mapp):
                    self.edges.append((u, v))
                else:
                    self.edges.append((u, mapp[v]))
        self.sampling_handler_in = at(probs_in)
        self.sampling_handler_out = at(probs_out)

    def get_batch(self, batch_size):
        node_num = G.number_of_nodes()
        batch_labels_in = []
        batch_labels_out = []
        batch_x_in = []
        batch_x_out = []
        batch_labels_in_neg= []
        batch_labels_out_neg = []
        batch_x_in_neg = []
        batch_x_out_neg = []
        for _ in xrange(batch_size):
            idx = self.sampling_handler_in.sample()
            tmp_x_in=self.edges[idx][0]
            tmp_labels_in=[self.edges[idx][1]]
            if tmp_labels_in[0]>=node_num-self.num_neg_new:
                batch_x_in_neg.append(tmp_x_in)
                batch_labels_in_neg.append(tmp_labels_in)
            else:
                batch_x_in.append(tmp_x_in)
                batch_labels_in.append(tmp_labels_in)
            idx = self.sampling_handler_out.sample()
            tmp_x_out=self.edges[idx][1]
            tmp_labels_out=[self.edges[idx][0]]
            if tmp_x_out>=node_num-self.num_neg_new:
                batch_x_out_neg.append(tmp_x_out)
                batch_labels_out_neg.append(tmp_labels_out)
            else:
                batch_x_out.append(tmp_x_out)
                batch_labels_out.append(tmp_labels_out)

        return batch_x_in, batch_x_out, batch_labels_in, batch_labels_out,batch_x_in_neg, batch_x_out_neg, batch_labels_in_neg, batch_labels_out_neg

