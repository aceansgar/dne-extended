'''
method 4
'''

import sys
import os
import json
import numpy as np
import time
import datetime
import random
from Queue import PriorityQueue as pq

from utils.env import *
from utils.data_handler import DataHandler as dh
from alias_table_sampling import AliasTable as at


def sigmoid(x):
    s=1.0/(1.0+1.0/np.exp(x))

    return s



def loop(params, G, embeddings, weights, metric, output_path, draw):
    print
    print("dynamic loop begin")
    params["get_next"]["input_file"] = os.path.join(DATA_PATH, params["get_next"]["input_file"])
    module_next = __import__(
        "get_next." + params["get_next"]["func"], fromlist=["get_next"]).GetNext
    gn = module_next(params["get_next"])

    mapp = range(G.number_of_nodes())
    rmapp = range(G.number_of_nodes())

    params_dynamic = params["dynamic_embedding"]
    K = params_dynamic["num_sampled"]

    # def cal_delta(num_to_del):  # rank edges
    #     num_pre = G.number_of_nodes()
    #     num_remain = num_pre - num_to_del
    #     for u, v in G.edges():
    #         if u >= num_remain or v >= num_remain:
    #             continue
    #         um, vm = mapp[u], mapp[v]
    #         delta_real = np.matmul(embeddings[[um]], weights[[vm]].T)[0, 0]
    #         tmp_degree = G.node[u]['in_degree'] * G.node[v]['out_degree']
    #         if tmp_degree == 0:
    #             G[u][v]['delta'] = np.log(K) - delta_real
    #         else:
    #             G[u][v]['delta'] = np.log(
    #                 float(G[u][v]['weight'] * G.graph['degree']) / float(
    #                     G.node[u]['in_degree'] * G.node[v]['out_degree'])) + np.log(K) - delta_real

    def rank_nodes(num_to_del):
        num_remain = G.number_of_nodes() - num_to_del
        num_modify = params_dynamic['num_modify']
        # if num_modify == 0:
        #     return
        # delta_list = [0.0] * num_remain
        # for u, v in G.edges():
        #     if u >= num_remain or v >= num_remain:
        #         continue
        #     delta_list[u] += float(G[u][v]['weight']) * abs(G[u][v]['delta'])
        #     delta_list[v] += float(G[u][v]['weight']) * abs(G[u][v]['delta'])

        in_degree = []
        out_degree = []
        node_degree = {}
        nodes_deltas = {}
        for nd in xrange(num_remain):
            nodes_deltas[nd] = []
        for nd in xrange(num_remain):
            in_degree.append(G.node[nd]['in_degree'])
            out_degree.append(G.node[nd]['out_degree'])
            node_degree[nd] = G.node[nd]['in_degree'] + G.node[nd]['out_degree']
        path_num = 2 * num_to_del
        sampling_handler_in = at(in_degree)
        sampling_handler_out = at(out_degree)
        path_id = 0

        while path_id < path_num:
            rem_node1 = sampling_handler_in.sample()
            tmp_in_degree = in_degree[rem_node1]
            if tmp_in_degree == 0:
                continue
            next_nodes = []
            for x in G[rem_node1]:
                if x >= num_remain:
                    next_nodes.append(x)
            len_next_nodes = len(next_nodes)
            if len_next_nodes == 0:
                continue
            next_node_pos = random.randrange(len_next_nodes)
            del_node = next_nodes[next_node_pos]
            next_nodes = []
            for x in G[del_node]:
                if x < num_remain:
                    next_nodes.append(x)
            len_next_nodes = len(next_nodes)
            if len_next_nodes == 0:
                continue
            next_node_pos = random.randrange(len_next_nodes)
            rem_node2 = next_nodes[next_node_pos]
            tmp_delta1 = np.matmul(embeddings[[rem_node1]],weights[[del_node]].T)[0,0]
            tmp_delta2 = np.matmul(weights[[rem_node2]],embeddings[[del_node]].T)[0,0]
            tmp_delta = np.maximum(0,tmp_delta1) * np.maximum(0,tmp_delta2)
            nodes_deltas[rem_node1].append(tmp_delta)
            nodes_deltas[rem_node2].append(tmp_delta)
            path_id += 1

        path_id = 0
        while path_id < path_num:
            rem_node1 = sampling_handler_out.sample()
            tmp_out_degree = out_degree[rem_node1]
            if tmp_out_degree == 0:
                continue
            next_nodes = []
            for x in G[rem_node1]:
                if x >= num_remain:
                    next_nodes.append(x)
            len_next_nodes = len(next_nodes)
            if len_next_nodes == 0:
                continue
            next_node_pos = random.randrange(len_next_nodes)
            del_node = next_nodes[next_node_pos]
            next_nodes = []
            for x in G[del_node]:
                if x < num_remain:
                    next_nodes.append(x)
            len_next_nodes = len(next_nodes)
            if len_next_nodes == 0:
                continue
            next_node_pos = random.randrange(len_next_nodes)
            rem_node2 = next_nodes[next_node_pos]
            tmp_delta1 = np.matmul(weights[[rem_node1]], embeddings[[del_node]].T)[0, 0]
            tmp_delta2 = np.matmul(embeddings[[rem_node2]], weights[[del_node]].T)[0, 0]
            tmp_delta = np.maximum(0, tmp_delta1) * np.maximum(0, tmp_delta2)
            nodes_deltas[rem_node1].append(tmp_delta)
            nodes_deltas[rem_node2].append(tmp_delta)
            path_id += 1

        delta_list = []
        for u in xrange(num_remain):
            delta_list.append(sum(nodes_deltas[u]))

        for u in G:
            if u >= num_remain:
                continue
            tmp_z = G.node[u]['in_degree'] + G.node[u]['out_degree']
            if tmp_z == 0:
                delta_list[u] = 0
            else:
                delta_list[u] /= tmp_z

        q = pq()
        for u in G:
            if u >= num_remain:
                continue
            if q.qsize() < num_modify:
                q.put_nowait((delta_list[u], u))
                continue
            items = q.get_nowait()
            if items[0] < delta_list[u]:
                q.put_nowait((delta_list[u], u))
            else:
                q.put_nowait(items)

        idx = num_remain - 1
        while not q.empty():
            u = q.get_nowait()[1]
            um = mapp[u]
            v = rmapp[idx]
            mapp[u] = idx
            rmapp[idx] = u
            mapp[v] = um
            rmapp[um] = v
            embeddings[[um, idx], :] = embeddings[[idx, um], :]
            weights[[um, idx], :] = weights[[idx, um], :]
            idx = idx - 1

    def reset(num_to_del):
        num_modify = params_dynamic['num_modify']
        num_remain = G.number_of_nodes() - num_to_del
        embeddings[[range(num_remain)], :] = embeddings[rmapp, :]
        weights[[range(num_remain)], :] = weights[rmapp, :]

    module_dynamic_embedding = __import__(
        "dynamic_embedding." + params_dynamic["func"],
        fromlist=["dynamic_embedding"]).NodeEmbedding

    time_path = output_path + "_time"

    # print
    # for nd in G.nodes():
    #     node_attr = G.node[nd]
    #     print(node_attr)
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

    dynamic_embeddings = []
    while True:
        print("\n")
        print("dynamic loop new loop begin, to delete some nodes")
        # print("all former nodes:", G.nodes())
        # print("all former edges:", G.edges())
        num_to_del = gn.get_next(G)
        print("dynamic loop new loop get_next done,num to del is:", num_to_del)
        num_init = G.number_of_nodes()
        print("G num_init:", num_init)
        num_remain = G.number_of_nodes() - num_to_del
        mapp = range(num_remain)
        rmapp = range(num_remain)
        # print
        # print("original mapp:", mapp)
        # print("original rmapp", rmapp)
        # print("dynamic loop new loop original map done")

        if num_to_del == 0:
            break

        rank_nodes(num_to_del)
        print("dynamic loop new loop rank nodes to be modified done")
        # print("mapp:", mapp)
        # print("rmapp:", rmapp)
        ne = module_dynamic_embedding(params_dynamic, embeddings, weights, G, mapp, rmapp, num_to_del)
        print("dynamic loop new loop    init dne object done")

        st = datetime.datetime.now()
        embeddings, weights = ne.train()
        print("dynamic loop new loop train dne object done")
        #modify
        ed = datetime.datetime.now()
        dh.append_to_file(time_path, str(ed - st) + "\n")

        reset(num_to_del)
        print("dynamic loop new loop reset embedding array done")
        embeddings = embeddings[:num_remain]
        weights = weights[:num_remain]
        print("dynamic loop new loop truncate embedding array done")
        for nodeid in xrange(num_remain,num_init):
            G.remove_node(nodeid)
        print("dynamic loop new loop delete nodes from graph done")
        print("dynamic loop get metric:")
        res = metric(embeddings)
        print("dynamic loop get metric end")
        draw(embeddings)
        print("dynamic loop draw embed done")
        dynamic_embeddings.append({"embeddings": embeddings.tolist(), "weights": weights.tolist()})
        print("dynamic loop dynamic_embeddings append done")
    print("dynamic loop end")

    with open(output_path + "_dynamic", "w") as f:
        f.write(json.dumps(dynamic_embeddings))
    print("dynamic loop return")
