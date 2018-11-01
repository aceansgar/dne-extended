import sys
import os
import json
import numpy as np
import time
import datetime
from Queue import PriorityQueue as pq

from utils.env import *
from utils.data_handler import DataHandler as dh


def loop(params, G, embeddings, weights, metric, output_path, draw):
    params["get_next"]["input_file"] = os.path.join(DATA_PATH, params["get_next"]["input_file"])
    module_next = __import__(
        "get_next." + params["get_next"]["func"], fromlist=["get_next"]).GetNext
    gn = module_next(params["get_next"])

    mapp = range(G.number_of_nodes())
    rmapp = range(G.number_of_nodes())

    params_dynamic = params["dynamic_embedding"]
    K = params_dynamic["num_sampled"]

    def cal_delta(num_neg_new):  # rank edges
        num_remain = G.number_of_nodes() - 2*num_neg_new
        for u, v in G.edges():
            if u >= num_remain or v >= num_remain:
                continue
            um, vm = mapp[u], mapp[v]
            delta_real = np.matmul(embeddings[[um]], weights[[vm]].T)[0, 0]
            G[u][v]['delta'] = delta_real - np.log(
                float(G[u][v]['weight'] * G.graph['degree']) / float(
                    G.node[u]['in_degree'] * G.node[v]['out_degree'])) + np.log(K)

    def rank_nodes(num_neg_new):
        num_remain = G.number_of_nodes() - 2*num_neg_new
        num_modify = params_dynamic['num_modify']
        if num_modify == 0:
            return
        delta_list = [0.0] * num_remain
        for u, v in G.edges():
            if u >= num_remain or v >= num_remain:
                continue
            delta_list[u] += float(G[u][v]['weight']) * abs(G[u][v]['delta'])
            delta_list[v] += float(G[u][v]['weight']) * abs(G[u][v]['delta'])

        for u in G:
            if u >= num_remain:
                continue
            delta_list[u] /= (G.node[u]['in_degree'] + G.node[u]['out_degree'])

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

    def reset(num_neg_new):
        num_modify = params_dynamic['num_modify']
        num_remain = G.number_of_nodes() - 2*num_neg_new
        embeddings[[range(num_remain)], :] = embeddings[rmapp, :]
        weights[[range(num_remain)], :] = weights[rmapp, :]

    module_dynamic_embedding = __import__(
        "dynamic_embedding." + params_dynamic["func"],
        fromlist=["dynamic_embedding"]).NodeEmbedding

    time_path = output_path + "_time"
    dynamic_embeddings = []
    while True:
        mapp = range(G.number_of_nodes())
        rmapp = range(G.number_of_nodes())

        num_neg_new = gn.get_next(G)
        if num_neg_new == 0:
            break
        cal_delta(num_neg_new)
        rank_nodes(num_neg_new)
        ne = module_dynamic_embedding(params_dynamic, embeddings, weights, G, mapp, rmapp, num_neg_new)

        st = datetime.datetime.now()
        embeddings, weights = ne.train()
        ed = datetime.datetime.now()
        dh.append_to_file(time_path, str(ed - st) + "\n")

        reset(num_neg_new)
        node_num = G.number_of_nodes()
        for node_id in xrange(node_num-2*num_neg_new,node_num):
            G.remove_node(node_id)
        embeddings = embeddings[:node_num-2*num_neg_new]

        res = metric(embeddings)
        draw(embeddings)
        dynamic_embeddings.append({"embeddings": embeddings.tolist(), "weights": weights.tolist()})

    with open(output_path + "_dynamic", "w") as f:
        f.write(json.dumps(dynamic_embeddings))
