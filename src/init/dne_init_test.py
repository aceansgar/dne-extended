'''
create all nodes in load_data
'''
import sys
import os
import json
import numpy as np
import time
import datetime

from utils.env import *
from utils.data_handler import DataHandler as dh

def init(params, metric, output_path, draw):
    print("dne_init begin")
    # load graph structure
    def load_data(params):
        params["network_file"] = os.path.join(DATA_PATH, params["network_file"])
        G = getattr(dh, params["func"])(params)
        return G

    time_path = output_path + "_time"

    G = load_data(params["load_data"])
    # print
    # print("all nodes:", G.nodes())
    # print
    # print("all edges:", G.edges())
    # print

    module_embedding = __import__(
            "init_embedding." + params["init_train"]["func"], fromlist = ["init_embedding"]).NodeEmbedding
    ne = module_embedding(params["init_train"], G)

    st = datetime.datetime.now()
    embeddings, weights = ne.train()
    ed = datetime.datetime.now()
    dh.append_to_file(time_path, str(ed - st) + "\n")
    with open(output_path + "_init", "w") as f:
        f.write(json.dumps({"embeddings": embeddings.tolist(), "weights": weights.tolist()}))
    print("dne_init metric begin")
    metric(embeddings,0)
    print("dne_init_metric done")
    draw(embeddings)
    print("dne_init done")


    return G, embeddings, weights

