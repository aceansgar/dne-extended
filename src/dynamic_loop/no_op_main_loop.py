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
    print
    print("dynamic loop begin")
    params["get_next"]["input_file"] = os.path.join(DATA_PATH, params["get_next"]["input_file"])

    module_next = __import__(
        "get_next." + params["get_next"]["func"], fromlist=["get_next"]).GetNext
    gn = module_next(params["get_next"])
    time_path = output_path + "_time"


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

        if num_to_del == 0:
            break


        st = datetime.datetime.now()


        ed = datetime.datetime.now()
        dh.append_to_file(time_path, str(ed - st) + "\n")
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
