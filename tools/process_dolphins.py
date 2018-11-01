import os
import sys
import re
import json
import math
import argparse
import time
import subprocess
import numpy as np
import networkx as nx
import tensorflow as tf
import datetime
from operator import itemgetter
import collections

FILE_PATH = os.path.dirname(os.path.abspath(__file__))
FATHER_PATH = os.path.join(FILE_PATH, '..')
DATA_PATH = os.path.join(FATHER_PATH, 'data')

def main():
    dataset_name = "dolphins_nw_pre"
    dataset_path = os.path.join(DATA_PATH, dataset_name)
    later_name = "dolphins_nw"
    later_path = os.path.join(DATA_PATH, later_name)
    with open(dataset_path, 'r') as f_pre:
        with open(later_path, 'w') as f_later:
            for line in f_pre:
                line = line.strip()
                if len(line) == 0:
                    continue
                items = line.split()
                if len(items) == 1:
                    f_later.write(str(items[0]) + '\n')
                if len(items) == 2:
                    a = int(items[0])
                    b = int(items[1])
                    if a > b:
                        f_later.write(str(b) + '\t' + str(a) + '\n')
                    else:
                        f_later.write(str(a) + '\t' + str(b) + '\n')

if __name__ == "__main__":
    main()