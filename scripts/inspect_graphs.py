# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import networkx as nx
from morphx.processing import clouds

wd = os.path.expanduser('~/gt/gt_results/')
files = glob.glob(wd + '*.pkl')

for file in files:
    curr_hybrid = clouds.load_gt(file)
    graph = curr_hybrid.graph()
    for i in range(len(graph.nodes)):
        del graph.node[i]['position']

    nx.write_gexf(graph, file[:-4] + '.gexf')
