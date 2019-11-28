# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import networkx as nx
from morphx.data.cloudset import CloudSet
from morphx.processing import graphs


def test_traversion():
    wd = os.path.expanduser('~/gt/gt_results/')
    min_dist = 10000
    source = 0
    data = CloudSet(wd, min_dist, 1000, global_source=source)

    graph = data.curr_hybrid.graph()
    chosen = graphs.global_bfs_dist(graph, min_dist*data.radius_factor, source=source)

    traverser = data.curr_hybrid.traverser()
    print(traverser)

    for item in traverser:
        assert item in chosen

    for item in traverser:
        if item != source:
            path = nx.shortest_path(graph, source=source, target=item, weight='weight')
            dist = 0
            for i in range(len(path) - 1):
                weight = graph[path[i]][path[i + 1]]['weight']
                dist += weight
            assert dist > min_dist * data.radius_factor
