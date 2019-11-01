# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import networkx as nx
import numpy as np


def global_bfs(graph: nx.Graph, source: int) -> np.ndarray:
    """ Performs a BFS on the given graph.

    Args:
        graph: networkx graph on which BFS should be performed.
        source: index of node which should be used as starting point of BFS.

    Returns:
        np.ndarray with nodes sorted recording to the result of the BFS.
    """
    tree = nx.bfs_tree(graph, source)
    return np.array(list(tree.nodes))
