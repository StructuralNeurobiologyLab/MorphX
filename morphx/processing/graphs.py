# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import networkx as nx
import numpy as np
from collections import defaultdict, deque


def global_bfs_dist(g: nx.Graph, min_dist: float, source=-1) -> np.ndarray:
    """ Performs a BFS on a weighted graph. Only nodes with a minimum distance to other added nodes in their
    neighborhood get added to the final BFS result. This way, the graph can be split into subsets of approximately equal
    size based on the output of this method.

    Args:
        g: The weighted networkx graph on which the BFS should be performed. Weights must be accessible
            by g[a][b]['weight'] for the edge from node a to node b.
        source: The source node from which the BFS should start. Default is -1 which stands for a random node
        min_dist: The minimum distance between nodes in the BFS result.

    Returns:
        np.ndarray with nodes sorted recording to the result of the filtered BFS
    """
    if source == -1:
        source = np.random.randint(g.number_of_nodes())

    visited = [source]
    chosen = [source]

    # add all neighbors with respective weights
    neighbors = g.neighbors(source)
    de = deque([(i, g[source][i]['weight']) for i in neighbors])
    while de:
        curr, weight = de.pop()
        if curr not in visited:
            visited.append(curr)

            # only nodes with minimum distance to other chosen nodes get added
            # TODO: Does euclidic distance restriction make sense?
            if weight >= min_dist:
                chosen.append(curr)
                weight = 0

            # add all neighbors with their weights added to the current weight
            neighbors = g.neighbors(curr)
            de.extendleft([(i, g[curr][i]['weight'] + weight) for i in neighbors if i not in visited])

    # return only chosen nodes
    return np.array(chosen)


def local_bfs_num(g: nx.Graph, source: int, num: int, mapping: defaultdict) -> np.ndarray:
    """ Performs a BFS on the given graph until the number of corresponding mesh vertices is larger
    than the given threshold.

    Args:
        g: The networkx graph on which the BFS should be performed.
        source: The source node from which the BFS should start.
        num: The threshold for the number of vertices included in the current BFS result.
        mapping: The mapping dict between skeleton nodes and vertices.

    Returns:
        np.ndarray with nodes sorted recording to the result of the limited BFS.
    """
    visited = [source]
    total = len(mapping[source])
    de = deque(list(nx.neighbors(g, source)))
    while total < num:
        if len(de) != 0:
            curr = de.pop()
            if curr not in visited:
                total += len(mapping[curr])
                visited.append(curr)
                de.extendleft(list(nx.neighbors(g, curr)))
        else:
            break

    return np.array(visited)


def local_bfs_dist(g: nx.Graph, source: int, max_dist: int) -> np.ndarray:
    """ Performs a BFS on a weighted graph until maximum distance for each path is reached.

    Args:
        g: The weighted networkx graph on which the BFS should be performed. Weights must be accessible
            by g[a][b]['weight'] for the edge from node a to node b.
        source: The source node from which the BFS should start.
        max_dist: The maximum distance (same unit as weights) which should limit the BFS.

    Returns:
        np.ndarray with nodes sorted recording to the result of the limited BFS
    """
    visited = [source]
    neighbors = g.neighbors(source)
    de = deque([(i, g[source][i]['weight']) for i in neighbors if g[source][i]['weight'] <= max_dist])
    while de:
        curr, weight = de.pop()
        if curr not in visited:
            visited.append(curr)
            neighbors = g.neighbors(curr)
            de.extendleft([(i, g[curr][i]['weight'] + weight)
                           for i in neighbors if g[curr][i]['weight'] + weight <= max_dist and i not in visited])

    return np.array(visited)
