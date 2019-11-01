# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

from collections import defaultdict, deque
import networkx as nx
import numpy as np


def global_bfs(g: nx.Graph, source: int) -> np.ndarray:
    """ Performs a BFS on the given graph.

    Args:
        g: networkx graph on which BFS should be performed.
        source: index of node which should be used as starting point of BFS.

    Returns:
        np.ndarray with nodes sorted recording to the result of the BFS.
    """
    tree = nx.bfs_tree(g, source)
    return np.array(list(tree.nodes))


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
    neighbors = list(nx.neighbors(g, source))
    try:
        weights = [g[source][i]['weight'] for i in neighbors]
    except KeyError:
        raise Exception("edge without weight detected.")
        return np.array([])
    de = deque([(neighbors[i], weights[i])
                for i in range(len(neighbors)) if weights[i] <= max_dist])
    while de:
        curr, weight = de.pop()
        if curr not in visited:
            visited.append(curr)
            neighbors = list(nx.neighbors(g, curr))
            weights = [g[curr][i]['weight'] + weight for i in neighbors]
            de.extendleft([(neighbors[i], weights[i])
                           for i in range(len(neighbors)) if weights[i] <= max_dist])

    return np.array(visited)
