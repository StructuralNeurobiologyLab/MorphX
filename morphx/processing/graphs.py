# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import networkx as nx
from collections import deque


def bfs_base_points_euclid(g: nx.Graph, min_dist: float, source: int = -1) -> np.ndarray:
    """ Performs a BFS on a weighted graph. Only nodes with a minimum euclidian distance get added
        to the result. The graph nodes must contain the attribute 'position' as a numpy array with
        x,y,z position.

    Args:
        g: The weighted networkx graph on which the BFS should be performed. Nodes must have a position attribute
         with their xyz positions as a numpy array, e.g. g.nodes[0]['position'] = np.array([1,2,3])
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
    de = deque([(i, 0) for i in neighbors])
    while de:
        curr, buddy = de.pop()
        if curr not in visited:
            visited.append(curr)
            # only nodes with minimum distance to other chosen nodes get added
            buddy_pos = g.nodes[chosen[buddy]]['position']
            pos = g.nodes[curr]['position']
            if np.linalg.norm(buddy_pos - pos) >= min_dist:
                clear = True
                for node in chosen:
                    if np.linalg.norm(pos - g.nodes[node]['position']) < min_dist:
                        clear = False
                        break
                if clear:
                    buddy = len(chosen)
                    chosen.append(curr)
            # add all neighbors with their weights added to the current weight
            neighbors = g.neighbors(curr)
            de.extendleft([(i, buddy) for i in neighbors if i not in visited])
    # return only chosen nodes
    return np.array(chosen)


def bfs_euclid(g: nx.Graph, source: int, max_dist: float) -> np.ndarray:
    """ Performs a BFS on a graph until maximum euclidian distance for each path is reached. The
        graph nodes must contain the attribute 'position' as a numpy array with x,y,z position.

    Args:
        g: The weighted networkx graph on which the BFS should be performed. Nodes must have a position attribute
         with their xyz positions as a numpy array, e.g. g.nodes[0]['position'] = np.array([1,2,3])
        source: The source node from which the BFS should start.
        max_dist: The maximum distance which should limit the BFS.

    Returns:
        np.ndarray with nodes sorted recording to the result of the limited BFS
    """
    source_pos = g.nodes[source]['position']
    visited = [source]
    neighbors = g.neighbors(source)
    de = deque([i for i in neighbors if np.linalg.norm(source_pos - g.nodes[i]['position']) <= max_dist])
    while de:
        curr = de.pop()
        if curr not in visited:
            visited.append(curr)
            neighbors = g.neighbors(curr)
            de.extendleft([i for i in neighbors if np.linalg.norm(source_pos - g.nodes[i]['position']) <= max_dist
                           and i not in visited])
    return np.array(visited)


def bfs_num(g: nx.Graph, source: int, neighbor_num: int) -> np.ndarray:
    """ Performs a BFS on a graph until maximum number of visited nodes is reached.

    Args:
        g: The networkx graph on which the BFS should be performed.
        source: The source node from which the BFS should start.
        neighbor_num: The maximum number of nodes which should limit the BFS.

    Returns:
        np.ndarray with nodes sorted recording to the result of the limited BFS
    """
    visited = [source]
    neighbors = g.neighbors(source)
    de = deque([i for i in neighbors])
    while de:
        if len(visited) > neighbor_num:
            return np.array(visited)
        curr = de.pop()
        if curr not in visited:
            visited.append(curr)
            neighbors = g.neighbors(curr)
            de.extendleft([i for i in neighbors if i not in visited])
    return np.array(visited)


def bfs_iterative(g: nx.Graph, source: int, context: int):
    """ Splits the graph into chunks of nearly similar size 'context' and returns these
        chunks as a list of node lists.

    Args:
        g: The networkx graph on which the BFS should be performed.
        source: The source node from which the BFS should start.
        context: The size of the chunks. If there are leave nodes it can happen that the
            chunks are smaller than this.
    """
    chunks = []
    visited = []
    neighbors = g.neighbors(source)
    de = deque([i for i in neighbors])
    while de:
        curr = de.pop()
        if curr not in visited:
            visited.append(curr)
            local_visited = [curr]
            neighbors = g.neighbors(curr)
            local_de = deque([i for i in neighbors if i not in visited])
            while local_de:
                if len(local_visited) >= context:
                    chunks.append(local_visited)
                    break
                local_curr = local_de.pop()
                if local_curr not in visited:
                    visited.append(local_curr)
                    local_visited.append(local_curr)
                    neighbors = g.neighbors(local_curr)
                    local_de.extendleft([i for i in neighbors if i not in visited])
            if local_visited not in chunks:
                chunks.append(local_visited)
            de += local_de
    return chunks

