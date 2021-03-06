# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import networkx as nx
from collections import deque
from typing import List


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


def extract_label_subgraphs(g: nx.Graph, label: int) -> List[List[int]]:
    # get nodes with specific label
    snodes = []
    subgraphs = []
    for node in g.nodes:
        if g.nodes[node]['label'] == label:
            snodes.append(node)
    # get all minimal subgraphs which include at least one node of the specified label
    for node in snodes:
        for i in range(len(snodes)):
            subgraph = bfs_until_n_snodes(g, node, label, i)
            if subgraph not in subgraphs:
                subgraphs.append(subgraph)
    return subgraphs


def bfs_until_n_snodes(g: nx.Graph, source: int, label: int, n: int) -> List[int]:
    num = 0
    result = [source]
    if g.nodes[source]['label'] == label:
        num += 1
    neighbors = g.neighbors(source)
    de = deque([i for i in neighbors])
    while de:
        if num == n:
            return result
        curr = de.pop()
        if curr not in result:
            result.append(curr)
            if g.nodes[curr]['label'] == label:
                num += 1
            neighbors = g.neighbors(curr)
            de.extendleft([i for i in neighbors if i not in result])
    if num != n:
        raise ValueError(f"Graph does not contain {n} nodes with the label {label}")
    else:
        return result
