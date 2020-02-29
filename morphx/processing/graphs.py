# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import ipdb
import numpy as np
import networkx as nx
from collections import deque
from scipy.spatial import cKDTree
from morphx.classes.hybridcloud import HybridCloud


def bfs_base_points(g: nx.Graph, min_dist: float, source: int = -1) -> np.ndarray:
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
                buddy = len(chosen)
                chosen.append(curr)

            # add all neighbors with their weights added to the current weight
            neighbors = g.neighbors(curr)
            de.extendleft([(i, buddy) for i in neighbors if i not in visited])

    # return only chosen nodes
    return np.array(chosen)


def bfs_euclid_sphere(g: nx.Graph, source: int, max_dist: float) -> np.ndarray:
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


def bfs_vertices(hc: HybridCloud, source: int, vertex_max: int) -> np.ndarray:
    """ Performs a BFS on a graph until the number of vertices which correspond to
        the currently selected nodes reaches the maximum.

    Args:
        hc: The HybridCloud with the graph, nodes and vertices on which the BFS should be performed
        source: The source node from which the BFS should start.
        vertex_max: The maximum number of vertices after which the BFS should stop.

    Returns:
        np.ndarray with nodes sorted recording to the result of the limited BFS
    """
    visited = [source]
    vertex_num = len(hc.verts2node[source])
    neighbors = hc.graph().neighbors(source)
    de = deque([i for i in neighbors])
    while de:
        curr = de.pop()
        if curr not in visited:
            if vertex_num + len(hc.verts2node[curr]) <= vertex_max:
                visited.append(curr)
                vertex_num += len(hc.verts2node[curr])
                neighbors = hc.graph().neighbors(curr)
                de.extendleft([i for i in neighbors if i not in visited])
            else:
                return np.array(visited)

    return np.array(visited)


def bfs_vertices_euclid(hc: HybridCloud, source: int, vertex_max: int, euclid: int, context: int = 20) -> np.ndarray:
    """ 1. Reduce number of nodes of interest by exploiting the skeleton structure and extracting nodes
        within a certain euclidian radius
        2. Get independent from the (possibly irregular) skeleton by performing a k-nearest neighbor search
        on the node extract with respect to the source
        3. Iterate the k-nearest neighbors in ascending order until maximum number of corresponding vertices
        is reached

    Args:
        hc: The HybridCloud with the graph, nodes and vertices on which the BFS should be performed
        source: The source node from which the BFS should start.
        vertex_max: The maximum number of vertices which should be included in the chunk
        euclid: All nodes within this radius of the source are iterated
        context: Used for creating the node context which speeds up the k-nearest neighbor search. Starting
            from the source, a bfs is performed until 'context' consecutive nodes are outside of the 'euclid'
            radius

    Returns:
        np.ndarray with nodes which were extracted during this bfs
    """
    g = hc.graph()
    source_pos = g.nodes[source]['position']
    # run bfs in order to get node context on which following k-nearest neighbor search is faster
    visited = [source]
    node_extract = []
    neighbors = g.neighbors(source)
    de = deque([(i, [i]) for i in neighbors])
    while de:
        curr, out_preds = de.pop()
        if curr not in visited:
            visited.append(curr)
            if np.linalg.norm(source_pos - g.nodes[curr]['position']) <= euclid:
                node_extract.append(curr)
                out_preds = []
            # don't add neighbors if previous 'context' nodes are outside of euclidian sphere
            if len(out_preds) < context:
                neighbors = g.neighbors(curr)
                for i in neighbors:
                    if i not in visited:
                        new_out_preds = out_preds.copy()
                        new_out_preds.append(i)
                        de.extendleft([(i, new_out_preds)])

    # the node context contains all nodes within a certain euclidian radius. In order to get independent from the
    # skeleton, the graph structure gets abandoned and the nodes are iterated in order of their distance to the source.
    # This is computationally feasible as the extracted node context only contains a few (probably < 100) nodes.
    extract_coords = np.zeros((len(node_extract), 3))
    for ix, node in enumerate(node_extract):
        extract_coords[ix] = g.nodes[node]['position']
    tree = cKDTree(extract_coords)
    dist, ind = tree.query(source_pos, k=len(extract_coords))
    if isinstance(ind, int):
        ind = np.array([ind])

    # iterate query result in order of distance (small to large) and add nodes to result as long as the number of
    # corresponding vertices is below the specified number
    vertex_num = len(hc.verts2node[source])
    bfs_result = [source]
    for ix in ind:
        node = node_extract[ix]
        if vertex_num + len(hc.verts2node[node]) <= vertex_max:
            bfs_result.append(node)
            vertex_num += len(hc.verts2node[node])
        else:
            break

    return np.array(bfs_result)
