# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import random
import networkx as nx
from collections import deque
from morphx.data import basics
from typing import Union, Tuple, List
from scipy.spatial import cKDTree
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.processing import hybrids, ensembles, clouds


# -------------------------------------- DISPATCHER METHODS -------------------------------------- #


def extract_cloud_subset(obj: Union[HybridCloud, CloudEnsemble],
                         local_bfs: np.ndarray) -> Tuple[PointCloud, np.ndarray]:
    if isinstance(obj, HybridCloud):
        return hybrids.extract_subset(obj, local_bfs)
    elif isinstance(obj, CloudEnsemble):
        return ensembles.extract_subset(obj, local_bfs)
    else:
        raise ValueError(f'Expected CloudEnsemble or HybridCloud instance but got '
                         f'{type(obj)} instead.')


def filter_preds(obj: Union[HybridCloud, CloudEnsemble]) -> PointCloud:
    if isinstance(obj, CloudEnsemble):
        pc = obj.flattened
        pc.set_predictions(obj.predictions)
        obj = pc
    return clouds.filter_preds(obj)


def load_obj(data_type: str, file: str) -> Union[HybridMesh, HybridCloud, PointCloud, CloudEnsemble]:
    if data_type == 'obj':
        return basics.load_pkl(file)
    if data_type == 'ce':
        return ensembles.ensemble_from_pkl(file)
    if data_type == 'hc':
        hc = HybridCloud()
        return hc.load_from_pkl(file)
    if data_type == 'hm':
        hm = HybridMesh()
        return hm.load_from_pkl(file)
    else:
        pc = PointCloud()
        return pc.load_from_pkl(file)


# -------------------------------------- BFS ALGORITHMS -------------------------------------- #

def density_splitting(obj: Union[HybridCloud, CloudEnsemble], source: int, vertex_max: int, radius: int = 1000) \
        -> np.ndarray:
    """ Traverses the skeleton with a BFS. For each node, all nodes within 'radius' are sorted according to their
        distance. Then these nodes are added to the resulting node array as long as the total number of corresponding
        vertices is still below vertex_max.

    Args:
        obj: MorphX object which contains the skeleton.
        source: The index of the node to start with.
        vertex_max: The vertex threshold after which the BFS is stopped.
        radius: Workaround to be independent from the skeleton microstructure. All nodes within this radius get sorted
            according to their distance and are then added to the result.

    Returns:
        Index array which contains indices of nodes in the hc node array which are part of the bfs result.
    """
    source = int(source)
    chosen = []
    idx_nodes = np.arange(len(obj.nodes))
    visited = [source]
    dia_nodes = idx_nodes[np.linalg.norm(obj.nodes - obj.nodes[source], axis=1) <= radius]
    vertex_num = 0
    # sort nodes within 'radius' by their distance
    tree = cKDTree(obj.nodes[dia_nodes])
    dist, ind = tree.query(obj.nodes[source], k=len(dia_nodes))
    if isinstance(ind, int):
        ind = [ind]
    for ix in ind:
        node = dia_nodes[ix]
        # add nodes as long as number of corresponding vertices is still below the threshold
        if vertex_num + len(obj.verts2node[node]) <= vertex_max:
            chosen.append(node)
            vertex_num += len(obj.verts2node[node])
        else:
            return np.array(chosen)
    # traverse all nodes
    neighbors = obj.graph().neighbors(source)
    de = deque([i for i in neighbors])
    while de:
        curr = de.pop()
        # visited is node list for traversing the graph
        if curr not in visited:
            visited.append(curr)
            dia_nodes = idx_nodes[np.linalg.norm(obj.nodes - obj.nodes[curr], axis=1) <= radius]
            tree = cKDTree(obj.nodes[dia_nodes])
            dist, ind = tree.query(obj.nodes[source], k=len(dia_nodes))
            if isinstance(ind, int):
                ind = [ind]
            for ix in ind:
                node = dia_nodes[ix]
                # chosen is node list for gathering all valid nodes
                if node not in chosen:
                    if vertex_num + len(obj.verts2node[node]) <= vertex_max:
                        chosen.append(node)
                        vertex_num += len(obj.verts2node[node])
                    else:
                        return np.array(chosen)
            neighbors = obj.graph().neighbors(curr)
            de.extendleft([i for i in neighbors])
    return np.array(chosen)


def context_splitting(obj: Union[HybridCloud, CloudEnsemble], source: int, max_dist: float,
                      radius: int = 1000) -> np.ndarray:
    """ Traverses the skeleton with a BFS. For each node, all neighboring nodes within 'radius' get added to the
        resulting array if their distance to the 'source' node is below 'max_dist'.

    Args:
        obj: MorphX object which contains the skeleton.
        source: The index of the node to start with.
        max_dist: The maximum distance which limits the BFS.
        radius: Workaround to be independent from the skeleton microstructure. Radius of sphere around each node
            in which all nodes should get processed.

    Returns:
        np.ndarray with nodes sorted recording to the result of the limited BFS
    """
    idx_nodes = np.arange(len(obj.nodes))
    visited = []
    chosen = []
    de = deque([source])
    neighbors = obj.graph().neighbors(source)
    de.extendleft([i for i in neighbors if np.linalg.norm(obj.nodes[source] - obj.nodes[i]) <= max_dist])
    while de:
        curr = de.pop()
        if curr not in visited:
            visited.append(curr)
            # get all nodes within radius and check if condition is satisfied
            dia_nodes = idx_nodes[np.linalg.norm(obj.nodes - obj.nodes[curr], axis=1) <= radius]
            tree = cKDTree(obj.nodes[dia_nodes])
            dist, ind = tree.query(obj.nodes[curr], k=len(dia_nodes))
            if isinstance(ind, int):
                ind = [ind]
            for ix in ind:
                node = dia_nodes[ix]
                if node not in chosen:
                    if np.linalg.norm(obj.nodes[source] - obj.nodes[node]) <= max_dist:
                        chosen.append(node)
            neighbors = obj.graph().neighbors(curr)
            de.extendleft([i for i in neighbors if np.linalg.norm(obj.nodes[source] - obj.nodes[i]) <= max_dist])
    return np.array(chosen)


def bfs_vertices(hc: Union[HybridCloud, CloudEnsemble], source: int, vertex_max: int) -> np.ndarray:
    """ Performs a BFS on a graph until the number of vertices which correspond to
        the currently selected nodes reaches the maximum.

    Args:
        hc: The HybridCloud with the graph, nodes and vertices on which the BFS should be performed
        source: The source node from which the BFS should start.
        vertex_max: The maximum number of vertices after which the BFS should stop.

    Returns:
        np.ndarray with nodes sorted according to the result of the limited BFS
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


def label_split(obj: Union[HybridCloud, CloudEnsemble], center_l: int) -> List[np.ndarray]:
    """ Works for graphs which have one central component (labeled with center_l) and multiple subgraphs
        emerging from that component. This method splits such graphs into the central component and subgraphs.
        Each component can thereby contain small label variations, each subgraph is purely labeled by the most
        dominating label in that subgraph.

        Procedure:
        1. Reduce graph to nodes which contain consecutive nodes of only one label.
        2. Find the largest central component in that reduced graph (by number of vertices).
        3. Split the graph into subgraphs by removing the central component and taking the remaining connected
           components.
        4. Label all subgraphs according to the dominating vertex labels within them.

        Example:
        1-1-2-1-2-2-2-3-1-3 with center_l = 2
        1. Reduce to 1-2-1-2-3-1-3
        2. Three consecutive nodes of 2 have largest number of vertices
        3. Split into: 1-2-1  2  3-1-3
        4. Relabel: 1-1-1  2  3-3-3  (assuming the 1 and 3 labels have the dominating number of vertices)

    Args:
        obj: The MorphX object which contains the graph and the corresponding vertices
        center_l: The label of the central component

    Returns:
        List of subgraphs as node arrays.
    """
    source = obj.nodes[random.randrange(len(obj.nodes))]


def label_components(obj: Union[HybridCloud, CloudEnsemble], source: int) -> dict:
    """ Reduces graph of given object to nodes which contain consecutive nodes of only one label.

    Args:
        obj: MorphX object which contains the skeleton and node_labels
        source: The id of the node where the BFS should start.

    Returns:
        Dict of lists of consecutive nodes with only one label, keyed by the index of a new, representative node.
    """
    visited = [source]
    reduced = {0: [source]}
    neighbors = obj.graph().neighbors(source)
    de = deque([(i, 0, obj.node_labels[source]) for i in neighbors])
    while de:
        curr, r_node, prev = de.pop()
        if curr not in visited:
            visited.append(curr)
            curr_l = obj.node_labels[curr]
            # label transition => create new reduced node
            if curr_l != prev:
                r_node += 1
                while r_node in reduced:
                    r_node += 1
                reduced[r_node] = []
            reduced[r_node].append(curr)
            neighbors = obj.graph().neighbors(curr)
            de.extendleft([(i, r_node, curr_l) for i in neighbors if i not in visited])
    return reduced
