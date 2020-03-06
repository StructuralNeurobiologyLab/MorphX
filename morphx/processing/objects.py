# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
from collections import deque
from morphx.data import basics
from typing import Union, Tuple
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


def bfs_vertices_diameter(hc: Union[HybridCloud, CloudEnsemble], source: int, vertex_max: int, radius: int = 1000) \
        -> np.ndarray:
    """ Traverses the skeleton nodes. For each node, all nodes within 'radius' are sorted according to their distance.
        Then the nodes are added to the result as long as vertex number is below threshold.

    Returns:
        Index array which contains indices of nodes in the hc node array which are part of the bfs result.
    """
    source = int(source)
    chosen = []
    idx_nodes = np.arange(len(hc.nodes))
    visited = [source]
    dia_nodes = idx_nodes[np.linalg.norm(hc.nodes - hc.nodes[source], axis=1) <= radius]
    vertex_num = 0
    # sort nodes within 'radius' by their distance
    tree = cKDTree(hc.nodes[dia_nodes])
    dist, ind = tree.query(hc.nodes[source], k=len(dia_nodes))
    for ix in ind:
        node = dia_nodes[ix]
        # add nodes as long as number of corresponding vertices is still below the threshold
        if vertex_num + len(hc.verts2node[node]) <= vertex_max:
            chosen.append(node)
            vertex_num += len(hc.verts2node[node])
        else:
            return np.array(chosen)
    # traverse all nodes
    neighbors = hc.graph().neighbors(source)
    de = deque([i for i in neighbors])
    while de:
        curr = de.pop()
        if curr not in visited:
            visited.append(curr)
            dia_nodes = idx_nodes[np.linalg.norm(hc.nodes - hc.nodes[curr], axis=1) <= radius]
            tree = cKDTree(hc.nodes[dia_nodes])
            dist, ind = tree.query(hc.nodes[source], k=len(dia_nodes))
            for ix in ind:
                node = dia_nodes[ix]
                if node not in chosen:
                    if vertex_num + len(hc.verts2node[node]) <= vertex_max:
                        chosen.append(node)
                        vertex_num += len(hc.verts2node[node])
                    else:
                        return np.array(chosen)
            neighbors = hc.graph().neighbors(curr)
            de.extendleft([i for i in neighbors])
    return np.array(chosen)


def bfs_base_points_density(hc: Union[HybridCloud, CloudEnsemble], vertex_max: int, source: int = -1) -> np.ndarray:
    """ Extracts base points which have an approximate number of vertices between them.

    Args:
        hc: the HybridCloud with the graph and vertices.
        vertex_max: the approximate number of vertices which should be between two base points
            (corresponding to the nodes between them)
        source: the starting point.
    """
    # after context nodes the number of vertices between the current and the starting node is checked
    # if vertex number is higher than threshold the node gets added as a base point (doing this context-
    # wise speeds up the process in contrast to doing it node-wise)
    context = 20
    if source == -1:
        source = np.random.randint(hc.graph().number_of_nodes())
    chosen = [source]
    visited = [source]
    neighbors = hc.graph().neighbors(source)
    de = deque([(i, [source], 0) for i in neighbors])
    while de:
        curr, preds, verts_num = de.pop()
        if curr not in visited:
            visited.append(curr)
            # sum up all corresponding vertices after each 'context' nodes
            if len(preds) >= context:
                for node in preds:
                    verts_num += len(hc.verts2node[node])
                if verts_num > vertex_max:
                    # include this node only if there are enough vertices between this one and all nodes
                    # which were already chosen
                    include = True
                    for chosen_node in reversed(chosen):
                        if np.linalg.norm(hc.nodes[curr] - hc.nodes[chosen_node]) < 5000:
                            include = False
                        # if not enough_vertices(hc, vertex_max, curr, chosen_node):
                        #     include = False
                        #     break
                    if include:
                        chosen.append(curr)
                        verts_num = 0
                # reset the context to process the next 'context' nodes. The number of vertices
                # is still the old one unless the node was added as a base point
                preds = []
            preds.append(curr)
            neighbors = hc.graph().neighbors(curr)
            de.extendleft([(i, preds, verts_num) for i in neighbors])
    return np.array(chosen)


def enough_vertices(hc: Union[HybridCloud, CloudEnsemble], vertex_max: int, source: int, goal: int) -> bool:
    """ Checks if number of vertices corresponding to the skeleton nodes between the source and the
        nearest node in chosen is below a certain threshold. Can be used to ensure a minimum distance
        between base points.

    Args:
        hc: the HybridCloud with the graph and vertices
        vertex_max: the threshold from the description
        source: the starting point
        goal: a list of nodes from which the nearest node to the source is decisive

    Returns:
        True if there are enough vertices between the source and the next node in chosen, False
        otherwise
    """
    visited = [source]
    vertex_num = len(hc.verts2node[source])
    neighbors = hc.graph().neighbors(source)
    de = deque([(i, vertex_num) for i in neighbors])
    while de:
        curr, vertex_num = de.pop()
        if curr not in visited:
            visited.append(curr)
            vertex_num += len(hc.verts2node[curr])
            # if goal node was reached, but not enough vertices are in between, return False
            if curr == goal and vertex_num < vertex_max:
                return False
            # if there are enough vertices and goal has not been or has just been reached, return True
            if vertex_num > vertex_max:
                return True
            neighbors = hc.graph().neighbors(curr)
            de.extendleft([(i, vertex_num) for i in neighbors])
    return True


def bfs_vertices(hc: Union[HybridCloud, CloudEnsemble], source: int, vertex_max: int) -> np.ndarray:
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


def bfs_vertices_euclid(hc: Union[HybridCloud, CloudEnsemble], source: int, vertex_max: int, euclid: int,
                        cutoff: int = 20) -> np.ndarray:
    """ Starting from the source, a bfs is performed until 'context' consecutive nodes have a distance > 'euclid'
        from the source. The resulting node list is then iterated until the maximum number of vertices is reached.
        This gets all nodes within a certain radius in the order they would appear when traversing the skeleton.

    Args:
        hc: The HybridCloud with the graph, nodes and vertices on which the BFS should be performed
        source: The source node from which the BFS should start.
        vertex_max: The maximum number of vertices which should be included in the chunk
        euclid: All nodes within this radius of the source are iterated
        cutoff: BFS gets performed until 'context' consecutive nodes have a distance > 'euclid' from
            the source

    Returns:
        np.ndarray with nodes which were extracted during the bfs
    """
    g = hc.graph()
    source_pos = g.nodes[source]['position']
    # run bfs in order to get node context on which following k-nearest neighbor search is faster
    visited = [source]
    node_extract = [source]
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
            if len(out_preds) < cutoff:
                neighbors = g.neighbors(curr)
                for i in neighbors:
                    if i not in visited:
                        new_out_preds = out_preds.copy()
                        new_out_preds.append(i)
                        de.extendleft([(i, new_out_preds)])
    bfs_result = []
    vertex_num = 0
    ix = 0
    while vertex_num <= vertex_max and ix < len(node_extract):
        vertex_num += len(hc.verts2node[node_extract[ix]])
        bfs_result.append(node_extract[ix])
        ix += 1
    return np.array(bfs_result)