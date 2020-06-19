# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
from collections import deque
from morphx.data import basics
from typing import Union, Tuple, Optional, Iterable, List
from scipy.spatial import cKDTree
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.processing import hybrids, ensembles, clouds
import networkx as nx


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
        max_dist: The maximum (Euclidean) distance (in nm) between the two most distance nodes, which limits the BFS.
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


def context_splitting_kdt(obj: Union[HybridCloud, CloudEnsemble], source: int, max_dist: float,
                          radius: Optional[int] = None) -> np.ndarray:
    """ Performs a query ball search (via kd-tree) to find the connected sub-graph within `max_dist` starting at the
    source node in the obj's weighted graph.

    Args:
        obj: The HybridCloud/CloudEnsemble with the graph, nodes and vertices.
        source: The source node.
        max_dist: The maximum distance to the source node (Euclidean distance, i.e. not along the graph!).
        radius: Optionally add edges between nodes within `radius`.

    Returns:
        The nodes within the requested context for every source node - same ordering as `sources`.
    """
    node_ixs = np.arange(len(obj.nodes)).tolist()
    g = nx.Graph()
    g.add_edges_from(obj.edges)

    # remove nodes outside max radius
    # TODO: use sg = nx.subgraph(g, ixs) instead of pruning the original graph with a loop..
    kdt = cKDTree(obj.nodes)
    ixs = np.concatenate(kdt.query_ball_point([obj.nodes[source]], max_dist)).tolist()
    diff = set(node_ixs).difference(set(ixs))
    # fails if there is a node without edge!
    for ix in diff:
        g.remove_node(ix)

    if radius is not None:
        # add edges within 1µm
        kdt = cKDTree(obj.nodes[ixs])
        pairs = kdt.query_pairs(radius)
        # remap to subset of indices
        g.add_edges_from([(ixs[p[0]], ixs[p[1]]) for p in pairs])

    return np.array(list(nx.node_connected_component(g, source)))


def context_splitting_kdt_many(obj: Union[HybridCloud, CloudEnsemble], sources: Iterable[int], max_dist: float,
                               radius: Optional[int] = None) -> List[np.ndarray]:
    """ Performs a query ball search (via kd-tree) to find the connected sub-graph within `max_dist` starting at the
    source node in the obj's weighted graph.

    Args:
        obj: The HybridCloud/CloudEnsemble with the graph, nodes and vertices.
        sources: The source nodes.
        max_dist: The maximum distance to the source node (Euclidean distance, i.e. not along the graph!).
        radius: Optionally add edges between nodes within `radius`.

    Returns:
        The nodes within the requested context for every source node - same ordering as `sources`.
    """
    g = nx.Graph()
    g.add_edges_from(obj.edges)

    ctxs = []

    kdt = cKDTree(obj.nodes)
    # TODO: use query_ball_tree if sources is large
    ixs_nn = kdt.query_ball_point(obj.nodes[sources], max_dist)
    for source, ixs in zip(sources, ixs_nn):
        # remove nodes outside max_dist
        sg = nx.subgraph(g, ixs)
        if radius is not None:
            # add edges within 1µm
            kdt = cKDTree(obj.nodes[ixs])
            pairs = kdt.query_pairs(radius)
            # remap to subset of indices
            sg.add_edges_from([(ixs[p[0]], ixs[p[1]]) for p in pairs])
        ctxs.append(np.array(list(nx.node_connected_component(sg, source))))
    return ctxs


def context_splitting_graph_many(obj: Union[HybridCloud, CloudEnsemble], sources: Iterable[int],
                                 max_dist: float) -> List[list]:
    """ Performs a dijkstra shortest paths on the obj's weighted graph to retrieve the skeleton nodes within `max_dist`
    for every source node ID in `sources`.

    Args:
        obj: The HybridCloud/CloudEnsemble with the graph, nodes and vertices.
        sources: The source nodes.
        max_dist: The maximum distance to the source node (along the graph).

    Returns:
        The nodes within the requested context for every source node - same ordering as `sources`.
    """
    g = obj.graph()
    paths = nx.all_pairs_dijkstra_path(g, weight='weight', cutoff=max_dist)
    return [list(paths[s].keys()) for s in sources]


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
