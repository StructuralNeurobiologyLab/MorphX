# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import networkx as nx
import numpy as np
from collections import defaultdict, deque


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


def extract_mesh_subset(skel_nodes: np.ndarray, vertices: np.ndarray, mapping: defaultdict) -> np.ndarray:
    """ Returns the mesh subset of given skeleton nodes based on a mapping dict between skeleton and mesh.

    Args:
        skel_nodes: Index array of skeleton nodes (entries refer to indices of ``skel_nodes`` array from pointcloud)
            as base for the mesh subset.
        vertices: The total mesh in form of a coordinate array of shape (n, 3).
        mapping: A mapping dict between skeleton and mesh with skeleton nodes as keys and lists of the corresponding
            mesh vertices (to which the respective node is nearest) as values.

    Returns:
        An coordinate array of shape (n, 3) with the subset of vertices.
    """
    total = []
    for i in skel_nodes:
        total.extend(mapping[i])
    return vertices[total]


def sample_subset(subset: np.ndarray, vertex_number: int, random_seed=None) -> np.ndarray:
    """ Creates a (pseudo)random sample point cloud with a specific number of points from the given subset of mesh
    vertices. If the requested number of points is larger than the given subset, the subset gets enriched with its own
    augmented points before sampling.

    Args:
        subset: An array of mesh vertices with shape (n,3).
        vertex_number: The number of points which should make up the sample point cloud.
        random_seed: Possibility for making the sampling deterministic.

    Returns:
        Array of sampled points
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    dim = max(subset.shape)
    if dim < vertex_number:
        deficit = vertex_number - dim
        aug_extract = np.array([])
        while max(aug_extract.shape) < deficit:
            extract = subset[np.random.choice(np.arange(dim), int(dim/(1/4)))]
            # TODO: Better augmentation needed?
            if max(aug_extract.shape) == 0:
                aug_extract = extract+np.random.random(extract.shape)
            else:
                aug_extract = np.concatenate((aug_extract, extract+np.random.random(extract.shape)), axis=0)
        compensation = aug_extract[np.random.choice(np.arange(max(aug_extract.shape)), deficit)]
        return np.concatenate((subset, compensation), axis=0)
    else:
        return subset[np.random.choice(np.arange(dim), vertex_number)]


def normalize_cloud(cloud: np.ndarray) -> np.ndarray:
    """ Centers and normalizes point cloud.

    Args:
        cloud: Point cloud as array of coordinates.

    Returns:
        Centered and normalized point cloud as array of coordinates.
    """
    centroid = np.mean(cloud, axis=0)
    c_cloud = cloud-centroid
    nc_cloud = c_cloud / np.linalg.norm(c_cloud)
    return nc_cloud
