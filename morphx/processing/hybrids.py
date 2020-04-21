# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch, Philipp Schubert

import numpy as np
from typing import Tuple
from collections import deque
import networkx as nx
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.pointcloud import PointCloud


# -------------------------------------- HYBRID EXTRACTION --------------------------------------- #

def extract_subset(hybrid: HybridCloud, local_bfs: np.ndarray) -> Tuple[PointCloud, np.ndarray]:
    """ Returns the mesh subset of given skeleton nodes based on a mapping dict
     between skeleton and mesh.

    Notes:
        * Node IDs f input and output HybridCloud will not be the same!

    Args:
        hybrid: MorphX HybridCloud object from which the subset should be extracted.
        local_bfs: Skeleton node index array which was generated by a local BFS.

    Returns:
        Mesh subset as HybridCloud object, vertex IDs from the parent
        HybridCloud with the same ordering as vertices (enables "global" look-up).
    """
    idcs = []
    local_bfs = np.sort(local_bfs)
    for i in local_bfs:
        idcs.extend(hybrid.verts2node[i])
    verts = hybrid.vertices[idcs]
    feats, labels, node_labels = None, None, None
    if hybrid.features is not None and len(hybrid.features) > 0:
        feats = hybrid.features[idcs]
    if hybrid.labels is not None and len(hybrid.labels) > 0:
        labels = hybrid.labels[idcs]
    if hybrid.node_labels is not None and len(hybrid.node_labels) > 0:
        node_labels = hybrid.node_labels[local_bfs]
    g = hybrid.graph(simple=True)
    sub_g = g.subgraph(local_bfs)
    relabel_dc = {n: ii for ii, n in enumerate(sub_g.nodes)}
    sub_g = nx.relabel.relabel_nodes(sub_g, relabel_dc)
    edges = np.array(sub_g.edges)
    nodes = hybrid.nodes[local_bfs]
    hc = HybridCloud(nodes=nodes, edges=edges, node_labels=node_labels,
                   vertices=verts, labels=labels, features=feats,
                   no_pred=hybrid.no_pred, obj_bounds=hybrid.obj_bounds,
                   encoding=hybrid.encoding)
    return hc, np.array(idcs)


def extract_mesh_subset(hm: HybridMesh, node_ids: np.ndarray) -> HybridMesh:
    """ Returns the mesh subset of given skeleton nodes based on the face
     mapping dict between skeleton and mesh.

    Args:
        hm: HybridMesh from which the subset should be extracted.
        node_ids: Skeleton node index array.

    Returns:
        Mesh subset as HybridMesh object.
    """
    mapping_face = hm.faces2node

    total_face = set()
    for node_ix in node_ids:
        total_face.update(set(mapping_face[node_ix]))

    total_face = np.array(list(total_face), dtype=np.int)

    if len(total_face) == len(hm.faces):
        # all faces haven been selected
        faces = hm.faces
        vertices = hm.vertices
        labels = hm.labels
        normals = hm.normals
        features = hm.features
        nodes = hm.nodes
        node_labels = hm.node_labels
        edges = hm.edges
    else:
        faces = hm.faces[total_face]
        total_vertex = np.unique(faces.flatten()).astype(int)  # also sorts the ids
        vertices = hm.vertices[total_vertex]
        if len(hm.labels) > 0:
            labels = hm.labels[total_vertex]
        else:
            labels = None
        if len(hm.normals) > 0:
            normals = hm.normals[total_vertex]
        else:
            normals = None
        if len(hm.features) > 0:
            features = hm.features[total_vertex]
        else:
            features = None
        # normalize new faces to be contiguous
        face_shape = faces.shape
        faces = faces.flatten()
        updated_face_ixs = {face_id: k for k, face_id in enumerate(np.unique(faces))}
        for ix, old_face_ix in enumerate(faces):
            # update face indices
            faces[ix] = updated_face_ixs[old_face_ix]
        # switch to original shape
        faces = faces.reshape(face_shape)
        node_labels = None
        if hm.node_labels is not None and len(hm.node_labels) > 0:
            node_labels = hm.node_labels[node_ids]
        g = hm.graph(simple=True)
        sub_g = g.subgraph(node_ids)
        relabel_dc = {n: ii for ii, n in enumerate(sub_g.nodes)}
        sub_g = nx.relabel.relabel_nodes(sub_g, relabel_dc)
        nodes = hm.nodes[node_ids]
        edges = np.array(sub_g.edges)
    hm = HybridMesh(nodes=nodes, node_labels=node_labels, edges=edges,
                    vertices=vertices, faces=faces, normals=normals, labels=labels,
                    features=features, encoding=hm.encoding)
    return hm


# -------------------------------------- HYBRID GRAPH ALGORITHMS --------------------------------------- #

def label_search(hc: HybridCloud, node_labels: np.ndarray, source: int) -> int:
    """ Performs BFS on nodes starting from source until first node with label != -1 has been found.

    Notes:
        * Node IDs f input and output HybridCloud will not be the same!

    Args:
        hc: HybridCloud in which source is part of the graph
        node_labels: array of node labels (cannot be accessed through hc as it doesn't exists).
        source: The node for which the first neighboring node with label != -1 should be found.

    Returns:
        The index of the first node with label != -1
    """
    if node_labels is None:
        return source
    g = hc.graph()
    visited = [source]
    neighbors = g.neighbors(source)
    de = deque([i for i in neighbors])
    while de:
        curr = de.pop()
        if node_labels[curr] != -1:
            return curr
        if curr not in visited:
            visited.append(curr)
            neighbors = g.neighbors(curr)
            de.extendleft([i for i in neighbors if i not in visited])
    return source


