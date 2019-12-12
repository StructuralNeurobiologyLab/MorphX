# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import pickle
import os
import time
import numpy as np
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.pointcloud import PointCloud
from morphx.classes.meshcloud import MeshCloud


def extract_cloud_subset(hybrid: HybridCloud, local_bfs: np.ndarray) -> PointCloud:
    """ Returns the mesh subset of given skeleton nodes based on a mapping dict between skeleton and mesh.

    Args:
        hybrid: MorphX HybridCloud object from which the subset should be extracted.
        local_bfs: Skeleton node index array which was generated by a local BFS.

    Returns:
        Mesh subset as PointCloud object
    """
    mapping = hybrid.vert2skel
    vertices = hybrid.vertices
    labels = hybrid.labels

    total = []
    for i in local_bfs:
        total.extend(mapping[i])

    if labels is not None:
        return PointCloud(vertices[total], labels=labels[total])
    else:
        return PointCloud(vertices[total])


def extract_mesh_subset(hm: HybridMesh, local_bfs: np.ndarray) -> tuple:
    """ Returns the mesh subset of given skeleton nodes based on a mapping dict between skeleton and mesh.

    Args:
        hm: HybridMesh from which the subset should be extracted.
        local_bfs: Skeleton node index array which was generated by a local BFS.

    Returns:
        Mesh subset as PointCloud object
    """
    mapping = hm.vert2skel
    faces = hm.faces

    total = []
    for i in local_bfs:
        total.extend(mapping[i])
    new_vertices = hm.vertices[total]
    new_labels = hm.labels[total]

    # filter faces belonging to the chosen vertices
    start = time.time()
    new_faces = faces[np.all(np.isin(faces, total), axis=1)]
    print("Filtering done in {} seconds".format(time.time()-start))

    # return full vertices and labels as filtered faces still point to original indices of vertices
    # The return object can then be reduced to the actual sample cloud by performing mesh sampling on it
    mc = MeshCloud(hm.vertices, new_faces, np.array([]), labels=hm.labels, encoding=hm.encoding)
    return mc, new_vertices, new_labels


