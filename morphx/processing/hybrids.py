# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import pickle
import os
import numpy as np
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.pointcloud import PointCloud


def extract_mesh_subset(hybrid: HybridCloud, local_bfs: np.ndarray) -> PointCloud:
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
