# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import point_cloud_utils as pcu
from math import ceil
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridmesh import HybridMesh
from scipy.spatial import cKDTree


# -------------------------------------- MESH SAMPLING ------------------------------------------- #

def sample_mesh_poisson_disk(hm: HybridMesh, sample_num: int) -> PointCloud:
    """ Uses poisson disk sampling and maps existing labels using a KDTree. It can not be guaranteed
        that the requested number of sample points can be generated. It can differ from the requested
        number by a small amount (around +-2%).

    Args:
        hm: The MeshCloud from which the samples should be generated
        sample_num: Requested number (approximate!) of sample points.

    Returns:
        PointCloud consisting of sampled points.
    """
    vertices = hm.vertices.astype(float)
    s_vertices, s_normals = pcu.sample_mesh_poisson_disk(vertices, hm.faces, np.array([]), ceil(sample_num))

    # map labels from input cloud to sample
    labels = None
    types = None

    tree = cKDTree(hm.vertices)
    dist, ind = tree.query(s_vertices, k=1)
    if len(hm.labels) > 0:
        labels = hm.labels[ind]
    if len(hm.types) > 0:
        types = hm.types[ind]

    result = PointCloud(vertices=s_vertices.reshape(-1, 3), labels=labels, types=types)
    return result
