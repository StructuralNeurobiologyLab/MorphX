# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import point_cloud_utils as pcu
from morphx.processing import clouds
from morphx.classes.pointcloud import PointCloud
from morphx.classes.meshcloud import MeshCloud
from scipy.spatial import cKDTree


# -------------------------------------- MESH SAMPLING ------------------------------------------- #

def sample_mesh_poisson_disk(mc: MeshCloud, sample_num: int) -> PointCloud:
    """ Uses Point-Disk-Sampling as described at https://github.com/fwilliams/point-cloud-utils and maps existing
        labels using a KDTree.

    Args:
        mc: The MeshCloud from which the samples should be generated
        sample_num: Requested number of sample points.

    Returns:
        PointCloud consisting of sampled points.
    """

    vertices = mc.vertices.astype(float)
    s_vertices, s_normals = pcu.sample_mesh_poisson_disk(vertices, mc.faces, np.array([]), sample_num)

    # TODO: This can be improved
    labels = None
    # map labels from input cloud to sample
    if mc.labels is not None:
        tree = cKDTree(mc.vertices)
        dist, ind = tree.query(s_vertices, k=1)
        labels = mc.labels[ind]

    # sample again, as pcu.sample doesn't always return the requested number of samples
    # (only a few points must be added)
    spc, ixs = clouds.sample_objectwise(PointCloud(s_vertices, labels=labels), sample_num)

    return spc
