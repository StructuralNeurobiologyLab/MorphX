# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import open3d as o3d
from morphx.classes.pointcloud import PointCloud
from morphx.classes.meshcloud import MeshCloud
from scipy.spatial import cKDTree


# -------------------------------------- MESH SAMPLING ------------------------------------------- #

def sample_mesh_poisson_disk(mc: MeshCloud, sample_num: int) -> PointCloud:
    """ Uses poisson disk sampling and maps existing labels using a KDTree.

    Args:
        mc: The MeshCloud from which the samples should be generated
        sample_num: Requested number of sample points.

    Returns:
        PointCloud consisting of sampled points.
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(mc.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(mc.faces)
    sample = mesh.sample_points_poisson_disk(sample_num)
    s_vertices = np.asarray(sample.points)

    # map labels from input cloud to sample
    labels = None
    if mc.labels is not None:
        tree = cKDTree(mc.vertices)
        dist, ind = tree.query(s_vertices, k=1)
        labels = mc.labels[ind]

    result = PointCloud(vertices=s_vertices, labels=labels)
    return result
