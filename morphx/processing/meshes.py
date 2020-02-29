# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import open3d as o3d
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridmesh import HybridMesh
from scipy.spatial import cKDTree


# -------------------------------------- MESH SAMPLING ------------------------------------------- #

def sample_mesh_poisson_disk(hm: HybridMesh, sample_num: int) -> PointCloud:
    """ Uses poisson disk sampling and maps existing labels using a KDTree.

    Args:
        hm: The MeshCloud from which the samples should be generated
        sample_num: Requested number of sample points.

    Returns:
        PointCloud consisting of sampled points.
    """
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(hm.vertices)
    mesh.triangles = o3d.utility.Vector3iVector(hm.faces)
    sample = mesh.sample_points_poisson_disk(int(sample_num))
    s_vertices = np.asarray(sample.points)

    # map labels from input cloud to sample
    labels = None
    if hm.labels is not None:
        tree = cKDTree(hm.vertices)
        dist, ind = tree.query(s_vertices, k=1)
        labels = hm.labels[ind]

    result = PointCloud(vertices=s_vertices, labels=labels)
    return result
