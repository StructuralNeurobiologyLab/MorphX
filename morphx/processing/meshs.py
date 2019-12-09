# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import pickle
import point_cloud_utils as pcu
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.meshcloud import MeshCloud
from morphx.classes.pointcloud import PointCloud


# -------------------------------------- MESH SAMPLING ------------------------------------------- #

def sample_mesh_poisson_disk(mesh: MeshCloud, sample_num) -> PointCloud:
    """ Uses Point-Disk-Sampling as described at https://github.com/fwilliams/point-cloud-utils

    Args:
        mesh: The MeshCloud from which the samples should be generated
        sample_num: Requested number of sample points.

    Returns:
        PointCloud consisting of sampled points.
    """
    s_vertices, s_normals = pcu.sample_mesh_poisson_disk(mesh.vertices, mesh.faces, mesh.normals, sample_num)
    return PointCloud(s_vertices)


# -------------------------------------- MESH I/O ------------------------------------------- #


def load_mesh_gt(path: str) -> HybridMesh:
    """ Loads MeshHybrid from a pickle file at the given path. Pickle files should contain a dict with the
    keys: 'nodes', 'edges', 'vertices', 'indices', 'normals', 'labels' and 'encoding', representing skeleton nodes
    and edges, mesh vertices, indices, normals and labels and the respective label encoding. """

    path = os.path.expanduser(path)
    with open(path, "rb") as f:
        data = pickle.load(f)
    f.close()
    faces = data['indices']
    faces = faces.reshape((-1, 3))
    mh = HybridMesh(data['nodes'], data['edges'], data['vertices'], faces, data['normals'],
                    labels=data['labels'], encoding=data['encoding'])
    return mh
