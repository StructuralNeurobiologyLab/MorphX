# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import pickle
import point_cloud_utils as pcu
from morphx.processing import clouds
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.meshcloud import MeshCloud
from morphx.classes.pointcloud import PointCloud
from scipy.spatial import cKDTree


# -------------------------------------- MESH SAMPLING ------------------------------------------- #

def sample_mesh_poisson_disk(mesh: MeshCloud, sample_num) -> PointCloud:
    """ Uses Point-Disk-Sampling as described at https://github.com/fwilliams/point-cloud-utils and maps existing
        labels using a KDTree.

    Args:
        mesh: The MeshCloud from which the samples should be generated
        sample_num: Requested number of sample points.

    Returns:
        PointCloud consisting of sampled points.
    """
    vertices = mesh.vertices.astype(float)
    s_vertices, s_normals = pcu.sample_mesh_poisson_disk(vertices, mesh.faces, mesh.normals, sample_num)

    # TODO: This can be improved
    labels = None
    # map labels from input cloud to sample
    if mesh.labels is not None:
        tree = cKDTree(mesh.vertices)
        dist, ind = tree.query(s_vertices, k=1)
        labels = mesh.labels[ind]

    # sample again, as pcu.sample doesn't always return the requested number of samples
    # (only a few points must be added)
    spc = clouds.sample_cloud(PointCloud(s_vertices, labels=labels), sample_num)

    return spc


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
