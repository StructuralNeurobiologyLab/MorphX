# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import math
import os
import pickle
import numpy as np
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud


def sample_cloud(pc: PointCloud, vertex_number: int, random_seed=None) -> PointCloud:
    """ Creates a (pseudo)random sample point cloud with a specific number of points from the given subset of mesh
    vertices. If the requested number of points is larger than the given subset, the subset gets enriched with slightly
    augmented points before sampling.

    Args:
        pc: MorphX PointCloud object which should be sampled.
        vertex_number: The number of points which should make up the sample point cloud.
        random_seed: Possibility for making the sampling deterministic.

    Returns:
        PointCloud with sampled points (and labels).
    """
    cloud = pc.vertices
    labels = pc.labels

    if len(cloud) == 0:
        return pc

    if random_seed is not None:
        np.random.seed(random_seed)

    dim = cloud.shape[1]
    sample = np.zeros((vertex_number, dim))
    sample_l = np.zeros((vertex_number, 1))

    deficit = vertex_number - len(cloud)

    vert_ixs = np.arange(len(cloud))
    np.random.shuffle(vert_ixs)
    sample[:min(len(cloud), vertex_number)] = cloud[vert_ixs[:vertex_number]]
    if labels is not None:
        sample_l[:min(len(cloud), vertex_number)] = labels[vert_ixs[:vertex_number]]

    # add augmented points to reach requested number of samples
    if deficit > 0:
        # deficit could be bigger than cloudsize
        offset = len(cloud)
        for it in range(math.ceil(deficit/len(cloud))):
            compensation = min(len(cloud), len(sample)-offset)
            np.random.shuffle(vert_ixs)
            sample[offset:offset+compensation] = cloud[vert_ixs[:compensation]]
            if labels is not None:
                sample_l[offset:offset+compensation] = cloud[vert_ixs[:compensation]]
            offset += compensation

        # TODO: change to augmentation method from elektronn3
        sample[len(cloud):] += np.random.random(sample[len(cloud)].shape)

    if labels is not None:
        return PointCloud(sample, labels=sample_l)
    else:
        return PointCloud(sample)


def center_cloud(pc: PointCloud, normalize=False) -> PointCloud:
    """ Centers (and normalizes) point cloud.

    Args:
        pc: MorphX PointCloud object which should be centered.
        normalize: flag for optional normalization of the cloud.

    Returns:
        PointCloud object with centered (and normalized) points.
    """
    cloud = pc.vertices

    centroid = np.mean(cloud, axis=0)
    c_cloud = cloud - centroid

    if normalize:
        c_cloud = c_cloud / np.linalg.norm(c_cloud)

    return PointCloud(c_cloud, labels=pc.labels)


def merge_clouds(pc1: PointCloud, pc2: PointCloud) -> PointCloud:
    """ Merges 2 PointCloud Objects if dimensions match and if either both clouds have labels or none has.

    Args:
        pc1: First PointCloud object
        pc2: Second PointCloud object

    Returns:
        PointCloud object which was build by merging the given two clouds.
    """
    dim1 = pc1.vertices.shape[1]
    dim2 = pc2.vertices.shape[1]
    if dim1 != dim2:
        raise Exception("PointCloud dimensions do not match")

    merged_vertices = np.zeros((len(pc1.vertices)+len(pc2.vertices), dim1))
    merged_labels = np.zeros(merged_vertices.shape)
    merged_vertices[:len(pc1.vertices)] = pc1.vertices
    merged_vertices[len(pc1.vertices):] = pc2.vertices

    if pc1.labels is None and pc2.labels is None:
        return PointCloud(merged_vertices)
    elif pc1.labels is None or pc2.labels is None:
        raise Exception("PointCloud label is None at one PointCloud but exists at the other. "
                        "PointClouds are not compatible")
    else:
        merged_labels[:len(pc1.vertices)] = pc1.labels
        merged_labels[len(pc1.vertices):] = pc2.labels
        return PointCloud(merged_vertices, labels=merged_labels)


def save_cloud(cloud: PointCloud, path, name='cloud', simple=True) -> int:
    """ Saves PointCloud object to given path.

    Args:
        cloud: PointCloud object which should be saved.
        path: Location where the object should be saved to (only folder).
        name: Name of file in which the object should be saved.
        simple: If object is also HybridCloud, then this flag can be used to only save basic information instead of the
            total object.

    Returns:
        1 if saving process was successful, 0 otherwise.
    """
    path = os.path.join(path, name + '.pkl')
    try:
        with open(path, 'wb') as f:
            if isinstance(cloud, HybridCloud) and simple:
                cloud = {'nodes': cloud.nodes, 'edges': cloud.edges, 'vertices': cloud.vertices,
                         'labels': cloud.labels}
            pickle.dump(cloud, f)
    except FileNotFoundError:
        print("Saving was not successful as given path is not valid.")
        return 0
    return 1


def load_cloud(path) -> PointCloud:
    """ Loads and returns PointCloud object from given path. """

    with open(path, 'rb') as f:
        cloud = pickle.load(f)

    if isinstance(cloud, dict):
        return HybridCloud(cloud['nodes'], cloud['edges'], cloud['vertices'], labels=cloud['labels'])

    return cloud
