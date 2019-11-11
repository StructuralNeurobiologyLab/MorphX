# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import math
import numpy as np
import open3d as o3d
from morphx.classes.pointcloud import PointCloud


def sample_cloud(cloud: np.ndarray, vertex_number: int, random_seed=None) -> np.ndarray:
    """ Creates a (pseudo)random sample point cloud with a specific number of points from the given subset of mesh
    vertices. If the requested number of points is larger than the given subset, the subset gets enriched with its own
    augmented points before sampling.

    Args:
        cloud:
        vertex_number: The number of points which should make up the sample point cloud.
        random_seed: Possibility for making the sampling deterministic.

    Returns:
        Array of sampled points and their labels
    """
    if len(cloud) == 0:
        return cloud

    if random_seed is not None:
        np.random.seed(random_seed)

    dim = cloud.shape[1]
    sample = np.zeros((vertex_number, dim))
    # if labels is not None:
    #     sample_lab = np.zeros((vertex_number, 1))
    # else:
    #     sample_lab = None
    deficit = vertex_number - len(cloud)

    vert_ixs = np.arange(len(cloud))
    np.random.shuffle(vert_ixs)
    sample[:min(len(cloud), vertex_number)] = cloud[vert_ixs[:vertex_number]]
    # if labels is not None:
    #     sample_lab[:min(len(cloud), vertex_number)] = labels[vert_ixs[:vertex_number]]

    # add augmented points to reach requested number of samples
    if deficit > 0:
        # deficit could be bigger than cloudsize
        offset = len(cloud)
        for it in range(math.ceil(deficit/len(cloud))):
            compensation = min(len(cloud), len(sample)-offset)
            np.random.shuffle(vert_ixs)
            sample[offset:offset+compensation] = cloud[vert_ixs[:compensation]]
            # if labels is not None:
            #     sample_lab[offset:offset + compensation] = cloud[vert_ixs[:compensation]]
            offset += compensation

        # TODO: change to augmentation method from elektronn3
        sample[len(cloud):] += np.random.random(sample[len(cloud)].shape)

    return sample


def center_cloud(pc: PointCloud, normalize=False) -> PointCloud:
    """ Centers (and normalizes) point cloud.

    Args:
        pc: A morphx point cloud object.
        normalize: flag for optional normalization of the cloud.

    Returns:
        MorphX PointCloud object with centered (and normalized) vertices.
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


def save_cloud(cloud: np.ndarray, path: str) -> bool:
    """Saves point cloud to file at given path (e.g. as ply file)

    Args:
        cloud: Point cloud as array of coordinates.
        path: string which describes path where cloud should be saved (e.g. '~/home/cloud.ply')

    Returns:
        True if saving was successful, False if not
    '"""
    # TODO: Rewrite method for saving MorphX PointCloud objects
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(cloud)
    if o3d.io.write_point_cloud(path, pc):
        return True
    else:
        print('Something went wrong when saving the point cloud')
        return False


def visualize_clouds(clouds: list, capture=False, path=""):
    """Uses open3d to visualize a given point cloud in a new window or save the cloud without showing.

    Args:
        clouds: List of MorphX PointCloud objects
        capture: Flag to only save screenshot without showing the cloud.
        path: filepath where screenshot should be saved.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for cloud in clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud)
        vis.add_geometry(pcd)
    if capture:
        vis.capture_screen_image(path, True)
    else:
        vis.run()
