# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
import open3d as o3d


def sample_cloud(cloud: np.ndarray, vertex_number: int, random_seed=None) -> np.ndarray:
    """ Creates a (pseudo)random sample point cloud with a specific number of points from the given subset of mesh
    vertices. If the requested number of points is larger than the given subset, the subset gets enriched with its own
    augmented points before sampling.

    Args:
        cloud: An array of mesh vertices with shape (n,3).
        vertex_number: The number of points which should make up the sample point cloud.
        random_seed: Possibility for making the sampling deterministic.

    Returns:
        Array of sampled points
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    deficit = vertex_number - len(cloud)

    vert_ixs = np.arange(len(cloud))
    np.random.shuffle(vert_ixs)

    sample = cloud[vert_ixs[:vertex_number]]

    compensation = []
    while len(compensation) < deficit:
        np.random.shuffle(vert_ixs)
        add_comp = cloud[vert_ixs[:deficit - len(compensation)]]
        compensation.append(add_comp)

    compensation = np.array(compensation)
    # TODO: change to augmentation method from elektronn3
    aug_comp = compensation + np.random.random(compensation.shape)

    return np.concatenate((sample, aug_comp), axis=0)


def center_cloud(cloud: np.ndarray, normalize=False) -> np.ndarray:
    """ Centers (and normalizes) point cloud.

    Args:
        cloud: Point cloud as array of coordinates.
        normalize: flag for optional normalization of the cloud.

    Returns:
        Centered and normalized point cloud as array of coordinates.
    """
    centroid = np.mean(cloud, axis=0)
    c_cloud = cloud - centroid

    if normalize:
        c_cloud = c_cloud / np.linalg.norm(c_cloud)

    return c_cloud


def save_cloud(cloud: np.ndarray, path: str) -> bool:
    """Saves point cloud to file at given path (e.g. as ply file)

    Args:
        cloud: Point cloud as array of coordinates.
        path: string which describes path where cloud should be saved (e.g. '~/home/cloud.ply')

    Returns:
        True if saving was successful, False if not
    '"""
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
        clouds: Point cloud as array of coordinates.
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
