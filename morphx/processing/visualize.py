# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

""" This is separated from the clouds.py file because of bug in open3d when used with pytorch.
    See https://github.com/pytorch/pytorch/issues/21018 """

import open3d as o3d
import numpy as np
import ipdb


def visualize_clouds(clouds: list, capture: bool = False, path="", random_seed: int = 4):
    """Uses open3d to visualize a given point cloud in a new window or save the cloud without showing.

    Args:
        clouds: List of MorphX PointCloud objects which should be visualized.
        capture: Flag to only save screenshot without showing the cloud.
        path: filepath where screenshot should be saved.
        random_seed: flag for using the same colors.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # count total number of vertices
    vertex_num = 0
    label_count = 0
    for cloud in clouds:
        vertex_num += len(cloud.vertices)
        if cloud.labels is not None:
            label_count += 1

    if label_count != len(clouds) and label_count != 0:
        raise ValueError("Cannot display clouds with labels along clouds without labels.")

    # prepare arrays
    vertices = np.zeros((vertex_num, 3))
    labels = np.zeros((vertex_num, 1))

    # write vertices and labels from all clouds into the prepared arrays
    offset = 0
    for cloud in clouds:
        vertices[offset:offset+len(cloud.vertices), :] = cloud.vertices
        if cloud.labels is not None:
            # label array has same dimensions as vertices
            labels[offset:offset+len(cloud.vertices)] = cloud.labels
        offset += len(cloud.vertices)

    labels = labels.reshape(len(labels))

    # count labels in all clouds
    label_num = int(max(np.unique(labels)) + 1)

    # generate colors
    colors = np.random.choice(range(256), size=(label_num, 3)) / 255
    colors = colors[labels.astype(int)]

    # visualize result
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    if label_num > 1:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    vis.add_geometry(pcd)

    if capture:
        vis.capture_screen_image(path, True)
    else:
        vis.run()
