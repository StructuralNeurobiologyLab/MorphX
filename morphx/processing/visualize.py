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


def visualize_clouds(clouds: list, capture=False, path="", random_seed: int = 4):
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
    for cloud in clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.vertices)

        if cloud.labels is not None:
            # generate colors
            label_num = len(np.unique(cloud.labels))
            colors = np.random.choice(range(256), size=(label_num, 3)) / 255
            colors = colors[cloud.labels]
            pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.add_geometry(pcd)
    if capture:
        vis.capture_screen_image(path, True)
    else:
        vis.run()
