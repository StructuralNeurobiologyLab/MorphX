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


def visualize_clouds(clouds: list, capture=False, path=""):
    """Uses open3d to visualize a given point cloud in a new window or save the cloud without showing.

    Args:
        clouds: List of MorphX PointCloud objects which should be visualized.
        capture: Flag to only save screenshot without showing the cloud.
        path: filepath where screenshot should be saved.
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    for cloud in clouds:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.vertices)

        if cloud.labels is not None:
            # dendrite, axon, soma, bouton, terminal
            colors = [[0, 48, 73], [214, 40, 40], [247, 127, 0], [252, 191, 73], [234, 226, 183]]
            colors = (np.array(colors) / 255)
            colors = colors[cloud.labels][:, 0]
            pcd.colors = o3d.utility.Vector3dVector(colors)

        vis.add_geometry(pcd)
    if capture:
        vis.capture_screen_image(path, True)
    else:
        vis.run()
