# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

""" This is separated from the clouds.py file because of bug in open3d when used with pytorch.
    See https://github.com/pytorch/pytorch/issues/21018 """

import matplotlib.pyplot as plt
import open3d as o3d
import numpy as np
from morphx.processing import clouds
from getkey import getkey


def create_hist(labels: np.ndarray, save_path: str):
    plt.style.use('seaborn-white')
    bins = np.arange(min(labels) - 0.5, max(labels) + 1.5)
    plt.hist(labels, bins=bins, edgecolor='w')
    plt.xticks(np.arange(min(labels), max(labels) + 1))
    plt.grid()
    plt.title('Label distribution')
    plt.savefig(save_path)
    plt.close()


def visualize_parallel(cloud1: list, cloud2: list, static: bool = False, random_seed: int = 4,
                       name1: str = "cloud1", name2: str = "cloud2"):
    """Uses open3d to visualize two point clouds simultaneously.

    Args:
        cloud1: List of MorphX PointCloud objects which should be visualized.
        cloud2: Second list of MorphX PointCloud objects which should be visualized in parallel.
        random_seed: flag for using the same colors.
        static: Flag for chosing static or interactive view. The static view must be closed by entering
            a key into the console.
        name1: Display name for first cloud.
        name2: Display name for second cloud.
    """

    vis1 = o3d.visualization.Visualizer()
    vis1.create_window(window_name=name1, width=930, height=470, left=0, top=0)

    vis2 = o3d.visualization.Visualizer()
    vis2.create_window(window_name=name2, width=930, height=470, left=0, top=600)

    pcd1 = build_pcd(cloud1, random_seed)
    pcd2 = build_pcd(cloud2, random_seed)

    vis1.add_geometry(pcd1)
    vis2.add_geometry(pcd2)

    if static:
        while True:
            vis1.update_geometry()
            if not vis1.poll_events():
                break
            vis1.update_renderer()

            vis2.update_geometry()
            if not vis2.poll_events():
                break
            vis2.update_renderer()

            return getkey()
    else:
        while True:
            vis1.update_geometry()
            if not vis1.poll_events():
                break
            vis1.update_renderer()

            vis2.update_geometry()
            if not vis2.poll_events():
                break
            vis2.update_renderer()


def visualize_single(cloud_list: list, capture: bool = False, path="", random_seed: int = 4):
    """ Uses open3d to visualize a given point cloud in a new window or save the cloud without showing.

    Args:
        cloud_list: List of MorphX PointCloud objects which should be visualized.
        capture: Flag to only save screenshot without showing the cloud.
        path: filepath where screenshot should be saved.
        random_seed: flag for using the same colors.
    """

    vis = o3d.visualization.Visualizer()

    vis.create_window()

    pcd = build_pcd(cloud_list, random_seed)
    vis.add_geometry(pcd)

    if capture:
        vis.capture_screen_image(path, True)
    else:
        vis.run()


def build_pcd(cloud_list: list, random_seed: int = 4):
    """ Builds an Open3d point cloud object out of the given list of morphx PointClouds.

    Args:
        cloud_list: List of MorphX PointCloud objects which should be visualized.
        random_seed: flag for using the same colors.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    merged = None
    for cloud in cloud_list:
        if merged is None:
            merged = cloud
        else:
            merged = clouds.merge_clouds(merged, cloud)

    labels = merged.labels
    vertices = merged.vertices

    # count labels in all clouds
    labels = labels.reshape(len(labels))
    label_num = int(max(np.unique(labels)) + 1)

    # generate colors
    colors = np.random.choice(range(256), size=(label_num, 3)) / 255
    colors = colors[labels.astype(int)]

    # visualize result
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
