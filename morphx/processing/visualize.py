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
from getkey import getkey
from morphx.processing import clouds
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.pointcloud import PointCloud


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

    pcd1 = build_pcd(cloud1, random_seed=random_seed)
    pcd2 = build_pcd(cloud2, random_seed=random_seed)

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


def visualize_clouds(cloud_list: list, capture: bool = False, path="", random_seed: int = 4):
    """ Uses open3d to visualize given point clouds in a single window or save the clouds without showing.

    Args:
        cloud_list: List of MorphX PointCloud objects which should be visualized.
        capture: Flag to only save screenshot without showing the cloud.
        path: filepath where screenshot should be saved.
        random_seed: flag for using the same colors.
    """

    pcd = build_pcd(cloud_list, random_seed=random_seed)
    core_visualizer(pcd, capture=capture, path=path)


def visualize_skeleton(hc: HybridCloud, capture: bool = False, path="", random_seed: int = 4):
    """ Uses open3d to visualize the skeleton of the given hybrid cloud in a single window or save the visualization
        without showing.

    Args:
        hc: The HybridCloud whose skeleton should be visualized.
        capture: Flag to only save screenshot without showing the cloud.
        path: filepath where screenshot should be saved.
        random_seed: flag for using the same colors.
    """
    pc = PointCloud(hc.nodes, labels=np.zeros(len(hc.nodes)))
    pcd = build_pcd([pc], random_seed=random_seed)
    core_visualizer(pcd, capture=capture, path=path)


def visualize_bfs(hc: HybridCloud, bfs: np.ndarray, capture: bool = False, path="", random_seed: int = 4):
    """ Uses open3d to visualize the result of a breadth first search on the skeleton of the given HybridCloud. The
        image can be saved without showing.

    Args:
        hc: The HybridCloud on whose skeleton the BFS was performed
        bfs: The result of the BFS in form of an array of node indices.
        capture: Flag to only save screenshots without showing the cloud.
        path: filepath where screenshot should be saved.
        random_seed: flag for using the same colors.
    """
    nodes = hc.nodes
    pure_skel = np.delete(nodes, bfs)
    pure_skel = PointCloud(pure_skel, labels=np.zeros(len(pure_skel)))
    bfs_skel = nodes[bfs]
    bfs_skel = PointCloud(bfs_skel, labels=np.ones(len(bfs_skel)))

    pcd = build_pcd([pure_skel, bfs_skel], random_seed=random_seed)
    core_visualizer(pcd, capture=capture, path=path)


def core_visualizer(pcd: o3d.geometry.PointCloud, capture: bool, path: str):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)

    if capture:
        vis.capture_screen_image(path, True)
    else:
        vis.run()


def build_pcd(cloud_list: list, random_seed: int) -> o3d.geometry.PointCloud:
    """ Builds an Open3d point cloud object out of the given list of morphx PointClouds.

    Args:
        cloud_list: List of MorphX PointCloud objects which should be visualized.
        random_seed: flag for using the same colors.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # merge all clouds in cloud_list
    merged = None
    for cloud in cloud_list:
        if merged is None:
            merged = cloud
        else:
            merged = clouds.merge_clouds(merged, cloud)

    labels = merged.labels
    vertices = merged.vertices

    # add 3D points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # assign colors if labels exist
    if labels is not None:
        labels = labels.reshape(len(labels))
        label_num = int(max(np.unique(labels)) + 1)

        # generate colors
        colors = np.random.choice(range(256), size=(label_num, 3)) / 255
        colors = colors[labels.astype(int)]

        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd
