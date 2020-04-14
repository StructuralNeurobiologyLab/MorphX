# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

""" This is separated from the clouds.py file because of bug in open3d when used with pytorch.
    See https://github.com/pytorch/pytorch/issues/21018 """

import os
import glob
from tqdm import tqdm
import open3d as o3d
import numpy as np
from typing import Union
from getkey import getkey
from morphx.data import basics
from morphx.processing import clouds, objects
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.pointcloud import PointCloud
from morphx.classes.cloudensemble import CloudEnsemble


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


def visualize_skeleton(hc: HybridCloud, colored: np.ndarray = None, bfs: np.ndarray = None):
    """ Uses open3d to visualize the skeleton of the given hybrid cloud in a single window or save the visualization
        without showing.

    Args:
        hc: The HybridCloud whose skeleton should be visualized.
        colored: indices of nodes which should appear in red.
        bfs: some point indices which should be highlighted
    """
    skel = o3d.geometry.PointCloud()
    skel.points = o3d.utility.Vector3dVector(hc.nodes)
    edges = o3d.geometry.LineSet()
    edges.points = o3d.utility.Vector3dVector(hc.nodes)
    edges.lines = o3d.utility.Vector2iVector(hc.edges)
    if colored is not None:
        colored = colored.reshape(-1).astype(int)
        colors = np.zeros((len(hc.nodes), 3))
        colors[colored.reshape(-1)] = np.array([1, 0, 0])
        skel.colors = o3d.utility.Vector3dVector(colors)
        edges.colors = o3d.utility.Vector3dVector(colors)
    pc = prepare_bfs(hc, bfs)
    pcd = build_pcd([pc], random_seed=4)
    o3d.visualization.draw_geometries([skel, edges, pcd])


def prepare_bfs(hc: HybridCloud, bfs: np.ndarray) -> PointCloud:
    """ Enriches the BFS result with small point cubes for better visualization.

    Args:
        hc: The HybridCloud on whose skeleton the BFS was performed
        bfs: The result of the BFS in form of an array of node indices.

    Returns:
        PointCloud with enriched BFS result.
    """

    nodes = hc.nodes
    bfs = bfs.astype(int)
    bfs_skel = nodes[bfs]
    # create small point cubes around BFS points for better visualization
    sphere_size = 1000
    size = len(bfs_skel)
    a_bfs_skel = np.zeros((size * sphere_size, 3))
    for i in range(sphere_size):
        a_bfs_skel[i * size:i * size + size] = bfs_skel
    a_bfs_skel += (np.random.random((len(a_bfs_skel), 3)) - 0.5) * 500

    labels = np.ones(len(a_bfs_skel))
    labels[:] = 9
    return PointCloud(a_bfs_skel, labels=labels)


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
    bfs = bfs.astype(int)
    pure_skel = np.delete(nodes, bfs, axis=0)
    pure_skel = PointCloud(pure_skel, labels=np.zeros(len(pure_skel)))

    bfs_skel = prepare_bfs(hc, bfs)

    pcd = build_pcd([pure_skel, bfs_skel], random_seed=random_seed)
    core_visualizer(pcd, capture=capture, path=path)


def visualize_training_set(set_path: str, gt_path: str):
    """ Saves images of all predicted files from different trainings using visualize_prediction_set for each
        training. """
    set_path = os.path.expanduser(set_path)
    gt_path = os.path.expanduser(gt_path)
    dirs = os.listdir(set_path)
    for di in dirs:
        path = set_path + di + '/'
        visualize_prediction_set(path, path + 'images/', gt_path)


def visualize_prediction_set(input_path: str, output_path: str, gt_path: str, random_seed: int = 4,
                             data_type: str = 'ce'):
    """ Saves images of all predicted files at input_path using the visualize_prediction method for each file. """
    files = glob.glob(input_path + 'sso_*.pkl')
    gt_files = glob.glob(gt_path + 'sso_*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    # find corresponding gt file, map predictions to labels and save images
    for file in tqdm(files):
        for gt_file in gt_files:
            slashs = [pos for pos, char in enumerate(gt_file) if char == '/']
            name = gt_file[slashs[-1] + 1:-4]
            if name in file:
                obj = objects.load_obj(data_type, gt_file)
                pred = basics.load_pkl(file)
                obj.set_predictions(pred[1])
                visualize_prediction(obj, output_path + name + '.png', random_seed=random_seed)


def visualize_data_set(input_path: str, output_path: str, random_seed: int = 4, data_type: str = 'ce'):
    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        obj = objects.load_obj(data_type, file)
        visualize_clouds([obj], capture=True, path=output_path + name + '.png', random_seed=random_seed)


def visualize_prediction(obj: Union[CloudEnsemble, HybridCloud], out_path: str, random_seed: int = 4):
    """ Saves images of prediction of given file in out_path. First, the predictions get reduced onto the labels.
        Labels without predictions get filtered. Second, the vertex labels are mapped onto the skeleton, where
        nodes without vertices take on the label of the nearest neighbor with vertices. Third, the node labels get
        mapped onto the mesh again. Then an image of the mesh is saved to file.

    Args:
          obj: MorphX object with predictions
          out_path: Folder where image should be saved
          random_seed: Defines the color coding
    """
    obj.generate_pred_labels()
    if isinstance(obj, CloudEnsemble):
        obj = obj.hc
    # reduce object to vertices where predictions exist
    mask = obj.pred_labels != -1
    mask = mask.reshape(-1)
    red_obj = HybridCloud(nodes=obj.nodes, edges=obj.edges, vertices=obj.vertices[mask], labels=obj.labels[mask],
                          pred_labels=obj.pred_labels[mask])
    obj.set_pred_node_labels(red_obj.pred_node_labels)
    obj.prednodel2predvertl()
    obj.set_labels(obj.pred_labels)
    visualize_clouds([obj], capture=True, path=out_path, random_seed=random_seed)


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
        if isinstance(cloud, CloudEnsemble):
            cloud = cloud.flattened
        if merged is None:
            merged = cloud
        else:
            merged = clouds.merge_clouds([merged, cloud])

    labels = merged.labels
    vertices = merged.vertices

    # add 3D points
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(vertices)

    # assign colors if labels exist
    if labels is not None and len(labels) != 0:
        labels = labels.reshape(len(labels))
        label_num = int(max(np.unique(labels)) + 1)

        # generate colors (either fixed or randomly)
        if label_num <= 10:
            colors = np.array([[122, 174, 183], [197, 129, 104], [87, 200, 50], [137, 58, 252],
                               [133, 1, 1], [107, 114, 219], [4, 52, 124], [46, 41, 78],
                               [46, 41, 78], [46, 41, 78]]) / 255
        else:
            colors = np.random.choice(range(256), size=(label_num, 3)) / 255
        colors = colors[labels.astype(int)]

        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def core_visualizer(pcd: o3d.geometry.PointCloud, capture: bool, path: str):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    frame = o3d.geometry.TriangleMesh().create_coordinate_frame(10000)
    frame.paint_uniform_color(np.array([0, 0, 0]))
    vis.add_geometry(frame)

    if capture:
        vis.capture_screen_image(path, True)
    else:
        vis.run()
