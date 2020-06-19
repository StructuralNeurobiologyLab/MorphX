# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import numpy as np
import open3d as o3d
from tqdm import tqdm
from typing import Union
from morphx.processing import ensembles
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.pointcloud import PointCloud


def voxel_down_dataset(input_path: str, output_path: str, voxel_size: Union[dict, int] = 500):
    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    print("Starting to voxel down dataset...")
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]

        ce = ensembles.ensemble_from_pkl(file)
        ce = voxel_down(ce, voxel_size=voxel_size)
        ce.save2pkl(output_path + name + '.pkl')


def voxel_down(ce: CloudEnsemble, voxel_size: Union[dict, int] = 500) -> CloudEnsemble:
    if type(voxel_size) is not dict:
        voxel_size = dict(hc=voxel_size)
        for k in ce.clouds:
            voxel_size[k] = voxel_size['hc']
    hc = ce.hc
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hc.vertices)
    pcd, idcs = pcd.voxel_down_sample_and_trace(voxel_size['hc'], pcd.get_min_bound(), pcd.get_max_bound())
    idcs = np.max(idcs, axis=1)
    new_labels = None
    new_types = None
    new_features = None
    if len(hc.labels) != 0:
        new_labels = hc.labels[idcs]
    if len(hc.types) != 0:
        new_types = hc.types[idcs]
    if len(hc.features) != 0:
        new_features = hc.features[idcs]
    new_hc = HybridCloud(hc.nodes, hc.edges, vertices=np.asarray(pcd.points), labels=new_labels,
                         types=new_types, features=new_features, encoding=hc.encoding,
                         node_labels=hc.node_labels, no_pred=hc.no_pred)
    new_clouds = {}
    for key in ce.clouds:
        pc = ce.clouds[key]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ce.clouds[key].vertices)
        pcd, idcs = pcd.voxel_down_sample_and_trace(voxel_size[key], pcd.get_min_bound(), pcd.get_max_bound())
        idcs = np.max(idcs, axis=1)
        new_pc = PointCloud(np.asarray(pcd.points), labels=pc.labels[idcs], encoding=pc.encoding, no_pred=pc.no_pred)
        new_clouds[key] = new_pc

    return CloudEnsemble(new_clouds, new_hc, no_pred=ce.no_pred)
