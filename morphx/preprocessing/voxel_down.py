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
from morphx.processing import ensembles
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.pointcloud import PointCloud


def process_dataset(input_path: str, output_path: str):
    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    print("Starting to voxel down dataset...")
    for file in tqdm(files):
        process_single_thread([file, output_path])


def process_single_thread(args):
    file = args[0]
    output_path = args[1]

    slashs = [pos for pos, char in enumerate(file) if char == '/']
    name = file[slashs[-1] + 1:-4]

    ce = ensembles.ensemble_from_pkl(file)
    ce = voxel_down(ce)
    ce.save2pkl(output_path + name + '.pkl')


def voxel_down(ce: CloudEnsemble) -> CloudEnsemble:
    voxel_size = 500
    hc = ce.hc
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(hc.vertices)
    pcd, idcs = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound())
    idcs = np.max(idcs, axis=1)
    new_hc = HybridCloud(hc.nodes, hc.edges, vertices=np.asarray(pcd.points), labels=hc.labels[idcs],
                         features=hc.labels[idcs], encoding=hc.encoding, node_labels=hc.node_labels, no_pred=hc.no_pred)
    new_clouds = {}
    for key in ce.clouds:
        pc = ce.clouds[key]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(ce.clouds[key].vertices)
        pcd, idcs = pcd.voxel_down_sample_and_trace(voxel_size, pcd.get_min_bound(), pcd.get_max_bound())
        idcs = np.max(idcs, axis=1)
        new_pc = PointCloud(np.asarray(pcd.points), labels=pc.labels[idcs], encoding=pc.encoding, no_pred=pc.no_pred)
        new_clouds[key] = new_pc

    return CloudEnsemble(new_clouds, new_hc, no_pred=ce.no_pred)


if __name__ == '__main__':
    process_dataset('/u/jklimesch/thesis/gt/gt_meshsets/raw/',
                    '/u/jklimesch/thesis/gt/gt_meshsets/voxeled_test/')
