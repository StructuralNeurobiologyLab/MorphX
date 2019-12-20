# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
from tqdm import tqdm
from morphx.processing import meshes, clouds, graphs, hybrids
from morphx.classes.hybridmesh import HybridMesh
from morphx.classes.hybridcloud import HybridCloud


def process_dataset(input_path: str, output_path: str):
    """ Converts all HybridMeshs, saved as pickle files at input_path, into poisson disk sampled HybridClouds and
        saves them at output_path with the same names.

    Args:
        input_path: Path to pickle files with HybridMeshs.
        output_path: Path to folder in which results should be stored.
    """

    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    print("Starting to transform mesh dataset into poisson dataset...")
    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]

        hm = meshes.load_mesh_gt(file)
        hc = hybridmesh2poisson(hm)
        clouds.save_cloud(hc, output_path, name=name)


def process_single(args):
    """ Converts single pickle file into poisson disk sampled HybridCloud.

    Args:
        args: (file, output_path) with file as the full path to a specific pickle file and output_path as the folder
            where the result should be stored.
    """

    file, output_path = args
    slashs = [pos for pos, char in enumerate(file) if char == '/']
    name = file[slashs[-1] + 1:-4]

    hm = meshes.load_mesh_gt(file)
    hc = hybridmesh2poisson(hm)
    clouds.save_cloud(hc, output_path, name=name)


def hybridmesh2poisson(hm: HybridMesh) -> HybridCloud:
    total_pc = None
    distance = 1000
    for base in tqdm(hm.traverser(min_dist=distance)):
        local_bfs = graphs.local_bfs_dist(hm.graph(), source=base, max_dist=distance)
        mc, new_vertices, new_labels = hybrids.extract_mesh_subset(hm, local_bfs)
        if len(mc.faces) == 0:
            continue

        pc = meshes.sample_mesh_poisson_disk(mc, new_vertices, new_labels, len(new_vertices))
        if total_pc is None:
            total_pc = pc
        else:
            total_pc = clouds.merge_clouds(total_pc, pc)
    hc = HybridCloud(hm.nodes, hm.edges, total_pc.vertices, labels=total_pc.labels, encoding=total_pc.encoding)
    return hc
