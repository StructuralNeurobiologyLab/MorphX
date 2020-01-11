# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
from tqdm import tqdm
from morphx.processing import meshes, clouds, graphs, hybrids, ensembles
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
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]
        ce = None
        try:
            hm = clouds.load_cloud(file)
        except ValueError:
            ce = ensembles.load_ensemble(file)
            hm = ce.get_cloud('cell')
        print("Processing file: " + name)
        hc = hybridmesh2poisson(hm)
        if ce is None:
            clouds.save_cloud(hc, output_path, name=name+'_poisson')
        else:
            ce.set_cloud(hc, 'cell')
            ensembles.save_ensemble(ce, output_path, name=name+'_poisson')


def process_single_thread(args):
    """ Converts single pickle file into poisson disk sampled HybridCloud.

    Args:
        args:
            args[0] - file: The full path to a specific pickle file
            args[1] - output_path: The folder where the result should be stored.
    """

    file = args[0]
    output_path = args[1]

    slashs = [pos for pos, char in enumerate(file) if char == '/']
    name = file[slashs[-1] + 1:-4]

    hm = clouds.load_cloud(file)
    # hc = hybridmesh2poisson(hm)
    clouds.save_cloud(hm, output_path, name=name+'_poisson')


def hybridmesh2poisson(hm: HybridMesh) -> HybridCloud:
    """ Extracts base points on skeleton of hm with a global BFS, performs local BFS around each base point and samples
        the vertices cloud around the local BFS result with poisson disk sampling. The uniform distances of the samples
        are not guaranteed, as the total sample cloud will always be constructed from two different sample processes
        which have their individual uniform distance.

    Args:
        hm: HybridMesh which should be transformed into a HybridCloud with poisson disk sampled points.
    """

    total_pc = None
    distance = 1000
    skel2node_mapping = True
    for base in tqdm(hm.traverser(min_dist=distance)):
        # local BFS radius = global BFS radius, so that the total number of poisson sampled vertices will double.
        local_bfs = graphs.local_bfs_dist(hm.graph(), source=base, max_dist=distance)
        if skel2node_mapping:
            print("Mapping skeleton to node for further processing. This might take a few minutes...")
            skel2node_mapping = False
        mc = hybrids.extract_mesh_subset(hm, local_bfs)
        if len(mc.faces) == 0:
            continue

        pc = meshes.sample_mesh_poisson_disk(mc, len(mc.vertices))
        if total_pc is None:
            total_pc = pc
        else:
            total_pc = clouds.merge_clouds(total_pc, pc)

    hc = HybridCloud(hm.nodes, hm.edges, total_pc.vertices, labels=total_pc.labels, encoding=total_pc.encoding)
    return hc


if __name__ == '__main__':
    process_dataset('/u/jklimesch/gt/gt_ensembles/batch2/', '/u/jklimesch/gt/gt_ensembles/')
