# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import pickle
from tqdm import tqdm
from morphx.processing import ensembles, hybrids, graphs


def split(data_path: str, base_distance: int, node_context: int, bio_density: float, capacity: int, tech_density: int,
          density: bool = True, chunk_size: int = None):
    """ Splits HybridClouds given as pickle files at data_path into multiple chunks and saves that chunking information
        in the new folder 'splitted' as a pickled dict. The dict has filenames of the HybridClouds as keys and lists of
        chunks as values. The chunks are saved as numpy arrays of the indices of skeleton nodes which belong to the
        respective chunk.

    Args:
        data_path: Path to HybridClouds saved as pickle files.
        base_distance: Distance between base points.
        node_context: Minimum distance between base points of different chunks. Used for a breadth first search as in
            :func:`morphx.graphs.global_bfs_dist`. Each base point is then the starting point for a local BFS defined
            in :func:`morphx.graphs.local_bfs_dist`, which takes `chunk_size`*1.5 as its maximum distance in order to
            get overlapping chunks.
        tech_density: poisson sampling density with which data set was preprocessed in point/um²
        bio_density: chunk sampling density in point/um²
        capacity: number of points which can get processed with given network architecture
        density: Flag for switching between density and context modes
        chunk_size: Is only used in context mode. Here, the chunks get generated by a certain size.
    """
    data_path = os.path.expanduser(data_path)
    files = glob.glob(data_path + '*.pkl')

    data_size = 0
    splitted_hcs = {}
    print("No splitting information exists for the given chunk size. Splitting of dataset is required.")
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        obj = ensembles.ensemble_from_pkl(file)
        obj.base_points(min_dist=base_distance)
        data_size += len(obj.base_points())
        chunks = []
        for node in obj.base_points():
            if density:
                vert_num = int(capacity*tech_density/bio_density)
                bfs = hybrids.bfs_vertices_euclid(obj, node, vert_num, node_context)
                chunks.append(bfs)
            else:
                if chunk_size is None:
                    raise ValueError('chunk_size parameter must be given in context mode.')
                local_bfs_small = graphs.bfs_euclid_sphere(obj.graph(), node, chunk_size / 1)
                chunks.append(local_bfs_small)
                local_bfs_medium = graphs.bfs_euclid_sphere(obj.graph(), node, chunk_size * 1.5)
                chunks.append(local_bfs_medium)
                local_bfs_large = graphs.bfs_euclid_sphere(obj.graph(), node, chunk_size * 1.2)
                chunks.append(local_bfs_large)
        splitted_hcs[name] = chunks

    output_path = data_path + 'splitted/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if density:
        filename = output_path + 'd_' + str(bio_density) + '.pkl'
    else:
        filename = output_path + 'c_' + str(chunk_size) + '.pkl'

    with open(filename, 'wb') as f:
        pickle.dump(splitted_hcs, f)
    f.close()
