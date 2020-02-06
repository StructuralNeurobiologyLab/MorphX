# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import pickle
from morphx.processing import clouds, graphs, objects


def split(data_path: str, chunk_size: int):
    """ Splits HybridClouds given as pickle files at data_path into multiple chunks and saves that chunking information
        in the new folder 'splitted' as a pickled dict. The dict has filenames of the HybridClouds as keys and lists of
        chunks as values. The chunks are saved as numpy arrays of the indices of skeleton nodes which belong to the
        respective chunk.

    Args:
        data_path: Path to HybridClouds saved as pickle files.
        chunk_size: Minimum distance between base points of different chunks. Used for a breadth first search as in
            :func:`morphx.graphs.global_bfs_dist`. Each base point is then the starting point for a local BFS defined
            in :func:`morphx.graphs.local_bfs_dist`, which takes `chunk_size`*1.5 as its maximum distance in order to
            get overlapping chunks.
    """
    data_path = os.path.expanduser(data_path)
    files = glob.glob(data_path + '*.pkl')

    data_size = 0
    splitted_hcs = {}
    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        hc = objects.load_pkl(file)
        hc.base_points(min_dist=chunk_size)
        data_size += len(hc.base_points())
        chunks = []
        for node in hc.base_points():
            local_bfs = graphs.local_bfs_dist(hc.graph(), node, chunk_size/1.5)
            chunks.append(local_bfs)
        splitted_hcs[name] = chunks

    output_path = data_path + 'splitted/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(output_path + str(chunk_size) + '.pkl', 'wb') as f:
        pickle.dump(splitted_hcs, f)
    f.close()
