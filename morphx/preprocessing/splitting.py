# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import pickle
from morphx.processing import clouds, graphs


def split(data_path: str, chunk_size: int):
    data_path = os.path.expanduser(data_path)
    files = glob.glob(data_path + '*.pkl')

    data_size = 0
    splitted_hcs = {}
    for file in files:
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        hc = clouds.load_cloud(file)

        hc.traverser(min_dist=chunk_size)
        data_size += len(hc.traverser())

        chunks = []
        for node in hc.traverser():
            local_bfs = graphs.local_bfs_dist(hc.graph(), node, chunk_size/1.5)
            chunks.append(local_bfs)
        splitted_hcs[name] = chunks

    output_path = data_path + 'splitted/'
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(output_path + str(chunk_size) + '.pkl', 'wb') as f:
        pickle.dump(splitted_hcs, f)
    f.close()
