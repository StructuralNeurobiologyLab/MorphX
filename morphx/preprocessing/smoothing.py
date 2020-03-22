# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import numpy as np
from tqdm import tqdm
from morphx.processing import ensembles, objects
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridcloud import HybridCloud
from typing import Union
from scipy.spatial import cKDTree


def smooth_dataset(input_path: str, output_path: str, data_type: str = 'ce'):
    files = glob.glob(input_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    print("Starting to smooth labels of dataset...")
    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]
        obj = objects.load_obj(data_type, file)
        obj = smooth_labels(obj)
        obj.save2pkl(output_path + name + '.pkl')


def smooth_labels(obj: Union[HybridCloud, CloudEnsemble], k: int = 20) -> Union[HybridCloud, CloudEnsemble]:
    if isinstance(obj, CloudEnsemble):
        hc = obj.hc
    else:
        hc = obj
    tree = cKDTree(hc.vertices)
    new_labels = hc.labels.copy()
    for ix, vertex in enumerate(tqdm(hc.vertices)):
        dist, ind = tree.query(vertex, k=k)
        neighbor_labels = hc.labels[ind]
        u_labels, counts = np.unique(neighbor_labels, return_counts=True)
        new_labels[ix] = u_labels[np.argmax(counts)]
    hc.set_labels(new_labels.astype(int))
    return obj


if __name__ == '__main__':
    smooth_dataset('/u/jklimesch/thesis/gt/20_03_18/',
                    '/u/jklimesch/thesis/gt/20_03_18/smoothed/')
