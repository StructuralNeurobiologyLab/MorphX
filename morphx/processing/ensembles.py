# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import numpy as np
import pickle as pkl
from math import ceil
from typing import Optional, Tuple
from morphx.processing import clouds
from morphx.classes.pointcloud import PointCloud
from morphx.classes.cloudensemble import CloudEnsemble


# -------------------------------------- ENSEMBLE SAMPLING ----------------------------------------- #


def sample_ensemble(ensemble: CloudEnsemble, vertex_number: int, random_seed: Optional[int] = None) \
        -> Tuple[Optional[PointCloud], dict]:
    """ Samples ensemble parts with respect to their vertex number. Each cloud in the ensemble gets
        len(cloud.vertices)/len(ensemble.vertices)*vertex_number points (ceiled if possible), where
        len(ensemble.vertices) is just the total number of vertices from all ensemble clouds. The
        samples from the different clouds are merged into one PointCloud, the cloud information
        gets saved in the obj_bounds dict of that PointCloud.

    Args:
        ensemble: The ensemble from whose objects the samples should be drawn.
        vertex_number: The number of requested sample points.
        random_seed: A random seed to make sampling deterministic.

    Returns:
        PointCloud object with ensemble cloud information in obj_bounds. Dict with ensemble object names
        as keys and indices of the samples drawn from this object as np.arrays. 
    """
    total = 0
    for key in ensemble.clouds.keys():
        total += len(ensemble.clouds[key].vertices)
    if total == 0:
        return None, {}
    current = 0
    result = None
    result_ixs = {}
    for key in ensemble.clouds.keys():
        verts = len(ensemble.clouds[key].vertices)/total*vertex_number
        if current + ceil(verts) <= vertex_number:
            verts = ceil(verts)
        sample, ixs = clouds.sample_cloud(ensemble.clouds[key], verts, random_seed=random_seed)
        result_ixs[key] = ixs
        if result is None:
            result = sample
        else:
            result = clouds.merge_clouds(result, sample, name2=key)
    return result, result_ixs


# -------------------------------------- ENSEMBLE I/O ------------------------------------------- #

# TODO: add saving in simple mode
def load_ensemble(path: str) -> Optional[CloudEnsemble]:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print("File with name: {} was not found at this location.".format(path))

    with open(path, 'rb') as f:
        obj = pkl.load(f)
    f.close()

    if isinstance(obj, CloudEnsemble):
        return obj
    else:
        return None


def save_ensemble(ensemble: CloudEnsemble, folder: str, name: str) -> int:
    full_path = os.path.join(folder, name + '.pkl')
    try:
        if not os.path.exists(folder):
            os.makedirs(folder)
        with open(full_path, 'wb') as f:
            pkl.dump(ensemble, f)
        f.close()
    except FileNotFoundError:
        print("Saving was not successful as given path is not valid.")
        return 1
    return 0
