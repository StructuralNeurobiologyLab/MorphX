# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import pickle
from typing import Union
import numpy as np
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.pointcloud import PointCloud
from morphx.processing import hybrids, ensembles


# -------------------------------------- OBJECT I/O ------------------------------------------- #


def save2pkl(obj: object, path: str, name='object') -> int:
    """ Dumps given object into pickle file at given path.

    Args:
        obj: Object which should be saved.
        path: Folder where the object should be saved to.
        name: Name of file in which the object should be saved.

    Returns:
        0 if saving process was successful, 1 otherwise.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        path = os.path.join(path, name + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        f.close()
    except FileNotFoundError:
        print("Saving was not successful as given path is not valid.")
        return 1
    return 0


def load_pkl(path):
    """ Loads an object from an existing pickle file.

    Args:
        path: File path of pickle file.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print(f"File with name: {path} was not found at this location.")
    with open(path, 'rb') as f:
        obj = pickle.load(f)
    f.close()
    return obj


# -------------------------------------- DISPATCHER METHODS ------------------------------------------- #


def extract_cloud_subset(obj: Union[HybridCloud, CloudEnsemble], local_bfs: np.ndarray) -> PointCloud:
    if isinstance(obj, HybridCloud):
        return hybrids.extract_subset(obj, local_bfs)
    if isinstance(obj, CloudEnsemble):
        return ensembles.extract_subset(obj, local_bfs)