# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
from typing import Union
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.classes.pointcloud import PointCloud
from morphx.processing import ensembles, hybrids


# -------------------------------------- EXTRACTION METHODS ------------------------------------------- #

def extract_cloud_subset(obj: Union[HybridCloud, CloudEnsemble], local_bfs: np.ndarray) -> PointCloud:
    if isinstance(obj, HybridCloud):
        return hybrids.extract_subset(obj, local_bfs)
    if isinstance(obj, CloudEnsemble):
        return ensembles.extract_subset(obj, local_bfs)
