# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
from morphx.classes.pointcloud import PointCloud


def get_variation(pc: PointCloud):
    var = np.unique(pc.labels, return_counts=True)
    return var[1] / len(pc.labels)
