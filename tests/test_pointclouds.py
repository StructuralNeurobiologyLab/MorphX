# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
from morphx.classes.pointcloud import PointCloud


def test_pred2labels():
    pc = PointCloud(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), labels=np.array([1, 2, 3]),
                    predictions={0: [0, 0, 0, 1], 1: [5, 4, 5, 5, 4, 3], 2: [7, 7, 7]})
    pc.preds2labels_mv()
    assert np.all(pc.labels == np.array([0, 5, 7]).reshape((3, 1)))


if __name__ == '__main__':
    test_pred2labels()
