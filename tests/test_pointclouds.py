# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
from morphx.classes.pointcloud import PointCloud


def test_generate_pred_labels():
    pc = PointCloud(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]]), labels=np.array([1, 2, 3]),
                    predictions={0: [2, 0, 0, 0, 1], 1: [5, 4, 5, 5, 4, 3], 2: [7, 7, 7, 6, 6, 6, 6]})
    expected = np.array([0, 5, 6]).reshape(-1, 1)
    assert np.all(pc.pred_labels == expected)

    expected = np.array([2, 5, 7]).reshape(-1, 1)
    assert np.all(pc.generate_pred_labels(False) == expected)


def test_get_coverage():
    pc = PointCloud(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]), labels=np.array([1, 2, 3, 4]),
                    predictions={0: [2, 0, 0, 0, 1], 2: [7, 7, 7, 6, 6, 6, 6]})
    assert pc.get_coverage() == 0.5

    pc = PointCloud(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]), labels=np.array([1, 2, 3, 4]),
                    predictions={0: [2, 0, 0, 0, 1], 1: [1], 2: [7, 7, 7, 6, 6, 6, 6], 3: [1]})
    assert pc.get_coverage() == 1

    pc = PointCloud(np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]), labels=np.array([1, 2, 3, 4]))
    assert pc.get_coverage() == 0


def test_prediction_smoothing():
    pc = PointCloud(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2],
                              [3, 3, 3], [3, 3, 3], [3, 3, 3]]), labels=np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]),
                    predictions={0: [2, 0, 0, 1], 1: [1], 2: [1], 4: [7, 7, 6], 5: [7], 6: [7, 7], 8: [8]})

    expected = np.array([1, 1, 1, -1, 7, 7, 7, -1, 7, -1, -1]).reshape(-1, 1)
    assert np.all(pc.prediction_smoothing(3) == expected)

def test_prediction_expansion():
    pc = PointCloud(np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1], [1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2],
                              [3, 3, 3], [3, 3, 3], [3, 3, 3]]), labels=np.array([1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]),
                    predictions={0: [2, 0, 0, 1], 1: [1], 2: [1], 4: [7, 7, 6], 5: [7], 6: [7, 7], 8: [8]})

    expected = np.array([1, 1, 1, 1, 7, 7, 7, 7, 7, 7, 7]).reshape(-1, 1)
    assert np.all(pc.prediction_expansion(3) == expected)


if __name__ == '__main__':
    test_generate_pred_labels()
    test_get_coverage()
    test_prediction_smoothing()
    test_prediction_expansion()
