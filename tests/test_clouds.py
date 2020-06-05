# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import time
import pytest
import numpy as np
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud
from morphx.processing import clouds


def test_object_filtering():
    pc = PointCloud(np.array([[i, i, i] for i in range(10)]), np.array([[i] for i in range(10)]),
                    obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])})
    new_pc = clouds.filter_objects(pc, ['obj1'])

    assert np.all(new_pc.vertices == np.array([[i, i, i] for i in range(5)]))
    assert np.all(new_pc.labels == np.array([[i] for i in range(5)]))
    assert len(new_pc.obj_bounds.keys()) == 1
    assert new_pc.obj_bounds['obj1'] == [0, 5]


def test_sampling():
    pc = PointCloud(vertices=np.array([[i, i, i] for i in range(10)], dtype=float),
                    labels=np.array([[i] for i in range(10)]),
                    features=np.array([[i] for i in range(10)]),
                    types=np.array([[i] for i in range(10)]),
                    pred_labels=np.array([[i] for i in range(10)]))
    sample, idcs = clouds.sample_cloud(pc, 5)
    assert len(sample.vertices) == 5
    for i in range(len(sample.vertices)):
        mask = np.all(pc.vertices == sample.vertices[i], axis=1)
        assert np.all(pc.labels[mask] == sample.labels[i])
        assert np.all(pc.features[mask] == sample.features[i])
        assert np.all(pc.types[mask] == sample.types[i])
        assert np.all(pc.pred_labels[mask] == sample.pred_labels[i])

    pc = PointCloud(vertices=np.array([[i, i, i] for i in range(10)], dtype=float),
                    labels=np.array([[i] for i in range(10)]),
                    features=np.array([[i] for i in range(10)]),
                    pred_labels=np.array([[i] for i in range(10)]))
    sample, idcs = clouds.sample_cloud(pc, 5)
    assert len(sample.vertices) == 5
    for i in range(len(sample.vertices)):
        mask = np.all(pc.vertices == sample.vertices[i], axis=1)
        assert np.all(pc.labels[mask] == sample.labels[i])
        assert np.all(pc.features[mask] == sample.features[i])
        assert len(sample.types) == 0
        assert np.all(pc.pred_labels[mask] == sample.pred_labels[i])

    pc = PointCloud(vertices=np.array([[i, i, i] for i in range(10)], dtype=float),
                    labels=np.array([[i] for i in range(10)]),
                    features=np.array([[i] for i in range(10)]),
                    pred_labels=np.array([[i] for i in range(10)]))
    sample, idcs = clouds.sample_cloud(pc, 10)
    assert len(sample.vertices) == 10
    for i in range(len(sample.vertices)):
        mask = np.all(pc.vertices == sample.vertices[i], axis=1)
        assert np.all(pc.labels[mask] == sample.labels[i])
        assert np.all(pc.features[mask] == sample.features[i])
        assert len(sample.types) == 0
        assert np.all(pc.pred_labels[mask] == sample.pred_labels[i])

    sample, idcs = clouds.sample_cloud(pc, 13)
    assert len(sample.vertices) == 13
    for i in range(len(sample.vertices)):
        mask = np.all(pc.vertices == sample.vertices[i], axis=1)
        assert np.all(pc.labels[mask] == sample.labels[i])
        assert np.all(pc.features[mask] == sample.features[i])
        assert len(sample.types) == 0
        assert np.all(pc.pred_labels[mask] == sample.pred_labels[i])

    sample, idcs = clouds.sample_cloud(pc, 333)
    assert len(sample.vertices) == 333
    for i in range(len(sample.vertices)):
        mask = np.all(pc.vertices == sample.vertices[i], axis=1)
        assert np.all(pc.labels[mask] == sample.labels[i])
        assert np.all(pc.features[mask] == sample.features[i])
        assert len(sample.types) == 0
        assert np.all(pc.pred_labels[mask] == sample.pred_labels[i])

    sample, idcs = clouds.sample_cloud(pc, 40)
    assert len(sample.vertices) == 40
    for i in range(len(sample.vertices)):
        mask = np.all(pc.vertices == sample.vertices[i], axis=1)
        assert np.all(pc.labels[mask] == sample.labels[i])
        assert np.all(pc.features[mask] == sample.features[i])
        assert len(sample.types) == 0
        assert np.all(pc.pred_labels[mask] == sample.pred_labels[i])


if __name__ == '__main__':
    test_sampling()