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
    assert np.all(new_pc.obj_bounds['obj1'] == np.array([0, 5]))


def test_merging():
    pc1 = PointCloud(np.array([[i, i, i] for i in range(10)]), labels=np.array([[i] for i in range(10)]))
    pc2 = PointCloud(np.array([[i, i, i] for i in range(10)]), features=np.array([[i] for i in range(10)]))
    merged = clouds.merge([pc1, pc2])
    assert len(merged.labels) == 0
    assert len(merged.features) == 0

    pc1 = PointCloud(np.array([[i, i, i] for i in range(10)]), labels=np.array([[i] for i in range(10)]))
    pc2 = PointCloud()
    merged = clouds.merge([pc1, pc2])
    assert len(merged.vertices) == len(pc1.vertices)

    pc1 = PointCloud(np.array([[i, i, i] for i in range(10)]),
                     obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])})
    pc2 = PointCloud(np.array([[i, i, i] for i in range(10)]),
                     obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])})
    merged = clouds.merge([pc1, pc2], ['pc1', 'pc2'], True)
    assert np.all(merged.obj_bounds['pc1_obj1'] == np.array([0, 5]))
    assert np.all(merged.obj_bounds['pc1_obj2'] == np.array([5, 10]))
    assert np.all(merged.obj_bounds['pc2_obj1'] == np.array([10, 15]))
    assert np.all(merged.obj_bounds['pc2_obj2'] == np.array([15, 20]))

    pc1 = PointCloud(np.array([[i, i, i] for i in range(10)]), labels=np.array([[i] for i in range(10)]),
                     features=np.array([[i] for i in range(10)]),
                     encoding={1: 'a', 2: 'b'},
                     obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])})
    pc2 = PointCloud(np.array([[i, i, i] for i in range(10)]), labels=np.array([[i] for i in range(10)]),
                     encoding={3: 'c', 1: 'a'},
                     no_pred=['obj1'],
                     obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])})
    merged = clouds.merge([pc1, pc2], ['pc1', 'pc2'], True)
    assert merged.encoding == {1: 'a', 2: 'b', 3: 'c'}
    assert len(merged.labels) == 20
    assert len(merged.features) == 0
    assert merged.no_pred == ['pc2_obj1']

    pc1 = PointCloud(np.array([[i, i, i] for i in range(10)]), labels=np.array([[i] for i in range(10)]),
                     features=np.array([[i] for i in range(10)]),
                     encoding={1: 'a', 2: 'b'},
                     no_pred=['obj2'],
                     predictions={0: [2, 3], 4: [1, 1]},
                     obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])})
    pc2 = PointCloud(np.array([[i, i, i] for i in range(10)]), labels=np.array([[i] for i in range(10)]),
                     encoding={3: 'c', 1: 'b'},
                     no_pred=['obj1'],
                     predictions={0: [2, 3], 4: [1, 1]},
                     obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])})
    merged = clouds.merge([pc1, pc2], ['pc1', 'pc2'], True)
    assert merged.predictions == {0: [2, 3], 4: [1, 1], 10: [2, 3], 14: [1, 1]}
    assert merged.encoding is None
    assert merged.no_pred == ['pc1_obj2', 'pc2_obj1']


def test_hybrid_merging():
    hc = HybridCloud(vertices=np.array([[i, i, i] for i in range(10)]), labels=np.array([[i] for i in range(10)]),
                     nodes=np.array([[i, i, i] for i in range(2)]),
                     edges=np.array([[0, 1], [1, 2]]),
                     features=np.array([[i] for i in range(10)]),
                     encoding={1: 'a', 2: 'b'},
                     no_pred=['obj2'],
                     predictions={0: [2, 3], 4: [1, 1]},
                     obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])})
    pc2 = PointCloud(np.array([[i, i, i] for i in range(10)]), labels=np.array([[i] for i in range(10)]),
                     encoding={3: 'c', 1: 'b'},
                     no_pred=['obj1'],
                     predictions={0: [2, 3], 4: [1, 1]},
                     obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])})
    merged = clouds.merge_hybrid(hc, [pc2], hc_name='hc', names=['pc2'], preserve_obj_bounds=True)
    assert len(merged.nodes) == len(hc.nodes)
    assert merged.no_pred == ['pc2_obj1', 'hc_obj2']


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
