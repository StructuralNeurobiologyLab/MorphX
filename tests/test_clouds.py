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


@pytest.mark.skip(reason="WIP")
def test_cloud_merging():
    pc1 = PointCloud(np.array([[i, i, i] for i in range(10)]), np.array([[i] for i in range(10)]),
                     obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])}, encoding={'e1': 1, 'e2': 2})
    pc2 = PointCloud(np.array([[i, i, i] for i in range(10)]), np.array([[i] for i in range(10)]))
    hc = HybridCloud(np.array([[1, 1, 1], [2, 2, 2]]), np.array([[0, 1]]),
                     vertices=np.array([[i, i, i] for i in range(10)]),
                     labels=np.array([[i] for i in range(10)]), encoding={'e1': 1, 'e3': 3})
    result = clouds.merge_clouds([pc1, pc2, hc], ['pc1', 'pc2', 'hc'])

    vertices = np.concatenate((pc1.vertices, pc2.vertices, hc.vertices), axis=0)
    labels = np.concatenate((pc1.labels, pc2.labels, hc.labels), axis=0)
    obj_bounds = {'obj1': np.array([0, 5]), 'obj2': np.array([5, 10]), 'pc2': np.array([10, 20]),
                  'hc': np.array([20, 30])}
    encoding = {'e1': 1, 'e2': 2, 'e3': 3}

    assert np.all(result.vertices == vertices)
    assert np.all(result.labels == labels)
    assert len(result.obj_bounds) == len(obj_bounds)
    for key in result.obj_bounds:
        assert np.all(result.obj_bounds[key] == obj_bounds[key])
    assert len(result.encoding) == len(encoding)
    for key in result.encoding:
        assert result.encoding[key] == encoding[key]


if __name__ == '__main__':
    start = time.time()
    test_cloud_merging()
    print('Finished after', time.time() - start)