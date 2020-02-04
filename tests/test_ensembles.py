# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import time
import ipdb
import numpy as np
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud
from morphx.classes.cloudensemble import CloudEnsemble
from morphx.processing import ensembles


def test_ensemble2pointcloud():
    pc1 = PointCloud(np.array([[i, i, i] for i in range(10)]), np.array([[i] for i in range(10)]),
                     obj_bounds={'obj1': np.array([0, 5]), 'obj2': np.array([5, 10])}, encoding={'e1': 1, 'e2': 2})
    pc2 = PointCloud(np.array([[i, i, i] for i in range(10)]), np.array([[i] for i in range(10)]))
    hc = HybridCloud(np.array([[1, 1, 1], [2, 2, 2]]), np.array([[0, 1]]), np.array([[i, i, i] for i in range(10)]),
                     encoding={'e1': 1, 'e3': 3},
                     obj_bounds={'obj3': np.array([0, 2]), 'obj4': np.array([2, 10])})
    ce = CloudEnsemble({'pc1': pc1, 'pc2': pc2, 'hc': hc})
    result = ensembles.ensemble2pointcloud(ce)

    obj_bounds = {'obj1': np.array([0, 5]), 'obj2': np.array([ 5, 10]),
                  'pc2': np.array([10, 20]), 'obj3': np.array([20, 22]),
                  'obj4': np.array([22, 30])}
    encoding = {'e1': 1, 'e2': 2, 'e3': 3}

    assert np.all(result.nodes == hc.nodes)
    assert np.all(result.edges == hc.edges)
    assert np.all(result.vertices == np.concatenate((pc1.vertices, pc2.vertices, hc.vertices), axis=0))
    assert np.all(result.labels == np.concatenate((pc1.labels, pc2.labels, np.ones((10, 1))*-1), axis=0))
    assert len(result.obj_bounds) == len(obj_bounds)
    for key in obj_bounds:
        assert np.all(obj_bounds[key] == result.obj_bounds[key])
    assert len(result.encoding) == len(encoding)
    for key in result.encoding:
        assert result.encoding[key] == encoding[key]


if __name__ == '__main__':
    start = time.time()
    test_ensemble2pointcloud()
    print('Finished after', time.time() - start)