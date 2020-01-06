# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import time
import numpy as np
from morphx.classes.pointcloud import PointCloud
from morphx.processing import clouds


def test_object_filtering():
    pc = PointCloud(np.array([[i, i, i] for i in range(10)]), np.array([[i] for i in range(10)]),
                    obj_bounds={'obj1': [0, 5], 'obj2': [5, 10]})
    new_pc = clouds.filter_objects(pc, ['obj1'])

    assert np.all(new_pc.vertices == np.array([[i, i, i] for i in range(5)]))
    assert np.all(new_pc.labels == np.array([[i] for i in range(5)]))
    assert len(new_pc.obj_bounds.keys()) == 1
    assert new_pc.obj_bounds['obj1'] == [0, 5]


if __name__ == '__main__':
    start = time.time()
    test_object_filtering()
    print('Finished after', time.time() - start)