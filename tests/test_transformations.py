# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import random
import numpy as np
from morphx.processing import clouds
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud
from scipy.spatial.transform import Rotation as Rot


def test_normalization():
    pc = PointCloud(np.array([[10, 10, 10], [20, 20, 20]]))
    hc = HybridCloud(np.array([[10, 10, 10], [20, 20, 20]]), np.array([[0, 1]]),
                     vertices=np.array([[10, 10, 10], [20, 20, 20]]))

    pc.scale(-10)
    hc.scale(-10)

    assert np.all(pc.vertices == np.array([[1, 1, 1], [2, 2, 2]]))
    assert np.all(hc.vertices == np.array([[1, 1, 1], [2, 2, 2]]))
    assert np.all(hc.nodes == np.array([[1, 1, 1], [2, 2, 2]]))

    # test transformation class from processing.clouds
    pc = PointCloud(np.array([[10, 10, 10], [20, 20, 20]]))
    hc = HybridCloud(np.array([[10, 10, 10], [20, 20, 20]]), np.array([[0, 1]]),
                     vertices=np.array([[10, 10, 10], [20, 20, 20]]))

    transform = clouds.Normalization(10)
    transform(pc)
    transform(hc)

    assert np.all(pc.vertices == np.array([[1, 1, 1], [2, 2, 2]]))
    assert np.all(hc.vertices == np.array([[1, 1, 1], [2, 2, 2]]))
    assert np.all(hc.nodes == np.array([[1, 1, 1], [2, 2, 2]]))


def test_rotate_randomly():
    angle_range = (60, 60)

    angles = np.random.uniform(angle_range[0], angle_range[1], (1, 3))[0]
    rot = Rot.from_euler('xyz', angles, degrees=True)

    pc = PointCloud(np.array([[10, 10, 10], [20, 20, 20]]))
    hc = HybridCloud(np.array([[10, 10, 10], [20, 20, 20]]), np.array([[0, 1]]),
                     vertices=np.array([[10, 10, 10], [20, 20, 20]]))

    pc.rotate_randomly(angle_range)
    hc.rotate_randomly(angle_range)

    assert np.all(pc.vertices == rot.apply(np.array([[10, 10, 10], [20, 20, 20]])))
    assert np.all(hc.vertices == rot.apply(np.array([[10, 10, 10], [20, 20, 20]])))
    assert np.all(hc.nodes == rot.apply(np.array([[10, 10, 10], [20, 20, 20]])))

    pc = PointCloud(np.array([[10, 10, 10], [20, 20, 20]]))
    hc = HybridCloud(np.array([[10, 10, 10], [20, 20, 20]]), np.array([[0, 1]]),
                     vertices=np.array([[10, 10, 10], [20, 20, 20]]))

    # test transformation class from processing.clouds
    transform = clouds.RandomRotate(angle_range, apply_flip=False)
    transform(pc)
    transform(hc)

    assert np.all(pc.vertices == rot.apply(np.array([[10, 10, 10], [20, 20, 20]])))
    assert np.all(hc.vertices == rot.apply(np.array([[10, 10, 10], [20, 20, 20]])))
    assert np.all(hc.nodes == rot.apply(np.array([[10, 10, 10], [20, 20, 20]])))


def test_center():
    pc = PointCloud(np.array([[10, 10, 10], [20, 20, 20]]))
    hc = HybridCloud(np.array([[10, 10, 10], [20, 20, 20]]), np.array([[0, 1]]),
                     vertices=np.array([[10, 10, 10], [20, 20, 20]]))
    relation = hc.vertices[0] - hc.nodes[1]

    pc.move(np.array([1, 1, 1]))
    hc.move(np.array([1, 1, 1]))

    assert np.all(pc.vertices == np.array([[11, 11, 11], [21, 21, 21]]))
    assert np.all(hc.vertices == np.array([[11, 11, 11], [21, 21, 21]]))
    assert np.all(hc.nodes == np.array([[11, 11, 11], [21, 21, 21]]))
    assert np.all(hc.vertices[0] - hc.nodes[1] == relation)

    pc = PointCloud(np.array([[10, 10, 10], [20, 20, 20]]))
    hc = HybridCloud(np.array([[10, 10, 10], [20, 20, 20]]), np.array([[0, 1]]),
                     vertices=np.array([[10, 10, 10], [20, 20, 20]]))
    relation = hc.vertices[0] - hc.nodes[1]

    # test transformation class from processing.clouds
    transform = clouds.Center()
    transform(pc)
    transform(hc)

    assert np.all(np.mean(pc.vertices, axis=0) == np.array([0, 0, 0]))
    assert np.all(np.mean(hc.vertices, axis=0) == np.array([0, 0, 0]))
    assert np.all(hc.vertices[0] - hc.nodes[1] == relation)


def test_composition():
    pc = PointCloud(np.array([[10, 10, 10], [20, 20, 20]]))
    hc = HybridCloud(np.array([[10, 10, 10], [20, 20, 20]]), np.array([[0, 1]]),
                     vertices=np.array([[10, 10, 10], [20, 20, 20]]))

    transform = clouds.Compose([clouds.Normalization(10), clouds.RandomRotate((60, 60)), clouds.Center()])
    transform(pc)
    transform(hc)

    assert np.all(np.round(np.mean(pc.vertices, axis=0)) == np.array([0, 0, 0]))
    assert np.all(np.round(np.mean(hc.vertices, axis=0)) == np.array([0, 0, 0]))

    dummy = np.array([[10, 10, 10], [20, 20, 20]]) / 10
    angle_range = (60, 60)
    angles = np.random.uniform(angle_range[0], angle_range[1], (1, 3))[0]
    rot = Rot.from_euler('xyz', angles, degrees=True)
    dummy = rot.apply(dummy)
    centroid = np.mean(dummy, axis=0)
    dummy = dummy - centroid

    assert np.all(pc.vertices == dummy)
    assert np.all(hc.vertices == dummy)
    assert np.all(hc.vertices == dummy)


def test_random_variation():
    np.random.seed(0)
    pc = PointCloud(np.array([[10, 10, 10], [20, 20, 20]]))
    transform = clouds.RandomVariation((-10000, 10000))
    transform(pc)

    expected = np.array([[986., 4314., 2065.], [918., -1507., 2938.]])
    assert np.all(np.round(pc.vertices) == expected)


if __name__ == '__main__':
    test_normalization()
    test_rotate_randomly()
    test_center()
    test_composition()
    test_random_variation()
