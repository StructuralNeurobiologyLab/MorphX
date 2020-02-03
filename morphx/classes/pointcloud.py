# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import numpy as np
from typing import Dict, Union, Optional
from scipy.spatial.transform import Rotation as Rot


class PointCloud(object):
    """
    Class which represents a collection of points in 3D space. The points could have labels.
    """

    def __init__(self, vertices: np.ndarray, labels: np.ndarray = None, encoding: dict = None,
                 obj_bounds: Optional[Dict[Union[str, int], np.ndarray]] = None, predictions: dict = None):
        """
        Args:
            vertices: Point coordinates with shape (n, 3).
            labels: Vertex label array with shape (n, 1).
            encoding: Dict with description strings for respective label as keys and unique labels as values.
            obj_bounds: Dict with object names as keys and start and end index of vertices which belong to this object.
                E.g. {'obj1': [0, 10], 'obj2': [10, 20]}. The vertices from index 0 to 9 then belong to obj1, the
                vertices from index 10 to 19 belong to obj2.
            predictions: Dict with vertex indices as keys and prediction lists as values. E.g. if vertex with index 1
                got the labels 2, 3, 4 as predictions, it would be {1: [2, 3, 4]}.
        """
        if len(vertices) > 0 and vertices.shape[1] != 3:
            raise ValueError("Vertices must have shape (N, 3).")
        self._vertices = vertices

        if labels is None:
            self._labels = np.empty(0)
        if labels is not None:
            if len(labels) != len(vertices):
                raise ValueError("Vertex label array must have same length as vertices array.")
            self._labels = labels.reshape(len(labels), 1).astype(int)

        self._encoding = encoding
        self._obj_bounds = obj_bounds
        self._predictions = predictions
        self._class_num = len(np.unique(labels))
        self._scale_count = 0

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def encoding(self) -> dict:
        return self._encoding

    @property
    def obj_bounds(self) -> dict:
        return self._obj_bounds

    @property
    def predictions(self) -> dict:
        if self._predictions is None:
            self._predictions = {ix: [] for ix in range(len(self._vertices))}
        return self._predictions

    @property
    def class_num(self) -> int:
        return self._class_num

    # -------------------------------------- LABEL ANALYSIS ------------------------------------------- #

    @property
    def weights_mean(self) -> np.ndarray:
        """ Extract frequences for each class and calculate weights as frequences.mean() / frequences, ignoring any
        labels which don't appear in the dataset (setting their weight to 0).

        Returns:
            np.ndarray with weights for each class.
        """
        if len(self._labels) != 0:
            total_labels = self._labels
            non_zero = []
            freq = []
            for i in range(self._class_num):
                freq.append((total_labels == i).sum())
                if freq[i] != 0:
                    # save for mean calculation
                    non_zero.append(freq[i])
                else:
                    # prevent division by zero
                    freq[i] = 1
            mean = np.array(non_zero).mean()
            freq = mean / np.array(freq)
            freq[(freq == mean)] = 0
            return freq
        else:
            return np.array([])

    @property
    def weights_occurence(self) -> np.ndarray:
        """ Extract frequences for each class and calculate weights as len(vertices) / frequences.

        Returns:
            np.ndarray with weights for each class.
        """
        if len(self._labels) != 0:
            class_num = self._class_num
            total_labels = self._labels
            freq = []
            for i in range(class_num):
                freq.append((total_labels == i).sum())
            freq = len(total_labels) / np.array(freq)
            return freq
        else:
            return np.array([])

    # -------------------------------------- PREDICTION HANDLING ------------------------------------------- #

    def preds2labels_mv(self) -> np.ndarray:
        """ For each vertex, a majority vote is applied to the existing predictions and the prediction with the highest
            occurance is set as label for this vertex. If there are no predictions, the label is set to -1.

        Returns:
            The newly generated labels.
        """
        for idx in range(len(self._labels)):
            preds = np.array(self._predictions[idx])
            if len(preds) > 0:
                u_preds, counts = np.unique(preds, return_counts=True)
                self._labels[idx] = u_preds[np.argmax(counts)]
            else:
                self._labels[idx] = -1
        if self._encoding is not None and -1 in self._labels:
            self._encoding['no_prediction'] = -1
        return self._labels

    def preds2labels_direct(self) -> np.ndarray:
        """ For each vertex, the first predictions is taken as the new label. If there are no predictions, the label is
            set to -1.

        Returns:
            The newly generated labels.
        """
        for idx in range(len(self._labels)):
            preds = np.array(self._predictions[idx])
            if len(preds) > 0:
                self._labels[idx] = preds[0]
            else:
                self._labels[idx] = -1
        if self._encoding is not None and -1 in self._labels:
            self._encoding['no_prediction'] = -1
        return self._labels

    # -------------------------------------- TRANSFORMATIONS ------------------------------------------- #

    def scale(self, factor: int):
        """ If factor < 0 vertices are divided by the factor. If factor > 0 vertices are multiplied by the
            factor. If factor == 0 nothing happens. """
        if factor == 0:
            return
        elif factor < 0:
            self._vertices = self._vertices / -factor
        else:
            self._vertices = self._vertices * factor
        self._scale_count += 1
        if self._scale_count > 1:
            raise ValueError('Multiple scaling should not happen')

    def rotate_randomly(self, angle_range: tuple = (-180, 180)):
        """ Randomly rotates vertices by performing an Euler rotation. The three angles are choosen randomly
            from the given angle_range. """
        # switch limits if lower limit is larger
        if angle_range[0] > angle_range[1]:
            angle_range = (angle_range[1], angle_range[0])

        angles = np.random.uniform(angle_range[0], angle_range[1], (1, 3))[0]
        r = Rot.from_euler('xyz', angles, degrees=True)
        if len(self._vertices) > 0:
            self._vertices = r.apply(self._vertices)

    def move(self, vector: np.ndarray):
        """ Moves vertices by adding the given vector """
        self._vertices = self._vertices + vector

    def add_noise(self, limits: tuple = (-1, 1)):
        """ Adds some random variation (amplitude given by the limits parameter) to vertices. """
        # switch limits if lower limit is larger
        if limits[0] > limits[1]:
            limits = (limits[1], limits[0])
        # do nothing if limits are the same
        if limits[0] == limits[1]:
            return

        variation = np.random.random(self._vertices.shape) * (limits[1] - limits[0]) + limits[0]
        self._vertices = self._vertices + variation
