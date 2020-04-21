# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import pickle
import numpy as np
from tqdm import tqdm
from typing import Dict, Union
import logging
from scipy.spatial import cKDTree
from morphx.data.basics import load_pkl
from typing import List, Optional, Tuple
from scipy.spatial.transform import Rotation as Rot


class PointCloud(object):
    """
    Class which represents a collection of points in 3D space. The points could have labels.
    """

    def __init__(self,
                 vertices: np.ndarray = None,
                 labels: np.ndarray = None,
                 pred_labels: np.ndarray = None,
                 features: np.ndarray = None,
                 types: np.ndarray = None,
                 encoding: dict = None,
                 obj_bounds: Dict[Union[str, int], np.ndarray] = None,
                 predictions: dict = None,
                 no_pred: List[str] = None):
        """
        Args:
            vertices: Point coordinates with shape (n, 3).
            labels: Vertex label array with shape (n, 1).
            pred_labels: Predicted labels (n, 1).
            features: Feature array with shape (n, m).
            types: Type array with shape (n, 1). Can be used for indicating differences between points by e.g. assigning
                different values to the different types.
            encoding: Dict with description strings for respective label as keys and unique labels as values.
            obj_bounds: Dict with object names as keys and start and end index of vertices which belong to this object
                in numpy arrays as values. E.g. {'obj1': [0, 10], 'obj2': [10, 20]}. The vertices from index 0 to 9
                then belong to obj1, the vertices from index 10 to 19 belong to obj2.
            predictions: Dict with vertex indices as keys and prediction lists as values. E.g. if vertex with index 1
                got the labels 2, 3, 4 as predictions, it would be {1: [2, 3, 4]}.
            no_pred: List of names of objects which should not be processed in model prediction or mapping.
        """
        if vertices is None:
            vertices = np.zeros((0, 3))
        if vertices.shape[1] != 3:
            raise ValueError("Vertices must have shape (N, 3).")
        self._vertices = vertices

        if labels is None:
            labels = np.zeros((0, 1))
        if len(labels) != 0 and len(labels) != len(vertices):
            raise ValueError("Vertex label array must have same length as vertices array.")
        self._labels = labels.reshape(len(labels), 1).astype(int)

        if pred_labels is None:
            pred_labels = np.zeros((0, 1))
        if len(pred_labels) != 0 and len(pred_labels) != len(vertices):
            raise ValueError("Predicted vertex label array must have same length as vertices array.")
        self._pred_labels = pred_labels.reshape(len(pred_labels), 1).astype(int)

        if features is None:
            features = np.zeros((0, 1))
        if len(features) != 0 and len(features) != len(vertices):
            raise ValueError("Feature array must have same length as vertices array.")
        self._features = features

        if types is None:
            types = np.zeros((0, 1))
        if len(types) != 0 and len(types) != len(vertices):
            raise ValueError("Type array must have same length as vertices array.")
        self._types = types.reshape(len(types), 1).astype(int)

        self._encoding = encoding
        self._obj_bounds = obj_bounds
        self._predictions = predictions

        if no_pred is None:
            self._no_pred = []
        else:
            self._no_pred = no_pred

        self._class_num = len(np.unique(labels))
        self._pred_num = None

    @property
    def vertices(self) -> np.ndarray:
        return self._vertices

    @property
    def labels(self) -> np.ndarray:
        return self._labels

    @property
    def pred_labels(self) -> np.ndarray:
        if len(self._pred_labels) == 0:
            return self.generate_pred_labels()
        return self._pred_labels

    @property
    def features(self) -> np.ndarray:
        return self._features

    @property
    def types(self) -> np.ndarray:
        return self._types

    @property
    def encoding(self) -> dict:
        return self._encoding

    @property
    def obj_bounds(self) -> dict:
        return self._obj_bounds

    @property
    def predictions(self) -> dict:
        if self._predictions is None:
            self._predictions = {}
        return self._predictions

    @property
    def no_pred(self) -> List[str]:
        return self._no_pred

    @property
    def class_num(self) -> int:
        return self._class_num

    @property
    def pred_num(self) -> int:
        if self._pred_num is None:
            self._pred_num = self.get_pred_num()
        return self._pred_num

    def __eq__(self, other: 'PointCloud'):
        if type(self) != type(other):
            return False
        attr_o = other.get_attr_dict()
        attr = self.get_attr_dict()
        if set(attr_o.keys()) != set(attr.keys()):
            return False
        for k, v in attr.items():
            if not np.all(v == attr_o[k]):
                return False
        return True

    # -------------------------------------- LABEL METHODS ------------------------------------------- #

    def map_labels(self, mappings: List[Tuple[int, int]]):
        """ In-place method for changing labels. Encoding gets updated.

        Args:
            mappings: List of tuples with original labels and target labels. E.g. [(1, 2), (3, 2)] means that
              the labels 1 and 3 will get replaced by 2.
        """
        for mapping in mappings:
            self._labels[self._labels == mapping[0]] = mapping[1]
            if self._encoding is not None and mapping[0] in self._encoding:
                self._encoding.pop(mapping[0], None)

    # --------------------------------------------- SETTER ------------------------------------------------ #

    def set_labels(self, labels: np.ndarray):
        if labels is None or len(labels) != len(self.vertices):
            raise ValueError("Label array must have same length as vertex array.")
        else:
            self._labels = labels

    def set_pred_labels(self, pred_labels: np.ndarray):
        if pred_labels is None or len(pred_labels) != len(self.vertices):
            raise ValueError("Pred label array must have same length as vertex array.")
        else:
            self._pred_labels = pred_labels

    def set_features(self, features: np.ndarray):
        if features is None or len(features) != len(self.vertices):
            raise ValueError("Feature array must have same length as vertex array.")
        else:
            self._features = features

    def set_types(self, types: np.ndarray):
        if types is None or len(types) != len(self.vertices):
            raise ValueError("Type array must have same length as vertex array.")
        else:
            self._types = types

    def set_encoding(self, encoding: dict):
        self._encoding = encoding

    def add_no_pred(self, obj_names: List[str]):
        for name in obj_names:
            if name not in self._no_pred:
                self._no_pred.append(name)

    def remove_obj_bounds(self):
        self._obj_bounds = None

    def set_predictions(self, predictions: dict):
        self._predictions = predictions

    # --------------------------------------- FEATURE HANDLING --------------------------------------------- #

    def types2feat(self, types2feat_map: Dict[int, np.ndarray]):
        """ Given a dict with feature arrays keyed by type ints, this method creates the feature array of the
            PointCloud. E.g. {1: [1, 0], 2: [0, 1]} would result in a one-hot encoding for types 1 and 2. This
            will override the feature array.

        Args:
            types2feat_map: keys must be equal to np.unique(type), so all and only all types must be included.
                All features associated with the types must have the same length
        """
        if len(self._types) == 0:
            return
        types = np.unique(self._types)
        if len(types) > len(types2feat_map):
            raise ValueError("Not enough types given for feature creation.")
        self._features = np.zeros((0, 1))
        for key in types2feat_map:
            if len(self._features) == 0:
                self._features = np.zeros((len(self._vertices), len(types2feat_map[key])))
            mask = (self._types == key).reshape(-1)
            if len(mask) == 0:
                continue
            self._features[mask] = types2feat_map[key]

    # -------------------------------------- PREDICTION HANDLING ------------------------------------------- #

    def get_pred_num(self) -> int:
        """ Returns number of predictions which are available for this PointCloud. """
        total = 0
        if self._predictions is None:
            return 0
        for key in self._predictions:
            total += len(self._predictions[key])
        return total

    def generate_pred_labels(self, mv: bool = True) -> np.ndarray:
        """ Flag mv = True: Each vertex gets the result of a majority vote on the predictions as label.
            If there are no predictions, the label is set to -1.
            Flag mc = False: For each vertex, the first predictions is taken as the new label. If there are no
            predictions, the label is set to -1.

        Returns:
            The newly generated labels. If there exist no predictions, the pred_label array will only contain -1.
        """
        if len(self._pred_labels) == 0:
            self._pred_labels = np.zeros((len(self._vertices), 1))
        self._pred_labels[:] = -1
        if self._predictions is None or len(self._predictions) == 0:
            return self._pred_labels
        for key in self._predictions.keys():
            preds = np.array(self._predictions[key])
            if mv:
                u_preds, counts = np.unique(preds, return_counts=True)
                self._pred_labels[key] = u_preds[np.argmax(counts)]
            else:
                self._pred_labels[key] = preds[0]
        if self._encoding is not None and -1 in self._pred_labels:
            self._encoding['no_pred'] = -1
        return self._pred_labels.astype(int)

    def get_coverage(self) -> Tuple:
        """ Get fraction of vertices with predictions.

        Returns:
            (Number of unpredicted labels, Total number of labels)
        """
        labels, counts = np.unique(self.pred_labels, return_counts=True)
        idx = np.argwhere(labels == -1)
        if len(idx) == 0:
            return 0, len(self._pred_labels)
        else:
            return counts[int(idx)], len(self._pred_labels)

    def prediction_smoothing(self, k: int = 20) -> Optional[np.ndarray]:
        """ Each vertex with existing prediction gets majority vote on labels from k nearest vertices
            with predicitions as label. """
        print("Prediction smoothing...")
        preds = self.pred_labels != -1
        preds = preds.reshape(-1)
        idcs = np.arange(len(self._vertices))
        preds = idcs[preds]
        if k > len(preds):
            k = len(preds)
        tree = cKDTree(self._vertices[preds])
        # copy labels as smoothing results should not influence smooting itself
        new_pred_labels = self.pred_labels.copy()
        for pred in tqdm(preds):
            dist, ind = tree.query(self._vertices[pred], k=k)
            neighbors = preds[ind]
            u_preds, counts = np.unique(self._pred_labels[neighbors], return_counts=True)
            new_pred_labels[pred] = u_preds[np.argmax(counts)]
        self._pred_labels = new_pred_labels
        return self._pred_labels.astype(int)

    def prediction_expansion(self, k: int = 20) -> np.ndarray:
        """ Each vertex gets majority vote on labels from k nearest vertices with predicitions as label.
            Should be called after preds2labels. If there are only vertices with predictions this method
            is the same as prediction_smoothing. """
        preds = np.fromiter(self._predictions.keys(), dtype=int)
        if k > len(preds):
            k = len(preds)
        tree = cKDTree(self._vertices[preds])
        # copy labels as expansion results should not influence expansion itself
        new_pred_labels = self.pred_labels.copy()
        verts_idcs = np.arange(len(self._vertices))
        for vert in verts_idcs:
            dist, ind = tree.query(self._vertices[vert], k=k)
            neighbors = preds[ind]
            u_preds, counts = np.unique(self._pred_labels[neighbors], return_counts=True)
            new_pred_labels[vert] = u_preds[np.argmax(counts)]
        self._pred_labels = new_pred_labels
        return self._pred_labels.astype(int)

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

    def rotate_randomly(self, angle_range: tuple = (-180, 180), random_flip: bool = False):
        """ Randomly rotates vertices by performing an Euler rotation. The three angles are chosen randomly
            from the given angle_range. If `random_flip` is True, flips axes independently around the origin. """
        # switch limits if lower limit is larger
        if angle_range[0] > angle_range[1]:
            angle_range = (angle_range[1], angle_range[0])

        angles = np.random.uniform(angle_range[0], angle_range[1], (1, 3))[0]
        r = Rot.from_euler('xyz', angles, degrees=True)
        if len(self._vertices) > 0:
            self._vertices = r.apply(self._vertices)
        if random_flip:
            flip_axes = (-1)**np.random.randint(0, 2, self._vertices.shape[1])
            self._vertices *= flip_axes

    def move(self, vector: np.ndarray):
        """ Moves vertices by adding the given vector """
        self._vertices = self._vertices + vector

    def add_noise(self, limits: tuple = (-1, 1), distr: str = 'uniform'):
        """
        Apply additive noise (drawn from `distr` and scaled by `distr_scale`) to vertices.

        Args:
            limits: Range of the noise values. Tuple is used as lower and upper bounds for ``distr='uniform'``
                or only the entry at index 1 is used as standard deviation if ``distr='normal'``. Note that the
                s.d. used to generate the vertex noise (i.i.d) is fixed by drawing a global value from the given normal
                distribution. This will lead to different noise levels within the given s.d. range (limits[1]).
            distr: Noise distribution, currently available: 'uniform' and 'Gaussian'.

        Returns:

        """
        if distr.lower() == 'normal':
            if abs(limits[0]) != abs(limits[1]):
                logging.warning(f'Lower ({limits[0]}) and upper ({limits[1]}) limits differ but chosen '
                                f'noise source was set to "normal". Using upper limit to scale standard '
                                f'normal values.')
            fixed_sd = np.random.standard_normal(1) * limits[1]
            variation = np.random.standard_normal(self._vertices.shape) * fixed_sd
        elif distr.lower() == 'uniform':
            # switch limits if lower limit is larger
            if limits[0] > limits[1]:
                limits = (limits[1], limits[0])
            # do nothing if limits are the same
            if limits[0] == limits[1]:
                return
            variation = np.random.random(self._vertices.shape) * (limits[1] - limits[0]) + limits[0]
        else:
            raise ValueError(f'Given value "{distr}" for noise distribution not available.')
        self._vertices = self._vertices + variation

    def mult_noise(self, distr_scale: float = 0.05, distr: str = 'uniform'):
        """
        Vertices will be multiplied with the factor (1+X), where X is drawn from the
        given distribution `distr` scaled by `distr_scale`.

        Args:
            distr_scale: Scale factor applied to the noise distribution values (i.e. s.d. for ``distr='normal'``) or
                lower and upper bound (``distr='uniform'``).
            distr: Noise distribution, currently available: 'uniform' and 'Gaussian'.
        """
        if distr.lower() == 'normal':
            variation = 1 + np.random.standard_normal(1)[0] * distr_scale
        elif distr.lower() == 'uniform':
            variation = 1 + np.random.random(1) * 2 * distr_scale - distr_scale
        else:
            raise ValueError(f'Given value "{distr}" for noise distribution not available.')
        self.scale(variation)

    def shear(self, limits: tuple = (-1, 1)):
        """
        Shears the vertices by applying a transformation matrix [[1, s_xy, s_xz], [s_yx, 1, s_yz], [s_zx, s_zy, 1]],
        where the factors s_ij are drawn from a uniform distribution.

        Args:
            limits: Range of the interval for the factors s_ij. Tuple defines lower and upper bound of the uniform
                distribution.
        """
        transform = np.random.random((3, 3)) * (limits[1] - limits[0]) + limits[0]
        np.fill_diagonal(transform, 1)
        self._vertices = self._vertices.dot(transform)

# -------------------------------------- CLOUD I/O ------------------------------------------- #

    def save2pkl(self, path: str) -> int:
        """ Saves point cloud into pickle file at given path.

        Args:
            path: Pickle file in which cloud should be saved.

        Returns:
            0 if saving process was successful, 1 otherwise.
        """
        try:
            attr_dict = self.get_attr_dict()
            with open(path, 'wb') as f:
                pickle.dump(attr_dict, f)
            f.close()
        except FileNotFoundError:
            print("Saving was not successful as given path is not valid.")
            return 1
        return 0

    def load_from_pkl(self, path: str):
        """
        Load attribute dict from pickle file.

        Args:
            path: Path to pickle file which contains the attribute dictionary.
        """
        self.__init__(**load_pkl(path))
        return self

    def get_attr_dict(self):
        attr_dict = {'vertices': self._vertices,
                     'labels': self._labels,
                     'pred_labels': self._pred_labels,
                     'features': self._features,
                     'types': self._types,
                     'encoding': self._encoding,
                     'obj_bounds': self._obj_bounds,
                     'predictions': self._predictions,
                     'no_pred': self._no_pred}
        return attr_dict
