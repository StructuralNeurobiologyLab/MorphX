# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import pickle
import numpy as np
from math import ceil
from typing import Union, Tuple, List, Optional
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud


# -------------------------------------- CLOUD SAMPLING ------------------------------------------- #


def sample_cloud(pc: PointCloud, vertex_number: int, random_seed=None) -> Tuple[PointCloud, np.ndarray]:
    """ Creates a (pseudo)random sample point cloud with a specific number of points from the given subset of mesh
    vertices. If the requested number of points is larger than the given subset, the subset gets enriched with slightly
    augmented points before sampling.

    Args:
        pc: MorphX PointCloud object which should be sampled.
        vertex_number: The number of points which should make up the sample point cloud.
        random_seed: Possibility for making the sampling deterministic.

    Returns:
        PointCloud with sampled points (and labels) and indices of the original vertices where samples are from.
    """
    cloud = pc.vertices
    labels = pc.labels

    if len(cloud) == 0:
        return pc, np.array([])

    if random_seed is not None:
        np.random.seed(random_seed)

    dim = cloud.shape[1]
    sample = np.zeros((vertex_number, dim))
    if len(labels) != 0:
        sample_l = np.zeros((vertex_number, 1))
    else:
        sample_l = None
    if len(pc.features) != 0:
        sample_f = np.zeros((vertex_number, pc.features.shape[1]))
    else:
        sample_f = None

    # How much additional augmented points must be generated in order to reach the requested number of samples
    deficit = vertex_number - len(cloud)

    vert_ixs = np.arange(len(cloud))
    np.random.shuffle(vert_ixs)
    sample[:min(len(cloud), vertex_number)] = cloud[vert_ixs[:vertex_number]]

    # Save vertex index of each sample point for later mapping
    sample_ixs = np.ones(vertex_number)
    sample_ixs[:min(len(cloud), vertex_number)] = vert_ixs[:vertex_number]

    if len(labels) != 0:
        sample_l[:min(len(cloud), vertex_number)] = labels[vert_ixs[:vertex_number]]
    if len(pc.features) != 0:
        sample_f[:min(len(cloud), vertex_number)] = pc.features[vert_ixs[:vertex_number]]

    if deficit > 0:
        # deficit could be bigger than cloudsize
        offset = len(cloud)
        for it in range(ceil(deficit/len(cloud))):
            # add augmented points to reach requested number of samples
            compensation = min(len(cloud), len(sample)-offset)
            np.random.shuffle(vert_ixs)
            sample[offset:offset+compensation] = cloud[vert_ixs[:compensation]]
            # Save vertex indices of additional points
            sample_ixs[offset:offset+compensation] = vert_ixs[:compensation]
            if len(labels) != 0:
                sample_l[offset:offset+compensation] = labels[vert_ixs[:compensation]]
            if len(pc.features) != 0:
                sample_f[offset:offset+compensation] = pc.features[vert_ixs[:compensation]]
            offset += compensation

        sample[len(cloud):] += np.random.random(sample[len(cloud)].shape)

    return PointCloud(sample, labels=sample_l, features=sample_f, encoding=pc.encoding, no_pred=pc.no_pred), sample_ixs


def sample_objectwise(pc: PointCloud, vertex_number: int, random_seed=None) -> Tuple[PointCloud, np.ndarray]:
    """ Creates a (pseudo)random sample point cloud with a specific number of points from the given subset of mesh
        vertices. If different objects are present within the PointCloud (indicated by the obj_bounds attribute),
        the number of sample points for each object is calculated by (number of object vertices)/(total number of
        vertices) * vertex_number. For each object, the method sample_cloud is used. If obj_bounds of the PointCloud
        is None, this method is identical to sample_cloud.

    Args:
        pc: PointCloud which should be sampled.
        vertex_number: The number of points which should be sampled.
        random_seed: Random seed for making the sampling deterministic.

    Returns:
        PointCloud with sampled points (and labels) and indices of the original vertices where samples are from.
    """
    if pc.obj_bounds is None:
        return sample_cloud(pc, vertex_number, random_seed)
    curr_num = 0
    samples = []
    names = []
    ixs = np.zeros(vertex_number)
    for key in pc.obj_bounds:
        bounds = pc.obj_bounds[key]
        if bounds[1]-bounds[0] != 0:
            sample_num = (bounds[1]-bounds[0])/len(pc.vertices)*vertex_number
            if curr_num + ceil(sample_num) <= vertex_number:
                sample_num = ceil(sample_num)
            else:
                sample_num = vertex_number - curr_num
            curr_cloud = PointCloud(pc.vertices[bounds[0]:bounds[1]], labels=pc.labels[bounds[0]:bounds[1]],
                                    features=pc.features[bounds[0]:bounds[1]])
            sample, sample_ixs = sample_cloud(curr_cloud, sample_num, random_seed)
            samples.append(sample)
            names.append(key)
            ixs[curr_num:curr_num+len(sample_ixs)] = sample_ixs
            curr_num += sample_num

    # use merge method for correct object boundary information
    result_sample = merge_clouds(samples, names)
    result_sample.add_no_pred(pc.no_pred)
    result_sample.set_encoding(pc.encoding)
    result_sample.remove_obj_bounds()
    return result_sample, ixs


# -------------------------------------- CLOUD FILTERING / LABEL MAPPING ------------------------------------------- #

def filter_preds(cloud: PointCloud) -> PointCloud:
    """ Returns a PointCloud with only those vertices and labels for which predictions exist. The predictions of
        these points get transfered to the returned PointCloud, all other attributes of the original cloud (encoding,
        obj_bounds, ...) are lost.

    Args:
        cloud: The PointCloud from which vertices with existing predictions should be filtered.

     Returns:
        PointCloud containing only vertices and labels with existing predictions.
    """
    idcs = []
    new_predictions = {}
    counter = 0
    for key in cloud.predictions:
        if len(cloud.predictions[key]) != 0:
            idcs.append(key)
            new_predictions[counter] = cloud.predictions[key]
            counter += 1
    return PointCloud(cloud.vertices[idcs], cloud.labels[idcs], predictions=new_predictions)


def filter_labels(cloud: PointCloud, labels: list) -> PointCloud:
    """ Returns a pointcloud which contains only those vertices which labels occuring in 'labels'. If 'cloud'
        is a HybridCloud, the skeleton is taken as it is and should later be filtered with the 'filter_traverser'
        method.

    Args:
        cloud: PointCloud which should be filtered.
        labels: List of labels for which the corresponding vertices should be extracted.

    Returns:
        PointCloud object which contains only vertices with the filtered labels. Skeletons in case of HybridClouds are
        the same.
    """
    mask = np.zeros(len(cloud.labels), dtype=bool)
    for label in labels:
        mask = np.logical_or(mask, cloud.labels == label)

    if isinstance(cloud, HybridCloud):
        f_cloud = HybridCloud(cloud.nodes, cloud.edges, vertices=cloud.vertices[mask], labels=cloud.labels[mask])
    else:
        f_cloud = PointCloud(cloud.vertices[mask], labels=cloud.labels[mask])
    return f_cloud


def filter_objects(cloud: PointCloud, objects: list) -> PointCloud:
    """ Creates a PointCloud which contains only the objects given in objects. There must exist an obj_bounds dict in
     order to use this method. The dict gets updated with the new object boundaries.

    Args:
        cloud: The initial Pointcloud from which objects should be filtered.
        objects: List of objects where each entry is also a key in the obj_bounds dict of the cloud.

    Returns:
        A PointCloud containing only the desired objects.
    """
    if cloud.obj_bounds is None:
        raise ValueError("Objects cannot be filtered because obj_bounds dict doesn't exist (is None).")
    size = 0
    for obj in objects:
        bounds = cloud.obj_bounds[obj]
        size += bounds[1]-bounds[0]

    new_vertices = np.zeros((size, 3))
    new_labels = None
    if cloud.labels is not None:
        new_labels = np.zeros((size, 1))
    new_obj_bounds = {}

    offset = 0
    for obj in objects:
        bounds = cloud.obj_bounds[obj]
        obj_size = bounds[1]-bounds[0]
        new_vertices[offset:offset+obj_size] = cloud.vertices[bounds[0]:bounds[1]]
        if cloud.labels is not None:
            new_labels[offset:offset+obj_size] = cloud.labels[bounds[0]:bounds[1]]
        new_obj_bounds[obj] = [offset, offset+obj_size]

    return PointCloud(new_vertices, labels=new_labels, encoding=cloud.encoding, obj_bounds=new_obj_bounds)


def map_labels(cloud: PointCloud, labels: list, target) -> PointCloud:
    """ Returns a PointCloud where all labels given in the labels list got mapped to the target label. E.g. if the
        label array was [1,1,2,3] and the label 1 and 2 were mapped onto the target 3, the label array now is [3,3,3,3].
        This method works for PointClouds and HybridClouds, not for more specific classes (HybridMesh is returned as
        HybridCloud).

    Args:
        cloud: The PointCloud whose labels should get merged.
        labels: A list of keys of the encoding dict of the PointCloud, or a list of actual labels which should get
            mapped onto the target.
        target: A key of the encoding dict of the PointCloud, or an actual label on which the labels should be mapped.

    Returns:
        A PointCloud where the labels were replaced by the target.
    """
    mask = np.zeros(cloud.labels.shape, dtype=bool)
    for label in labels:
        if cloud.encoding is not None and label in cloud.encoding.keys():
            label = cloud.encoding[label]
            mask = np.logical_or(mask, cloud.labels == label)
        else:
            mask = np.logical_or(mask, cloud.labels == label)

    if cloud.encoding is not None and target in cloud.encoding.keys():
        target = cloud.encoding[target]

    new_labels = cloud.labels.copy()
    new_labels[mask] = target

    if cloud.encoding is not None:
        new_encoding = cloud.encoding.copy()
        for label in labels:
            new_encoding.pop(label, None)
    else:
        new_encoding = None

    if isinstance(cloud, HybridCloud):
        new_cloud = HybridCloud(cloud.nodes, cloud.edges, vertices=cloud.vertices, labels=new_labels,
                                encoding=new_encoding)
    else:
        new_cloud = PointCloud(cloud.vertices, labels=new_labels, encoding=new_encoding)
    return new_cloud


# -------------------------------------- CLOUD TRANSFORMATIONS ------------------------------------------- #

class Transformation:
    @property
    def augmentation(self):
        return None

    @property
    def attributes(self):
        return None


class Compose:
    """ Composes several transformations together. """

    def __init__(self, transforms: list):
        self._transforms = transforms

    def __call__(self, pc: PointCloud):
        for t in self._transforms:
            t(pc)

    @property
    def transforms(self):
        return self._transforms


class Identity(Transformation):
    """ This transformation does nothing. """

    def __call__(self, pc: PointCloud):
        return

    @property
    def augmentation(self):
        return False

    @property
    def attributes(self):
        return 1


class Normalization(Transformation):
    def __init__(self, radius: int):
        if radius <= 0:
            radius = 1
        self._radius = -radius

    def __call__(self, pc: PointCloud):
        """ Divides the coordinates of the points by the context size (e.g. radius of the local BFS). If radius is not
            valid (<= 0) it gets set to 1, so that the normalization has no effect.
        """
        pc.scale(self._radius)

    @property
    def radius(self):
        return self._radius

    @property
    def augmentation(self):
        return False

    @property
    def attributes(self):
        return self._radius


class RandomRotate(Transformation):
    def __init__(self, angle_range: tuple = (-180, 180), apply_flip: bool = False):
        self.angle_range = angle_range
        self.apply_flip = apply_flip

    def __call__(self, pc: PointCloud):
        """ Randomly rotates a given PointCloud by performing an Euler rotation. The three angles are chosen randomly
            from the given angle_range. If the PointCloud is a HybridCloud then the nodes get rotated as well. Operates
            in-place for the given Pointcloud. If apply_flip is true, randomly flips spatial axes around origin
            independently.
        """
        pc.rotate_randomly(self.angle_range, random_flip=self.apply_flip)

    @property
    def augmentation(self):
        return True

    @property
    def attributes(self):
        return self.angle_range


class Center(Transformation):
    def __call__(self, pc: PointCloud):
        """ Centers the given PointCloud only with respect to vertices. If the PointCloud is an HybridCloud, the nodes
            get centered as well but are not taken into account for centroid calculation. Operates in-place for the
            given PointCloud
        """
        centroid = np.mean(pc.vertices, axis=0)
        pc.move(-centroid)

    @property
    def augmentation(self):
        return False


class RandomVariation(Transformation):
    def __init__(self, limits: tuple = (-1, 1), distr: str = 'uniform'):
        """

        Args:
            limits: Range of the noise values. Tuple is used as lower and upper bounds for ``distr='uniform'``
                or only the entry at index 1 as scale factor if ``distr='normal'``.
            distr: Noise distribution, currently available: 'uniform' and 'Gaussian'.
        """
        self.limits = limits
        self.noise_distr = distr

    def __call__(self, pc: PointCloud):
        """ Adds some random variation (amplitude given by the limits parameter) to vertices of the given PointCloud.
            Possible nodes get ignored. Operates in-place for the given PointCloud.
        """
        pc.add_noise(self.limits, self.noise_distr)

    @property
    def augmentation(self):
        return True

    @property
    def attributes(self):
        return self.limits, self.noise_distr


class RandomScale(Transformation):
    def __init__(self, distr_scale: float = 0.05, distr: str = 'uniform'):
        """
        Vertices will be multiplied with the factor (1+X), where X is drawn from the
        given distribution `distr` scaled by `distr_scale`.

        Args:
            distr_scale: Scale factor applied to the noise distribution values.
            distr: Noise distribution, currently available: 'uniform' and 'Gaussian'.
        """
        self.distr_scale = distr_scale
        self.noise_distr = distr

    def __call__(self, pc: PointCloud):
        """ Adds some random variation (amplitude given by the limits parameter) to vertices of the given PointCloud.
            Possible nodes get ignored. Operates in-place for the given PointCloud.
        """
        pc.mult_noise(self.distr_scale, self.noise_distr)

    @property
    def augmentation(self):
        return True

    @property
    def attributes(self):
        return self.distr_scale, self.noise_distr


class RandomShear(Transformation):
    def __init__(self, limits: tuple = (-1, 1)):
        self.limits = limits

    def __call__(self, pc: PointCloud):
        pc.shear(limits=self.limits)

    @property
    def augmentation(self):
        return True

    @property
    def attributes(self):
        return self.limits


# -------------------------------------- DIVERSE HELPERS ------------------------------------------- #


def merge_clouds(clouds: List[Union[PointCloud, HybridCloud]], names: Optional[List[Union[str, int]]] = None,
                 ignore_hybrids: bool = False) -> Optional[Union[PointCloud, HybridCloud]]:
    """ Merges the PointCloud objects in the given list. If the names list is given, the object boundary information
        is saved in the obj_bounds dict. Vertices of PointClouds without label / feature get the label /feature -1.
        If no PointCloud has labels / features, then the label /feature array of the merged PointCloud is empty.

    Args:
        clouds: List of clouds which should get merged.
        names: Names for each cloud in order to save object boundaries. This is only used if the clouds themselve have
            no obj_bounds dicts.
        ignore_hybrids: Flag for treating HybridClouds as sole PointClouds (ignoring nodes and edges).

    Returns:
        PointCloud which consists of the merged clouds.
    """
    for cloud in clouds:
        if cloud is None:
            clouds.remove(cloud)

    if names is not None:
        if len(names) != len(clouds):
            raise ValueError("Not enough names given.")

    feats_dim = 1
    if len(clouds[0].features) != 0:
        feats_dim = clouds[0].features.shape[1]
        for cloud in clouds:
            if len(cloud.features) != 0:
                if cloud.features.shape[1] != feats_dim:
                    raise ValueError("Feature dimensions of PointCloud do not fit. Clouds cannot be merged.")

    # find required size of new arrays
    total_verts = 0
    total_nodes = 0
    total_edges = 0
    for cloud in clouds:
        total_verts += len(cloud.vertices)
        if isinstance(cloud, HybridCloud) and cloud.nodes is not None and cloud.edges is not None:
            total_nodes += len(cloud.nodes)
            total_edges += len(cloud.edges)

    # TODO: Generalize to support graph merging as well
    if total_verts == 0:
        return None

    # reserve arrays of required size and initialize new attributes
    t_verts = np.zeros((total_verts, 3))
    t_labels = np.zeros((total_verts, 1))
    t_features = np.zeros((total_verts, feats_dim))
    t_types = np.zeros((total_verts, 1))
    nodes = np.zeros((total_nodes, 3))
    edges = np.zeros((total_edges, 2))
    offset = 0
    obj_bounds = {}
    encoding = {}
    no_pred = []

    for ix, cloud in enumerate(clouds):
        # handle hybrids
        if not ignore_hybrids:
            if isinstance(cloud, HybridCloud) and cloud.nodes is not None and cloud.edges is not None:
                nodes[offset:offset+len(cloud.nodes)] = cloud.nodes
                edges[offset:offset+len(cloud.edges)] = cloud.edges + offset

        # handle pointclouds
        t_verts[offset:offset+len(cloud.vertices)] = cloud.vertices
        if len(cloud.labels) != 0:
            t_labels[offset:offset+len(cloud.vertices)] = cloud.labels
        else:
            t_labels[offset:offset+len(cloud.vertices)] = -1
        if len(cloud.features) != 0:
            t_features[offset:offset+len(cloud.features)] = cloud.features
        else:
            t_features[offset:offset+len(cloud.features)] = -1
        if len(cloud.types) != 0:
            t_types[offset:offset+len(cloud.vertices)] = cloud.types
        else:
            t_types[offset:offset+len(cloud.vertices)] = -1

        # TODO: Handle similar keys from different clouds and handle obj_bounds
        #  which don't span the entire vertex array
        # Save object boundaries
        if cloud.obj_bounds is not None:
            for key in cloud.obj_bounds.keys():
                obj_bounds[key] = cloud.obj_bounds[key] + offset
        else:
            if names is not None:
                obj_bounds[names[ix]] = np.array([offset, offset+len(cloud.vertices)])
        offset += len(cloud.vertices)

        # Merge encodings
        if cloud.encoding is not None:
            for item in cloud.encoding:
                encoding[item] = cloud.encoding[item]

        # Merge no_preds
        if cloud.no_pred is not None:
            for item in cloud.no_pred:
                no_pred.append(item)

    if len(obj_bounds) == 0:
        obj_bounds = None
    if len(encoding) == 0:
        encoding = None
    if np.all(t_types == -1):
        t_types = None
    if np.all(t_labels == -1):
        t_labels = None
    if np.all(t_features == -1):
        t_features = None
    if np.all(nodes == 0):
        nodes = None
    if np.all(edges == 0):
        edges = None

    if ignore_hybrids:
        return PointCloud(t_verts, labels=t_labels, features=t_features, obj_bounds=obj_bounds, encoding=encoding,
                          no_pred=no_pred, types=t_types)
    else:
        return HybridCloud(nodes, edges, vertices=t_verts, labels=t_labels, features=t_features, obj_bounds=obj_bounds,
                           encoding=encoding, no_pred=no_pred, types=t_types)


def merge(clouds: List[PointCloud], names: List = None, preserve_obj_bounds: bool = False) -> PointCloud:
    """ Merges multiple PointClouds. HybridClouds or HybridMeshes are handeld as PointClouds, possible
        skeletons or meshes are ignored. The encoding is set to None if the clouds contain contradicting
        encodings.

        This will replace merge_clouds.

    Args:
        clouds: List of PointClouds which should get merged.
        names: List of names for all PointClouds in clouds. Each name must be unique. If preserve_obj_bounds
            is False, the resulting PointCloud contains object boundaries keyed by the respective names.
        preserve_obj_bounds: Flag for preserving existing object boundaries. The existing keys are updated
            with the respective cloud name. 'obj1' in a cloud with name 'cloud1' becomes to 'cloud1_obj1'.
            If there are no existing object boundaries, the key for the cloud's boundaries is its name.
    """
    # check inputs
    if names is not None and len(names) != len(clouds):
        raise ValueError("Name list for merging must have same length as cloud list.")
    if preserve_obj_bounds and names is None:
        raise ValueError("Name list is required for preserving the object bounds.")
    for name in names:
        if names.count(name) > 1:
            raise ValueError("All names in the name list must be unique.")
    # prepare new properties
    new_size = 0
    for pc in clouds:
        new_size += len(pc.vertices)
    new_verts = np.zeros((new_size, 3))
    new_labels = np.zeros((new_size, 1))
    new_pred_labels = np.zeros((new_size, 1))
    new_features = np.zeros((new_size, 1))
    new_types = np.zeros((new_size, 1))
    offset = 0
    new_encoding = {}
    new_predictions = {}
    new_no_pred = []
    if names is None:
        new_obj_bounds = None
    else:
        new_obj_bounds = {}
    for ix, pc in enumerate(clouds):
        new_verts[offset:offset + len(pc.vertices)] = pc.vertices
        new_labels[offset:offset + len(pc.vertices)] = pc.labels
        new_pred_labels[offset:offset + len(pc.vertices)] = pc.pred_labels
        new_features[offset:offset + len(pc.vertices)] = pc.features
        new_types[offset:offset + len(pc.vertices)] = pc.types
        # create new object bounds
        if names is not None:
            if not preserve_obj_bounds:
                # new object boundaries are keyed by the respective name
                new_obj_bounds[names[ix]] = np.array([offset, offset + len(pc.vertices)])
            else:
                if pc.obj_bounds is None:
                    # with no existing object boundaries, the boundaries for the total cloud are keyed
                    # by the respective name
                    new_obj_bounds[names[ix]] = np.array([offset, offset + len(pc.vertices)])
                else:
                    for key in pc.obj_bounds:
                        # existing boundary keys are updated by the cloud name, e.g. 'obj1' -> 'cloud1_obj1'
                        new_key = names[ix] + '_' + key
                        new_obj_bounds[new_key] = pc.obj_bounds[key] + offset
        # create new encoding
        if new_encoding is not None:
            if pc.encoding is not None:
                for key in pc.encoding:
                    if key in new_encoding:
                        # encoding is set to None if there are contradicting entries
                        if pc.encoding[key] != new_encoding[key]:
                            new_encoding = None
                            break
                    else:
                        new_encoding[key] = pc.encoding[key]
        # create new predictions
        if pc.predictions is not None:
            for key in pc.predictions:
                if key in new_predictions:
                    new_predictions[key].extend(pc.predictions[key])
                else:
                    new_predictions[key] = pc.predictions[key]
        # create new no_pred
        if pc.no_pred is not None:
            for item in pc.no_pred:
                if item not in new_no_pred:
                    new_no_pred.append(item)
    if len(new_no_pred) == 0:
        new_no_pred = None
    if len(new_predictions) == 0:
        new_predictions = None
    return PointCloud(vertices=new_verts, labels=new_labels, pred_labels=new_pred_labels, features=new_features,
                      types=new_types, encoding=new_encoding, obj_bounds=new_obj_bounds, predictions=new_predictions,
                      no_pred=new_no_pred)
