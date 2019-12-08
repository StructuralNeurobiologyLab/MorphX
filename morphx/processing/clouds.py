# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import math
import os
import pickle
import numpy as np
from morphx.classes.pointcloud import PointCloud
from morphx.classes.hybridcloud import HybridCloud
from scipy.spatial.transform import Rotation as Rot


# -------------------------------------- CLOUD SAMPLING ------------------------------------------- #


def sample_cloud(pc: PointCloud, vertex_number: int, random_seed=None) -> PointCloud:
    """ Creates a (pseudo)random sample point cloud with a specific number of points from the given subset of mesh
    vertices. If the requested number of points is larger than the given subset, the subset gets enriched with slightly
    augmented points before sampling.

    Args:
        pc: MorphX PointCloud object which should be sampled.
        vertex_number: The number of points which should make up the sample point cloud.
        random_seed: Possibility for making the sampling deterministic.

    Returns:
        PointCloud with sampled points (and labels).
    """
    cloud = pc.vertices
    labels = pc.labels

    if len(cloud) == 0:
        return pc

    if random_seed is not None:
        np.random.seed(random_seed)

    dim = cloud.shape[1]
    sample = np.zeros((vertex_number, dim))
    sample_l = np.zeros((vertex_number, 1))

    deficit = vertex_number - len(cloud)

    vert_ixs = np.arange(len(cloud))
    np.random.shuffle(vert_ixs)
    sample[:min(len(cloud), vertex_number)] = cloud[vert_ixs[:vertex_number]]
    if labels is not None:
        sample_l[:min(len(cloud), vertex_number)] = labels[vert_ixs[:vertex_number]]

    # add augmented points to reach requested number of samples
    if deficit > 0:
        # deficit could be bigger than cloudsize
        offset = len(cloud)
        for it in range(math.ceil(deficit/len(cloud))):
            compensation = min(len(cloud), len(sample)-offset)
            np.random.shuffle(vert_ixs)
            sample[offset:offset+compensation] = cloud[vert_ixs[:compensation]]
            if labels is not None:
                sample_l[offset:offset+compensation] = labels[vert_ixs[:compensation]]
            offset += compensation

        # TODO: change to augmentation method from elektronn3
        sample[len(cloud):] += np.random.random(sample[len(cloud)].shape)

    if labels is not None:
        return PointCloud(sample, labels=sample_l)
    else:
        return PointCloud(sample)


def filter_labels(cloud: PointCloud, labels: list) -> PointCloud:
    """ Returns a pointcloud which contains only those vertices witch labels occuring in 'labels'. If 'cloud'
        is a HybridCloud, the skeleton is taken as it is and should later be filtered with the 'filter_traverser'
        method.

    Args:
        cloud: PointCloud which should be filtered.
        labels: List of labels for which the corresponding vertices should be extracted.

    Returns:
        PointCloud object which contains only vertices with the filtered labels. Skeletons in case of HybridClouds are
        the same.
    """

    mask = np.zeros(cloud.labels.shape, dtype=bool)
    for label in labels:
        mask = np.logical_or(mask, cloud.labels == label)

    mask = mask.reshape(len(mask))
    if isinstance(cloud, HybridCloud):
        f_cloud = HybridCloud(cloud.nodes, cloud.edges, cloud.vertices[mask], labels=cloud.labels[mask])
    else:
        f_cloud = PointCloud(cloud.vertices[mask], labels=cloud.labels[mask])
    return f_cloud


# -------------------------------------- CLOUD I/O ------------------------------------------- #


def save_cloud(cloud: PointCloud, path: str, name='cloud', simple=True) -> int:
    """ Saves PointCloud object to given path.

    Args:
        cloud: PointCloud object which should be saved.
        path: Location where the object should be saved to (only folder).
        name: Name of file in which the object should be saved.
        simple: If object is also HybridCloud, then this flag can be used to only save basic information instead of the
            total object.

    Returns:
        1 if saving process was successful, 0 otherwise.
    """
    full_path = os.path.join(path, name + '.pkl')
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(full_path, 'wb') as f:
            if isinstance(cloud, HybridCloud) and simple:
                cloud = {'nodes': cloud.nodes, 'edges': cloud.edges, 'vertices': cloud.vertices,
                         'labels': cloud.labels}
            pickle.dump(cloud, f)
    except FileNotFoundError:
        print("Saving was not successful as given path is not valid.")
        return 0
    return 1


# TODO: Improve
def save_cloudlist(clouds: list, path: str, name='cloudlist') -> int:
    full_path = os.path.join(path, name + '.pkl')
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(full_path, 'wb') as f:
            pickle.dump(clouds, f)
    except FileNotFoundError:
        print("Saving was not successful as given path is not valid.")
        return 0
    return 1


def load_cloud(path) -> PointCloud:
    """ Loads and returns PointCloud object from given path. """

    with open(path, 'rb') as f:
        cloud = pickle.load(f)

    if isinstance(cloud, dict):
        return HybridCloud(cloud['nodes'], cloud['edges'], cloud['vertices'], labels=cloud['labels'])

    return cloud


# TODO: Remove when ground truth gets changed to new standard
def load_gt(path: str) -> HybridCloud:
    """ Loads morphx hybrid from a pickle file at the given path. Pickle files should contain a dict with the
    keys: 'skel_nodes', 'skel_edges', 'mesh_verts' and 'vert_labels' representing skeleton nodes and edges and mesh
    vertices and labels. """

    with open(path, "rb") as f:
        info_dict = pickle.load(f)
    f.close()
    hc = HybridCloud(info_dict['skel_nodes'], info_dict['skel_edges'], info_dict['mesh_verts'],
                     labels=info_dict['vert_labels'])
    return hc


# -------------------------------------- CLOUD TRANSFORMATIONS ------------------------------------------- #


class Identity:
    def __call__(self, pc: PointCloud):
        return pc


class Compose:
    """ Composes several transforms together. """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, pc: PointCloud):
        for t in self.transforms:
            pc = t(pc)
        return pc


class RandomRotate:
    """ Rotation of 3D PointClouds. Range of possible angles can be set with angle_range. """

    def __init__(self, angle_range: tuple = (-180, 180)):
        self.angle_range = angle_range

    def __call__(self, pc: PointCloud) -> PointCloud:
        """ Randomly rotates a given PointCloud by performing an Euler rotation. The three angles are choosen randomly
         from the given angle_range. If the PointCloud is a HybridCloud then the nodes get rotated as well. """

        angles = np.random.uniform(self.angle_range[0], self.angle_range[1], (1, 3))[0]
        r = Rot.from_euler('xyz', angles, degrees=True)

        vertices = pc.vertices
        r_vertices = r.apply(vertices)
        pc.set_vertices(r_vertices)

        if isinstance(pc, HybridCloud):
            pc.set_nodes(r.apply(pc.nodes))

        return pc


class Center:
    def __call__(self, pc: PointCloud) -> PointCloud:
        """ Centers the given PointCloud only with respect to vertices. If the PointCloud is an HybridCloud, the nodes
         get centered as well but are not taken into account for centroid calculation. """

        vertices = pc.vertices
        centroid = np.mean(vertices, axis=0)
        c_vertices = vertices - centroid
        pc.set_vertices(c_vertices)

        if isinstance(pc, HybridCloud):
            pc.set_nodes(pc.nodes - centroid)

        return pc


class RandomVariation:
    """ Additional noise for PointCloud objects. The noise amplitude range can be set with limits. """

    def __init__(self, limits: tuple = (-1, 1)):
        if limits[0] < limits[1]:
            self.limits = limits
        elif limits[0] > limits[1]:
            self.limits = (limits[1], limits[0])
        else:
            self.limits = (0, 0)

    def __call__(self, pc: PointCloud) -> PointCloud:
        """ Adds some random variation to vertices of the given PointCloud. Possible nodes get ignored. """

        if self.limits == (0, 0):
            return pc
        vertices = pc.vertices
        variation = np.random.random(vertices.shape) * (self.limits[1] - self.limits[0]) + self.limits[0]
        pc.set_vertices(vertices+variation)
        return pc

# -------------------------------------- CLOUD ANALYSIS ------------------------------------------- #


def get_variation(pc: PointCloud):
    var = np.unique(pc.labels, return_counts=True)
    return var[1] / len(pc.labels)


def calculate_weights_mean(pc: PointCloud, class_num: int):
    """ Extract frequences for each class and calculate weights as frequences.mean() / frequences, ignoring any
    labels which don't appear in the dataset (setting their weight to 0).

    Args:
        pc: Pointcloud for which the weights should be calculated.
        class_num: Number of classes.
    """

    total_labels = pc.labels
    non_zero = []
    freq = []
    for i in range(class_num):
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


def calculate_weights_occurence(pc: PointCloud, class_num: int):
    """ Extract frequences for each class and calculate weights as len(vertices) / frequences.

    Args:
        pc: Pointcloud for which the weights should be calculated.
        class_num: Number of classes.
    """

    total_labels = pc.labels
    freq = []
    for i in range(class_num):
        freq.append((total_labels == i).sum())
    freq = len(total_labels) / np.array(freq)
    return freq


# -------------------------------------- DIVERSE HELPERS ------------------------------------------- #


# TODO: Generalize to hybrids
def merge_clouds(pc1: PointCloud, pc2: PointCloud) -> PointCloud:
    """ Merges 2 PointCloud Objects if dimensions match and if either both clouds have labels or none has.

    Args:
        pc1: First PointCloud object
        pc2: Second PointCloud object

    Returns:
        PointCloud object which was build by merging the given two clouds.
    """
    dim1 = pc1.vertices.shape[1]
    dim2 = pc2.vertices.shape[1]
    if dim1 != dim2:
        raise Exception("PointCloud dimensions do not match")

    merged_vertices = np.zeros((len(pc1.vertices)+len(pc2.vertices), dim1))
    merged_labels = np.zeros((len(merged_vertices),1))
    merged_vertices[:len(pc1.vertices)] = pc1.vertices
    merged_vertices[len(pc1.vertices):] = pc2.vertices

    if pc1.labels is None and pc2.labels is None:
        return PointCloud(merged_vertices)
    elif pc1.labels is None or pc2.labels is None:
        raise Exception("PointCloud label is None at one PointCloud but exists at the other. "
                        "PointClouds are not compatible")
    else:
        merged_labels[:len(pc1.vertices)] = pc1.labels
        merged_labels[len(pc1.vertices):] = pc2.labels
        return PointCloud(merged_vertices, labels=merged_labels)
