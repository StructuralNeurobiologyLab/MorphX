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
from morphx.classes.hybridmesh import HybridMesh


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

        sample[len(cloud):] += np.random.random(sample[len(cloud)].shape)

    if labels is not None:
        return PointCloud(sample, labels=sample_l)
    else:
        return PointCloud(sample)


# -------------------------------------- CLOUD FILTERING / LABEL MAPPING ------------------------------------------- #


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
        f_cloud = HybridCloud(cloud.nodes, cloud.edges, cloud.vertices[mask], labels=cloud.labels[mask])
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
        This method works for PointClouds and HybridClouds, not for more specific classes.

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
        new_cloud = HybridCloud(cloud.nodes, cloud.edges, cloud.vertices, labels=new_labels, encoding=new_encoding)
    else:
        new_cloud = PointCloud(cloud.vertices, labels=new_labels, encoding=new_encoding)
    return new_cloud


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
        0 if saving process was successful, 1 otherwise.
    """
    full_path = os.path.join(path, name + '.pkl')
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(full_path, 'wb') as f:
            if isinstance(cloud, HybridCloud) and simple:
                cloud = {'nodes': cloud.nodes, 'edges': cloud.edges, 'vertices': cloud.vertices,
                         'labels': cloud.labels, 'encoding': cloud.encoding}
            pickle.dump(cloud, f)
        f.close()
    except FileNotFoundError:
        print("Saving was not successful as given path is not valid.")
        return 1
    return 0


# TODO: Improve
def save_cloudlist(clouds: list, path: str, name='cloudlist') -> int:
    full_path = os.path.join(path, name + '.pkl')
    try:
        if not os.path.exists(path):
            os.makedirs(path)
        with open(full_path, 'wb') as f:
            pickle.dump(clouds, f)
        f.close()
    except FileNotFoundError:
        print("Saving was not successful as given path is not valid.")
        return 1
    return 0


def load_cloud(path) -> PointCloud:
    """ Loads an MorphX object or an attribute dict from a pickle file.

    Args:
        path: Location of pickle file.
    """
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        print("File with name: {} was not found at this location.".format(path))

    with open(path, 'rb') as f:
        obj = pickle.load(f)
    f.close()

    # return if loaded object is MorphX class already
    if isinstance(obj, PointCloud):
        return obj

    # TODO: Create standard notation
    # check dict keys to find which object is saved and load the respective MorphX class
    if isinstance(obj, dict):
        keys = obj.keys()
        if 'indices' in keys:
            return HybridMesh(obj['nodes'], obj['edges'], obj['vertices'], obj['indices'], obj['normals'],
                              labels=obj['labels'], encoding=obj['encoding'])
        elif 'nodes' in keys:
            try:
                return HybridCloud(obj['nodes'], obj['edges'], obj['vertices'], labels=obj['labels'],
                                   encoding=obj['encoding'])
            except KeyError:
                return HybridCloud(obj['nodes'], obj['edges'], obj['vertices'], labels=obj['labels'])
        elif 'skel_nodes' in keys:
            return HybridCloud(obj['skel_nodes'], obj['skel_edges'], obj['mesh_verts'], labels=obj['vert_labels'])

    raise ValueError("Datatype of given object was not understood.")


# -------------------------------------- CLOUD TRANSFORMATIONS ------------------------------------------- #


class Compose:
    """ Composes several transformations together. """

    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, pc: PointCloud):
        for t in self.transforms:
            t(pc)


class Identity:
    """ This transformation does nothing. """

    def __call__(self, pc: PointCloud):
        return


class Normalization:
    """ Divides the coordinates of the points by the context size (e.g. radius of the local BFS). If radius is not
        valid (<= 0) it gets set to 1, so that the normalization has no effect. """

    def __init__(self, radius: int):
        self.radius = radius

    def __call__(self, pc: PointCloud):
        pc.normalize(self.radius)


class RandomRotate:
    """ Randomly rotates a given PointCloud by performing an Euler rotation. The three angles are choosen randomly
        from the given angle_range. If the PointCloud is a HybridCloud then the nodes get rotated as well. Operates
        in-place for the given Pointcloud. """

    def __init__(self, angle_range: tuple = (-180, 180)):
        self.angle_range = angle_range

    def __call__(self, pc: PointCloud):
        pc.rotate_randomly(self.angle_range)


class Center:
    """ Centers the given PointCloud only with respect to vertices. If the PointCloud is an HybridCloud, the nodes
     get centered as well but are not taken into account for centroid calculation. Operates in-place for the
     given PointCloud"""

    def __call__(self, pc: PointCloud):
        pc.center()


class RandomVariation:
    """ Adds some random variation (amplitude given by the limits parameter) to vertices of the given PointCloud.
        Possible nodes get ignored. Operates in-place for the given PointCloud. """

    def __init__(self, limits: tuple = (-1, 1)):
        self.limits = limits

    def __call__(self, pc: PointCloud):
        pc.add_noise()


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
        merged_labels[:len(pc1.vertices)] = pc1.labels.reshape((len(pc1.labels), 1))
        merged_labels[len(pc1.vertices):] = pc2.labels.reshape((len(pc2.labels), 1))
        return PointCloud(merged_vertices, labels=merged_labels)
