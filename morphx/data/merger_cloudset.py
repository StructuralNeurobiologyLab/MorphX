# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch, Yang Liu

import open3d as o3d
import glob
import numpy as np
import functools
from tqdm import tqdm
from typing import Callable
from scipy.spatial import cKDTree
from morphx.processing import graphs, hybrids, clouds
from morphx.classes.hybridcloud import HybridCloud, PointCloud
from morphx.data.cloudset import CloudSet


@functools.lru_cache(256)
def _load_merger_sample(fpath, query_radius=10e3, down_factor=50):
    # load hybrid cloud
    hc = clouds.load_cloud(fpath)
    # Downsampling
    pcd = o3d.geometry.PointCloud()
    verts = hc._vertices
    pcd.points = o3d.utility.Vector3dVector(verts)
    pcd = pcd.voxel_down_sample(voxel_size=down_factor)
    hc._vertices = np.asarray(pcd.points)

    # kdTree of vertices / nodes
    kdtree_vert = cKDTree(hc._vertices)  # this takes up 100s for large cells
    kdtree_node = cKDTree(hc._nodes)

    node2vert = kdtree_node.query_ball_tree(kdtree_vert, r=query_radius)
    node2node = kdtree_node.query_ball_tree(kdtree_node, r=query_radius)

    all_nodes = hc.nodes
    all_verts = hc._vertices
    all_node_labels = hc.merger_node_labels
    all_vert_labels = hc.labels

    # return node2vert, node2node, all_nodes, all_verts, all_node_labels, all_vert_labels
    return node2vert, node2node, all_nodes, all_verts, all_node_labels


class MergerCloudSetFast:

    def __init__(self,
                 data_path: str,
                 radius_nm: int,
                 sample_num: int,
                 transform: Callable = clouds.Identity(),
                 iterator_method: str = 'global_bfs',
                 global_source: int = -1,
                 radius_factor: float = 1.5,
                 class_num: int = 2,
                 label_filter: list = None,
                 verbose: bool = False,
                 include_skeleton: bool = True
                 ):
        self.files = glob.glob(data_path + '*.pkl')
        self.sample_num = sample_num
        self.transform = transform
        self.cur_fname = self.files[0]
        self.curr_node_idx = 0
        self.include_skeleton = include_skeleton

        # =============================
        # Tuneable parameters:
        # =============================
        # node_idx_list stores all the nodes that will be used as querying location to chunk the point cloud.
        # For segmentation of false-merger, this list should contain equal number of merger/non-merger nodes.
        self.sampled_node_idx = []
        # Number of positive/negative samples, resulting in total number of 2*self.num_posneg_samples
        self.num_posneg_samples = 50
        self.query_radius = 10e3
        # =============================
        # =============================

    def __getitem__(self, index):
        # Get a new file name after the samples are used up. (Determined by self.num_samples_per_cell)
        if self.curr_node_idx >= len(self.sampled_node_idx):
            self.curr_node_idx = 0
            self.cur_fname = self.files[np.random.randint(len(self.files))]
            node2vert, node2node, all_nodes, all_verts, all_node_labels = _load_merger_sample(self.cur_fname,
                                                                                              self.query_radius)
            update_success = self.update_sampled_node_idx(node2vert, node2node, all_nodes, all_node_labels)

            while not update_success:
                self.curr_node_idx = 0
                self.cur_fname = self.files[np.random.randint(len(self.files))]
                node2vert, node2node, all_nodes, all_verts, all_node_labels = _load_merger_sample(self.cur_fname,
                                                                                                  self.query_radius)
                update_success = self.update_sampled_node_idx(node2vert, node2node, all_nodes, all_node_labels)

        else:
            node2vert, node2node, all_nodes, all_verts, all_node_labels= _load_merger_sample(self.cur_fname,
                                                                                              self.query_radius)

        query_center_idx = self.sampled_node_idx[self.curr_node_idx]
        chunk_vert_ixs = node2vert[query_center_idx]
        chunk_node_ixs = node2node[query_center_idx]
        chunk_vertices = all_verts[chunk_vert_ixs]
        # chunk_vert_labels = all_vert_labels[chunk_vert_ixs]
        chunk_nodes = all_nodes[chunk_node_ixs]
        chunk_node_labels = all_node_labels[chunk_node_ixs]
        # convert all -1 to 0 in chunk_node_labels
        chunk_node_labels = np.where(chunk_node_labels == -1, 0, chunk_node_labels)

        # subset = PointCloud(vertices=chunk_vertices, labels=chunk_vert_labels)
        subset = PointCloud(vertices=chunk_vertices)
        sample_cloud = clouds.sample_cloud(subset, self.sample_num)

        # Tempprally add nodes into _vertices in point cloud
        num_nodes = len(chunk_nodes)
        sample_cloud._vertices = np.concatenate((sample_cloud._vertices, chunk_nodes), axis=0)

        # apply transformations
        if len(sample_cloud.vertices) > 0:
            self.transform(sample_cloud)

        # Remove nodes coordinates from _vertices
        chunk_nodes = sample_cloud._vertices[self.sample_num:]
        sample_cloud._vertices = sample_cloud._vertices[:-num_nodes]

        self.curr_node_idx += 1

        if self.include_skeleton:
            chunk_nodes, chunk_node_labels = self.skel_node_fixation(chunk_nodes, chunk_node_labels, num_nodes=500)
            return sample_cloud, chunk_nodes, chunk_node_labels
        else:
            return sample_cloud

    def update_sampled_node_idx(self, node2vert, node2node, all_nodes, node_labels) -> bool:
        node_idx_merger = np.argwhere(node_labels.squeeze() == 1).squeeze()
        node_idx_no_merger = np.argwhere(node_labels.squeeze() == 0).squeeze()

        # Iterate through every merger_node, and query all the nodes within the radius that centered around the
        # current merger_node. These nodes combined are considered to be the query-center that will generate chunk of
        # vertices that contains the merger.
        nodes_to_filter = set()
        for index in node_idx_merger:
            chunk_node_ixs_set = set(node2node[index])
            nodes_to_filter.update(chunk_node_ixs_set)

        # Filter out `nodes_to_filter` from node_idx_no_merger
        # Nodes that left are considered not containing the merger.
        filtered_nodes_idx_no_merger = [index for index in node_idx_no_merger if index not in nodes_to_filter]
        filtered_nodes_idx_no_merger = np.array(filtered_nodes_idx_no_merger)
        if len(filtered_nodes_idx_no_merger) == 0:
            return False

        # Also expand the node_idx_merger so that the merger is not always in the center of the chunk
        merger_radius = self.query_radius - 3.5e3
        assert merger_radius > 0, "self.query_radius must be bigger than 3.5e3"
        nodes_idx_merger_expanded = set()
        for index in node_idx_merger:
            chunk_node_ixs_set = set(node2node[index])
            nodes_idx_merger_expanded.update(chunk_node_ixs_set)
        nodes_idx_merger_expanded = np.array(list(nodes_idx_merger_expanded))
        if len(nodes_idx_merger_expanded) == 0:
            return False

        num_samples = self.num_posneg_samples

        merger_subset = np.random.choice(nodes_idx_merger_expanded, num_samples)
        no_merger_subset = np.random.choice(filtered_nodes_idx_no_merger, num_samples)

        self.sampled_node_idx = np.concatenate((merger_subset, no_merger_subset))
        np.random.shuffle(self.sampled_node_idx)

        return True

    def skel_node_fixation(self, skel_nodes, node_labels, num_nodes=500):
        """
        Upsample or downsample the skeleton node to a given fix number (num_nodes).
        This is necessary since PyTorch batch loader requires tensors to have the same size.
        Args:
            skel_nodes: shape (N,3), list of (x,y,z) coordinates.
            node_labels: shape (N, 1), list of node labels, either 0(non-merger-node) or 1(merger_node)
            num_nodes: number of nodes the output skeleton should have.

        Returns:

        """
        assert len(skel_nodes) == len(node_labels), "inconsistent length of skel_nodes array and node_labels array"
        if len(skel_nodes) > num_nodes:
            # Downsample the skeleton nodes to num_nodes
            diff = len(skel_nodes) - num_nodes
            # Get the list of indexes of label 1 and label 0
            idx_label_1 = np.argwhere(node_labels.squeeze() == 1).squeeze()
            idx_label_0 = np.argwhere(node_labels.squeeze() == 0).squeeze()

            # Calculate the ratio of label 1.
            if len(idx_label_1.shape) == 0:
                idx_label_1 = np.array([idx_label_1])
            ratio = len(idx_label_1) / len(node_labels)

            # Calculate the number of labels should be cut out for 1 and 0 respectively
            num_draws_1 = int(diff * ratio)
            num_draws_0 = diff - num_draws_1
            num_remain_1 = len(idx_label_1) - num_draws_1
            num_remain_0 = len(idx_label_0) - num_draws_0
            assert 0 < num_remain_0 < len(idx_label_0)

            # Randomly draw out indexes from label 1 and label 0 that should remain in the skeleton
            idx_remain_1 = np.random.choice(idx_label_1, num_remain_1, replace=False)
            idx_remain_0 = np.random.choice(idx_label_0, num_remain_0, replace=False)
            idx_remain = np.sort(np.concatenate((idx_remain_1, idx_remain_0)))
            skel_nodes = skel_nodes[idx_remain]
            node_labels = node_labels[idx_remain]

        if len(skel_nodes) < num_nodes:
            # Upsample the skeleton nodes to num_nodes
            diff = num_nodes - len(skel_nodes)
            # Randomly pick some indexes nodes and repeat those nodes in the original array
            random_idx = np.random.randint(len(skel_nodes), size=diff)
            for idx in random_idx:
                skel_nodes = np.append(skel_nodes, skel_nodes[idx].reshape(1, 3), axis=0)
                node_labels = np.append(node_labels, node_labels[idx])

        return skel_nodes, node_labels

    def __len__(self):
        return len(self.files) * self.num_posneg_samples * 2


class MergerCloudSet:
    """ Dataset iterator class that creates point cloud samples from point clouds in pickle files at data_path. """

    def __init__(self,
                 data_path: str,
                 radius_nm: int,
                 sample_num: int,
                 transform: Callable = clouds.Identity(),
                 iterator_method: str = 'global_bfs',
                 global_source: int = -1,
                 radius_factor: float = 1.5,
                 class_num: int = 2,
                 label_filter: list = None,
                 verbose: bool = False,
                 include_skeleton: bool = True):
        """ Initializes Dataset.

        Args:
            data_path: Absolute path to data.
            radius_nm: The size of the chunks in nanometers.
            sample_num: The number of samples for each chunk.
            transform: Transformations from elektronn3.data.transform.transforms3d which should be applied to incoming
                data.
            iterator_method: The method with which each cell should be iterated.
            global_source: The starting point of the iterator method.
            radius_factor: Factor with which radius of global BFS should be calculated. Should be larger than 1, as it
                adjusts the overlap between the cloud chunks.
            class_num: Number of classes.
            label_filter: List of labels after which the dataset should be filtered.
            include_skeleton: return point cloud and skeleton nodes (x,y,z) coordinates.
        """

        self.data_path = data_path
        self.radius_nm = radius_nm
        self.sample_num = sample_num
        self.iterator_method = iterator_method
        self.global_source = global_source
        self.transform = transform
        self.radius_factor = radius_factor
        self.class_num = class_num
        self.label_filter = label_filter
        self.verbose = verbose
        self.include_skeleton = include_skeleton
        # This two kd-trees will be initialized every time a new pcl is loaded with self.load_new()
        self.kdtree_vert = None
        self.kdtree_node = None

        # find and prepare analysis parameters
        self.files = glob.glob(data_path + '*.pkl')
        # self.files = glob.glob(data_path + 'sso_4924_info.pkl')  # TODO: delete this test
        self.size = 0
        self.size_cache = 0
        self._weights = np.ones(class_num)

        # option for single processing
        self.process_single = False

        # options for iterating the dataset
        # self.curr_hybrid_idx = 0
        self.curr_hybrid_idx = np.random.randint(0, len(self.files))
        self.curr_node_idx = 0
        self.radius_nm_global = radius_nm*self.radius_factor

        # =============================
        # Tuneable parameters:
        # =============================
        # node_idx_list stores all the nodes that will be used as querying location to chunk the point cloud.
        # For segmentation of false-merger, this list should contain equal number of merger/non-merger nodes.
        self.sampled_node_idx = []
        # Number of positive/negative samples, resulting in total number of 2*self.num_posneg_samples
        self.num_posneg_samples = 50
        self.query_radius = 10e3
        # =============================
        # =============================

        # # load first file
        # self.curr_hybrid = None
        # if len(self.files) > 0:
        #     load_new_success = self.load_new()
        #     while not load_new_success:
        #         load_new_success = self.load_new()

        self.analyse_data()

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        """ Index gets ignored. """
        # prepare new cell if current one is exhausted
        if self.curr_node_idx >= len(self.sampled_node_idx):

            # process_single finished => switch back to normal when done
            if self.process_single is True:
                self.process_single = False
                self.size = self.size_cache
                # self.load_new()
                load_new_success = self.load_new()
                while not load_new_success:
                    load_new_success = self.load_new()
                return None
            else:
                # self.load_new()
                load_new_success = self.load_new()
                while not load_new_success:
                    load_new_success = self.load_new()
                # print("Loaded new cell as point cloud.")

        # node_idx_merger = np.argwhere(self.curr_hybrid.node_labels == 1)
        # node_idx_no_merger = np.argwhere(self.curr_hybrid.node_labels == 0)

        spoint = self.sampled_node_idx[self.curr_node_idx]
        node_coord = self.curr_hybrid.nodes[spoint]
        chunk_node_ixs = self.kdtree_node.query_ball_point(node_coord, r=self.query_radius)
        chunk_vert_ixs = self.kdtree_vert.query_ball_point(node_coord, r=self.query_radius)
        chunk_vertices = self.curr_hybrid.vertices[chunk_vert_ixs]
        chunk_vert_labels = self.curr_hybrid.labels[chunk_vert_ixs]
        chunk_nodes = self.curr_hybrid.nodes[chunk_node_ixs]
        chunk_node_labels = self.curr_hybrid.merger_node_labels[chunk_node_ixs]
        # convert all -1 to 0 in chunk_node_labels
        chunk_node_labels = np.where(chunk_node_labels==-1, 0, chunk_node_labels)

        subset = PointCloud(vertices=chunk_vertices, labels=chunk_vert_labels)
        sample_cloud = clouds.sample_cloud(subset, self.sample_num)

        # Temporally add nodes in to _vertices in point cloud
        num_nodes = len(chunk_nodes)
        sample_cloud._vertices = np.concatenate((sample_cloud._vertices, chunk_nodes), axis=0)

        # apply transformations
        if len(sample_cloud.vertices) > 0:
            self.transform(sample_cloud)

        # Remove nodes coordinates from _vertices
        chunk_nodes = sample_cloud._vertices[self.sample_num:]
        sample_cloud._vertices = sample_cloud._vertices[:-num_nodes]

        # Set pointer to next node of global BFS
        self.curr_node_idx += 1

        if self.verbose:
            # return aug_cloud, local_bfs
            return sample_cloud, chunk_node_ixs
        elif self.include_skeleton:
            chunk_nodes, chunk_node_labels = self.skel_node_fixation(chunk_nodes, chunk_node_labels, num_nodes=500)
            return sample_cloud, chunk_nodes, chunk_node_labels
        else:
            return sample_cloud

    @property
    def weights(self):
        return self._weights

    def set_verbose(self):
        self.verbose = True

    def activate_single(self, hybrid: HybridCloud):
        """ Switch cloudset mode to only process the given hybrid

        Args:
            hybrid: The specific hybrid pointcloud which should be processed.
        """

        self.curr_hybrid = hybrid
        self.curr_hybrid.traverser(method=self.iterator_method,
                                   min_dist=self.radius_nm_global,
                                   source=self.global_source)
        self.size_cache = self.size
        self.size = len(self.curr_hybrid.traverser())
        self.process_single = True
        self.curr_node_idx = 0

    def load_new(self) -> bool:
        """ Load next hybrid from dataset and apply possible filters """

        # if self.verbose:
        print("Loading new cell from: {}.".format(self.files[self.curr_hybrid_idx]))

        self.curr_hybrid = clouds.load_cloud(self.files[self.curr_hybrid_idx])
        if self.label_filter is not None:
            self.curr_hybrid = clouds.filter_labels(self.curr_hybrid, self.label_filter)

        if self.verbose:
            print("Calculating traverser...")

        # self.curr_hybrid.traverser(method=self.iterator_method,
        #                            min_dist=self.radius_nm_global,
        #                            source=self.global_source)

        # Fill in node_idx_list with equal number of merger/non-merger node idx.
        node_idx_merger = np.argwhere(self.curr_hybrid.merger_node_labels == 1)
        node_idx_no_merger = np.argwhere(self.curr_hybrid.merger_node_labels == 0)

        # Iterate through every merger_node, and query all the nodes within the radius that centered around the
        # current merger_node. These nodes combined are considered to be the query-center that will generate chunk of
        # vertices that contains the merger.
        self.kdtree_vert = cKDTree(self.curr_hybrid.vertices)  # this takes up 100s for large cells
        self.kdtree_node = cKDTree(self.curr_hybrid.nodes)
        nodes_to_filter = set()
        for index in node_idx_merger:
            node_coord = self.curr_hybrid.nodes[index]
            chunk_node_ixs = self.kdtree_node.query_ball_point(node_coord, r=self.query_radius).tolist()
            chunk_node_ixs_set = set(chunk_node_ixs[0])
            nodes_to_filter.update(chunk_node_ixs_set)

        # Filter out `nodes_to_filter` from node_idx_no_merger
        node_idx_no_merger = node_idx_no_merger.flatten()
        # Nodes left are considered not containing the merger.
        filtered_nodes_idx_no_merger = [index for index in node_idx_no_merger if index not in nodes_to_filter]
        filtered_nodes_idx_no_merger = np.array(filtered_nodes_idx_no_merger)
        if len(filtered_nodes_idx_no_merger) == 0:
            # self.curr_hybrid_idx += 1
            # if self.curr_hybrid_idx >= len(self.files):
            #     self.curr_hybrid_idx = 0
            self.curr_hybrid_idx = np.random.randint(0, len(self.files))
            return False

        # Also expand the node_idx_merger so that the merger is not always in the center of the chunk
        merger_radius = self.query_radius - 3.5e3
        nodes_idx_merger_expanded = set()
        for index in node_idx_merger:
            node_coord = self.curr_hybrid.nodes[index]
            chunk_node_ixs = self.kdtree_node.query_ball_point(node_coord, r=merger_radius).tolist()
            chunk_node_ixs_set = set(chunk_node_ixs[0])
            nodes_idx_merger_expanded.update(chunk_node_ixs_set)
        nodes_idx_merger_expanded = np.array(list(nodes_idx_merger_expanded))
        if len(nodes_idx_merger_expanded) == 0:
            # self.curr_hybrid_idx += 1
            # if self.curr_hybrid_idx >= len(self.files):
            #     self.curr_hybrid_idx = 0
            self.curr_hybrid_idx = np.random.randint(0, len(self.files))
            return False

        # Randomly choose the subset
        num_samples = self.num_posneg_samples
        # if len(nodes_idx_merger_expanded) < self.num_posneg_samples or \
        #         len(filtered_nodes_idx_no_merger) < self.num_posneg_samples:
        #     num_samples = min(len(node_idx_merger), len(node_idx_no_merger))
        merger_subset = np.random.choice(np.squeeze(nodes_idx_merger_expanded), num_samples)
        no_merger_subset = np.random.choice(np.squeeze(filtered_nodes_idx_no_merger), num_samples)

        self.sampled_node_idx = np.concatenate((merger_subset, no_merger_subset))
        np.random.shuffle(self.sampled_node_idx)

        # if self.label_filter is not None:
        #     self.curr_hybrid.filter_traverser()

        # self.curr_hybrid_idx += 1
        # # start over if all files have been processed
        # if self.curr_hybrid_idx >= len(self.files):
        #     self.curr_hybrid_idx = 0
        self.curr_hybrid_idx = np.random.randint(0, len(self.files))

        self.curr_node_idx = 0

        # # load next if current cloud doesn't contain the requested labels
        # if len(self.curr_hybrid.traverser()) == 0:
        #     self.load_new()
        return True
        

    def analyse_data(self):
        """ Count number of chunks which can be generated with current settings and calculate class
            weights based on occurences in dataset. """

        if len(self.files) == 0:
            return

        # print("Analysing data...")
        # # put all clouds together for weight calculation
        # total_pc = self.curr_hybrid
        # datasize = len(self.curr_hybrid.traverser())
        #
        # # iterate remaining files
        # for i in tqdm(range(len(self.files)-1)):
        #     self.load_new()
        #     total_pc = clouds.merge_clouds(total_pc, self.curr_hybrid)
        #     datasize += len(self.curr_hybrid.traverser())
        # self.size = datasize
        # print("Chunking data into {} pieces.".format(datasize))
        #
        # self._weights = clouds.calculate_weights_mean(total_pc, self.class_num)

        self.size = len(self.files) * self.num_posneg_samples * 2


    def skel_node_fixation(self, skel_nodes, node_labels, num_nodes=500):
        """
        Upsample or downsample the skeleton node to a given fix number (num_nodes).
        This is necessary since PyTorch batch loader requires tensors to have the same size.
        Args:
            skel_nodes: shape (N,3), list of (x,y,z) coordinates.
            node_labels: shape (N, 1), list of node labels, either 0(non-merger-node) or 1(merger_node)
            num_nodes: number of nodes the output skeleton should have.

        Returns:

        """
        assert len(skel_nodes) == len(node_labels), "inconsistent length of skel_nodes array and node_labels array"
        if len(skel_nodes) > num_nodes:
            # Downsample the skeleton nodes to num_nodes
            diff = len(skel_nodes) - num_nodes
            # Get the list of indexes of label 1 and label 0
            idx_label_1 = np.argwhere(node_labels.squeeze() == 1).squeeze()
            idx_label_0 = np.argwhere(node_labels.squeeze() == 0).squeeze()

            # Calculate the ratio of label 1.
            ratio = len(idx_label_1) / len(node_labels)
            # Calculate the number of labels should be cut out for 1 and 0 respectively
            num_draws_1 = int(diff * ratio)
            num_draws_0 = diff - num_draws_1
            num_remain_1 = len(idx_label_1) - num_draws_1
            num_remain_0 = len(idx_label_0) - num_draws_0
            assert 0 < num_remain_0 < len(idx_label_0)

            # Randomly draw out indexes from label 1 and label 0 that should remain in the skeleton
            idx_remain_1 = np.random.choice(idx_label_1, num_remain_1, replace=False)
            idx_remain_0 = np.random.choice(idx_label_0, num_remain_0, replace=False)
            idx_remain = np.sort(np.concatenate((idx_remain_1, idx_remain_0)))
            skel_nodes = skel_nodes[idx_remain]
            node_labels = node_labels[idx_remain]

        if len(skel_nodes) < num_nodes:
            # Upsample the skeleton nodes to num_nodes
            diff = num_nodes - len(skel_nodes)
            # Randomly pick some indexes nodes and repeat those nodes in the original array
            random_idx = np.random.randint(len(skel_nodes), size=diff)
            for idx in random_idx:
                skel_nodes = np.append(skel_nodes, skel_nodes[idx].reshape(1, 3), axis=0)
                node_labels = np.append(node_labels, node_labels[idx])

        return skel_nodes, node_labels
