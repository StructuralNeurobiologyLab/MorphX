# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import pickle
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from morphx.processing import clouds, visualize
from morphx.data.cloudset import CloudSet


def create_hist(labels: np.ndarray, save_path: str):
    plt.style.use('seaborn-white')
    bins = np.arange(min(labels) - 0.5, max(labels) + 1.5)
    plt.hist(labels, bins=bins, edgecolor='w')
    plt.xticks(np.arange(min(labels), max(labels) + 1))
    plt.grid()
    plt.title('Label distribution')
    plt.savefig(save_path)
    plt.close()


def save_cloudset(cloudset: CloudSet, save_path: str):
    """ This method iterates through the dataset of the given cloudset and saves all the results so that they can be
        inspected by using the morphx.utils.viewcontrol load_cloudset method.

    Args:
        cloudset: CloudSet which was initialized on the dataset of intereset.
        save_path: path to folder where results should get saved into.
    """

    save_path = os.path.expanduser(save_path)
    cloudset.set_verbose()
    idx = cloudset.curr_hybrid_idx
    print("Save new hybrid to file.")
    clouds.save_cloud(cloudset.curr_hybrid, save_path, 'cloud_{}'.format(idx), simple=False)

    for i in range(len(cloudset)):
        if idx != cloudset.curr_hybrid_idx:
            idx = cloudset.curr_hybrid_idx
            print("Save new hybrid to file.")
            clouds.save_cloud(cloudset.curr_hybrid, save_path, 'cloud_{}'.format(idx), simple=False)

        sample, loc_bfs = cloudset[0]
        filename = save_path + 'h{}_s{}.pkl'.format(idx, i)
        with open(filename, 'wb') as f:
            pickle.dump([idx, loc_bfs, sample], f)


# TODO: Change this to classless methods which do very specific tasks
class Analyser:
    def __init__(self, data_path: str, cloudset: CloudSet):
        """ The analyser gets a dataset and a cloudset and is then applying the cloudset to that dataset by also
            performing some cloudset-independent analysis on the dataset.
        """

        self.data_path = data_path
        self.cloudset = cloudset
        self.files = glob.glob(data_path + '*.pkl')
        self.files.sort()

        # prepare header
        self.header = "Dataset analysis for: " + self.data_path + "\n"
        self.header += "Options: radius_nm = {}, radius_factor = {}, sample_num = {}\n"\
            .format(cloudset.radius_nm, cloudset.radius_factor, cloudset.sample_num)
        self.header += "Number of files: {}\n".format(len(self.files))
        self.header += "Label meaning: 0: Dendrite, 1: Axon, 2: Soma, 3: Bouton, 4: Terminal\n"

    def apply_cloudset(self, to_file: bool = False, to_png: bool = False, save_path: str = None):
        """ Iterates pickle files and applies self.cloudset to pointclouds in each file. For each pointcloud it
            iterates the cloudset and gathers information about the label distribution, the number of chunks and the
            size of the generated chunks. If to_file is set matplotlib histograms of the distribution for each
            pointcloud in total get saved.

        Args:
              to_file: Flag for saving analysis report and histograms to files.
              to_png: Flag for saving images of each pointcloud.
              save_path: path where files should be saved.
        """

        cloudset = self.cloudset
        total_output = self.header
        total_output += "\n### FILE OVERVIEW: ###\n"

        # prepare verbose option
        image_folder = ""
        if to_file:
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if to_png:
                    image_folder = save_path + "hybrid_images/"
                    if not os.path.exists(image_folder):
                        os.makedirs(image_folder)
            else:
                raise ValueError("Cannot print analysis to file if save_path is not set.")

        chunked_info = ""

        # iterate files
        for file in self.files:
            slashs = [pos for pos, char in enumerate(file) if char == '/']
            name = file[slashs[-1]+1:]
            print("Processing: " + name)

            # prepare iteration
            hybrid = clouds.load_cloud(file)
            traverser = hybrid.traverser(min_dist=cloudset.radius_nm * cloudset.radius_factor)
            chunk_num = len(traverser)

            self.cloudset.activate_single(hybrid)

            sample_num = self.cloudset.sample_num
            total_labels = np.zeros(sample_num*chunk_num)
            mean_size = 0

            # iterate chunks of current hybrid
            chunk_build = None
            for idx in tqdm(range(chunk_num)):
                chunk = cloudset[0]
                if chunk is None:
                    print("Finish at " + str(idx) + " of " + str(chunk_num))
                    break

                # estimate chunk size by simple bounding box
                vertices = chunk.vertices
                max_point = vertices.max(axis=0)
                min_point = vertices.min(axis=0)
                size = np.linalg.norm(max_point-min_point)
                mean_size += size

                total_labels[idx*sample_num:idx*sample_num+sample_num] = chunk.labels.reshape(len(chunk.labels))
                if chunk_build is None:
                    chunk_build = chunk
                else:
                    chunk_build = clouds.merge_clouds([chunk_build, chunk])
                idx += 1

            if to_png:
                visualize.visualize_clouds([chunk_build], capture=True,
                                           path=image_folder + '{}_chunked.png'.format(name))

            # evaluate information
            u_labels, counts = np.unique(total_labels, return_counts=True)
            mean_size = round(mean_size / chunk_num)

            # build hybrid information string
            hybrid_info = "\nFilename: {}\n" \
                          "Number of chunks: {}\n"\
                          "Mean size of chunks: {}\n"\
                          "Total number of labels: {}\n"\
                          "Labels and their counts:\n".format(name, chunk_num, mean_size, len(total_labels))
            for idx, el in enumerate(u_labels):
                percentage = round(counts[idx]/len(total_labels)*100, 1)
                hybrid_info += str(int(el)) + " -> count: " + str(counts[idx]) + " -> " + str(percentage) + "%\n"
            chunked_info += hybrid_info

            # create histogram plots
            if to_file:
                create_hist(total_labels, save_path + '{}_chunked_hist.png'.format(name[:-4]))

        total_output += chunked_info

        # output analysis
        if to_file:
            output = open(save_path + "chunking_analysis.txt", "w")
            output.write(total_output)
            output.close()
        else:
            print(total_output)

        return

    def get_overview(self, to_file: bool = False, to_png: bool = False, save_path: str = None):
        """ Iterates pickle files and extracts label distributions of pointclouds in each file. If to_file is set,
            these distributions get saved as matplotlib histogram

        Args:
              to_file: Flag for saving analysis report and histograms to files.
              to_png: Flag for saving images of each pointcloud.
              save_path: path where files should be saved.
        """

        cloudset = self.cloudset

        # prepare verbose option
        image_folder = ""
        if to_file:
            if save_path is not None:
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if to_png:
                    image_folder = save_path + 'hybrid_images/'
                    if not os.path.exists(image_folder):
                        os.makedirs(image_folder)
            else:
                raise ValueError("Cannot print analysis to file if save_path is not set.")

        # set up iteration parameters
        total_labels = np.array([])
        total_chunks = 0
        total_output = self.header
        total_output += "\n### FILE OVERVIEW: ###\n"
        hybrids_info = ""

        # iterate files
        for file in self.files:
            slashs = [pos for pos, char in enumerate(file) if char == '/']
            name = file[slashs[-1]:]
            print("Processing: " + name)

            # prepare hybrid
            hybrid = clouds.load_cloud(file)
            traverser = hybrid.traverser(min_dist=cloudset.radius_nm*cloudset.radius_factor)
            total_chunks += len(traverser)

            if to_png:
                visualize.visualize_clouds([hybrid], capture=True, path=image_folder + '{}.png'.format(name))

            # evaluate hybrid labels
            labels = hybrid.labels
            u_labels, counts = np.unique(labels, return_counts=True)
            if len(total_labels) == 0:
                total_labels = labels
            else:
                total_labels = np.concatenate((total_labels, labels), 0)

            # build hybrid information string
            hybrid_info = "\nFilename: {}\n" \
                          "Number of chunks: {}\n" \
                          "Total number of labels: {}\n"\
                          "Labels and their counts:\n".format(name, len(traverser), len(labels))
            for idx, el in enumerate(u_labels):
                percentage = round(counts[idx]/len(labels)*100, 1)
                hybrid_info += str(el) + " -> count: " + str(counts[idx]) + " -> " + str(percentage) + "%\n"
            hybrids_info += hybrid_info

            # create histogram plots
            if to_file:
                create_hist(labels, save_path + '{}_hist.png'.format(name[:-4]))

        # ----- GET TOTAL INFORMATION ----- #

        # build total information sring
        u_total, counts = np.unique(total_labels, return_counts=True)
        total_info = "\nTotal number of chunks: {}\n" \
                     "Total number of labels: {}\n" \
                     "Total labels and their counts:\n".format(total_chunks, len(total_labels))
        for idx, el in enumerate(u_total):
            percentage = round(counts[idx]/len(total_labels)*100, 1)
            total_info += str(el) + " -> count: " + str(counts[idx]) + " -> " + str(percentage) + "%\n"

        if to_file:
            create_hist(total_labels, save_path + 'total_hist.png')

        # build printable output
        total_output += total_info
        total_output += "\n### SINGLE FILE SPECS: ###\n"
        total_output += hybrids_info

        # output analysis
        if to_file:
            with open(save_path + "data_analysis.txt", "w") as output:
                output.write(total_output)
            output.close()
        else:
            print(total_output)
