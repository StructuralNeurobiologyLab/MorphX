# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import sys
import glob
import pickle
from getkey import keys
from morphx.processing import visualize, clouds
from morphx.classes.pointcloud import PointCloud


class ViewControl(object):
    """ Viewer class for comparison of ground truth with processed files or for viewing validation and training
        examples. """

    def __init__(self, path1: str, save_path: str, path2: str = None):
        """
        Args:
            path1: path to pickle files (if path2 != None and comparison of ground truth is intended, this must be the
                folder to the ground truth files).
            path2: Set if comparison of ground truth is intended. This should point to the directory of processed files.
        """

        self.path1 = os.path.expanduser(path1)
        self.files1 = glob.glob(path1 + '*.pkl')
        self.files1.sort()

        self.save_path = os.path.expanduser(save_path)

        self.path2 = path2
        self.files2 = None
        self.cmp = False
        if path2 is not None:
            self.path2 = os.path.expanduser(path2)
            self.files2 = glob.glob(path2 + '*.pkl')
            self.files2.sort()
            self.cmp = True

        if self.cmp:
            self.load = self.load_cmp
        else:
            self.load = self.load_val

    def start_view(self):
        self.load()

    def core_next(self, cloud1: PointCloud, cloud2: PointCloud, save_name):
        key = visualize.visualize_parallel([cloud1], [cloud2], static=True)
        if key == keys.RIGHT:
            return 1
        if key == keys.UP:
            print("Saving to png...")
            path = self.save_path + save_name
            visualize.visualize_clouds([cloud1], capture=True, path=path + '_1.png')
            visualize.visualize_clouds([cloud2], capture=True, path=path + '_2.png')
        if key == keys.DOWN:
            print("Displaying interactive view...")
            # display images for interaction
            visualize.visualize_parallel([cloud1], [cloud2])
        if key == keys.LEFT:
            return -1
        if key == keys.ENTER:
            print("Aborting inspection...")
            sys.exit()

    def load_val(self):
        idx = 0
        reverse = False

        # TODO: Make this pythonic and simpler
        while idx < len(self.files1):
            file = self.files1[idx]
            slashs = [pos for pos, char in enumerate(file) if char == '/']
            filename = file[slashs[-1]:-4]
            print("Viewing: " + filename)

            with open(file, 'rb') as f:
                results = pickle.load(f)
            if reverse:
                i = int(len(results)) - 2
            else:
                i = 0
            while i < int(len(results)):
                res = self.core_next(results[i], results[i + 1], filename + '_i{}'.format(i))
                i += 2*res
                if res < 0:
                    reverse = True
                    if i < 0:
                        break
                else:
                    reverse = False
            if reverse:
                idx -= 1
            else:
                idx += 1

    def load_cmp(self):
        idx = 0
        while idx < len(self.files1):
            gt_file = self.files1[idx]
            pred_file = self.files2[idx]

            slashs = [pos for pos, char in enumerate(gt_file) if char == '/']
            filename = gt_file[slashs[-1]+1:-4]

            gt = clouds.load_gt(gt_file)
            pred = clouds.load_cloud(pred_file)

            idx += self.core_next(gt, pred, filename)
