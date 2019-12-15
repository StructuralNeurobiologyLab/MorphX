# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

""" This script is for qualitative comparison of the ground truth dataset with another processed dataset. Both datasets
    must contain pickle files with the same names, the ground truth files must comply with clouds.load_gt, the
    processed files must comply with clouds.load_cloud. """

import os
import glob
import argparse
from morphx.processing import clouds, visualize

parser = argparse.ArgumentParser(description='Validate a network.')

parser.add_argument('--gt', type=str, required=True, help='Path to ground truth folder.')
parser.add_argument('--pr', type=str, required=True, help='Path to folder with processed files.')

args = parser.parse_args()

target_path = os.path.expanduser(args.ta)
pred_path = os.path.expanduser(args.pr)

target_files = glob.glob(target_path + '*.pkl')
pred_files = glob.glob(pred_path + '*.pkl')

target_files.sort()
pred_files.sort()

for idx, target_file in enumerate(target_files):
    pred_file = pred_files[idx]
    target = clouds.load_gt(target_file)
    pred = clouds.load_cloud(pred_file)

    visualize.visualize_parallel([target], [pred])
