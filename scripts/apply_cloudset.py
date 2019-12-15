# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

""" This script applies a cloudset to the given dataset and chunks it with the given parameter while saving images of
    all chunks to the given save path. Images of the total result and the original data will also be saved. """

import os
import glob
import re
import argparse
from morphx.processing import clouds, visualize
from morphx.data.cloudset import CloudSet

parser = argparse.ArgumentParser()
parser.add_argument('--da', type=str, help='Set data path.')
parser.add_argument('--sa', type=str, help='Set save path.')
parser.add_argument('--ra', type=int, default=10000, help='Radius in nanometers.')
parser.add_argument('--sp', type=int, default=1000, help='Number of sample points.')
args = parser.parse_args()

data_path = os.path.expanduser(args.da)
save_path = os.path.expanduser(args.sa)
files = glob.glob(data_path + '*.pkl')

radius = args.ra
sample_num = args.sp

for file in files:
    hc = clouds.load_gt(file)

    data = CloudSet(data_path, radius, sample_num)
    data.activate_single(hc)

    regex = re.findall(r"_(\d+).", file)
    sso_id = int(regex[0])

    samples = []
    for i in range(data.size):
        sample = data[0]
        samples.append(sample)
        visualize.visualize_clouds([sample], capture=True, path=save_path + 'sso_{}_r{}_s{}_i{}.png'.format(sso_id, radius, sample_num, i))

    visualize.visualize_clouds(samples, capture=True, path=save_path + 'full_sso_{}_chunked.png'.format(sso_id))
    visualize.visualize_clouds([hc], capture=True, path=save_path + 'full_sso_{}.png'.format(sso_id))
