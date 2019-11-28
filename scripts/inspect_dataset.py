# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

""" Move pickle files (with *_[sso_id]_* as name) to data_path. This script will then save images of the chunks
    of the object in the first pickle file """

import os
import glob
import re
from morphx.processing import clouds, visualize
from morphx.data.cloudset import CloudSet

data_path = os.path.expanduser('~/')
files = glob.glob(data_path + '*.pkl')

original = clouds.load_gt(files[0])

transform = clouds.Compose([clouds.Center(), clouds.RandomRotate(), clouds.RandomVariation(limits=(-10, 10))])
radius = 10000
sample_num = 800
data = CloudSet(data_path, radius, sample_num, transform=transform)

size = len(data.curr_hybrid.traverser())
regex = re.findall(r"_(\d+).", files[0])
sso_id = int(regex[0])

samples = []
for i in range(size):
    sample = data[0]
    samples.append(sample)
    visualize.visualize_single([sample], capture=True, path=data_path+'visualized_chunks/sso_{}_r{}_s{}_i{}.png'
                               .format(sso_id, radius, sample_num, i))

visualize.visualize_single([original], capture=True, path=data_path+'visualized_chunks/full_sso_{}.png'.format(sso_id))
