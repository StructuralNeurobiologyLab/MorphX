# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

""" This script starts an analysis on the given dataset with a cloudset which produces chunks recording to the given
    parameters. The analysis results are saved in the subfolder 'analysis' within the dataset folder. """

import os
import argparse
from morphx.data.analyser import Analyser
from morphx.data.cloudset import CloudSet

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, help='Set data path.')
parser.add_argument('--ra', type=int, default=10000, help='Radius in nanometers.')
parser.add_argument('--sp', type=int, default=1000, help='Number of sample points.')
args = parser.parse_args()

data_path = os.path.expanduser(args.path)
save_path = os.path.expanduser(args.path + 'analysis/')

cloudset = CloudSet(data_path, args.ra, args.sp)

analyser = Analyser(data_path, cloudset)
analyser.apply_cloudset(to_file=True, save_path=save_path)
analyser.get_overview(to_file=True, save_path=save_path)
