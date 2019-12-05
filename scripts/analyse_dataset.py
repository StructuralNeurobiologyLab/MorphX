# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import argparse
from morphx.data.analyser import Analyser
from morphx.data.cloudset import CloudSet

parser = argparse.ArgumentParser(description='Analyse dataset.')
parser.add_argument('--path', type=str, help='Set data path')

args = parser.parse_args()

data_path = os.path.expanduser(args.path)
save_path = os.path.expanduser(args.path + 'analysis/')

radius_nm = 10000
sample_num = 800
cloudset = CloudSet(data_path, radius_nm, sample_num)

analyser = Analyser(data_path, cloudset)
analyser.apply_cloudset(to_file=True, save_path=save_path)
analyser.get_overview(to_file=True, save_path=save_path)
