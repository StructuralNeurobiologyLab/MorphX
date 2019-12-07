# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2019 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import sys
import glob
import pickle
import random
import argparse
import morphx.processing.visualize as visualize
from getkey import keys

parser = argparse.ArgumentParser(description='Evaluate validation set.')
parser.add_argument('--na', type=str, required=True, help='Experiment name')
parser.add_argument('--sr', type=str, default='~/gt/simple_training/', help='Save root')
parser.add_argument('--ip', type=str, required=True, help='Inspection path')
parser.add_argument('--ra', action='store_true', help='View in random order')
parser.add_argument('--et', type=int, default=0, help='Entry point for viewing. Gets ignored when --ra flag is set.')

args = parser.parse_args()

base_path = os.path.expanduser(args.sr) + args.na + '/'

data_path = base_path + args.ip
save_path = base_path + 'im_examples/'
if not os.path.exists(save_path):
    os.makedirs(save_path)

print(data_path)
files = glob.glob(data_path + '*.pkl')
files.sort()

if args.ra:
    print("Shuffle files")
    random.shuffle(files)

idx = args.et
reverse = False
while idx < len(files):
    file = files[idx]
    slashs = [pos for pos, char in enumerate(file) if char == '/']
    filename = file[slashs[-1]:-4]
    print("Viewing: " + filename)

    with open(file, 'rb') as f:
        results = pickle.load(f)
    if reverse:
        i = int(len(results))-2
    else:
        i = 0
    while i < int(len(results)):
        orig = results[i]
        pred = results[i+1]
        key = visualize.visualize_parallel([orig], [pred],
                                           name1=filename+'_i{}_orig'.format(i),
                                           name2=filename+'_i{}_pred'.format(i+1),
                                           static=True)
        if key == keys.RIGHT:
            reverse = False
            i += 2
        if key == keys.UP:
            print("Saving to png...")
            path = save_path + '{}_i{}'.format(filename, i)
            visualize.visualize_single([orig], capture=True, path=path + '_1.png')
            visualize.visualize_single([pred], capture=True, path=path + '_2.png')
        if key == keys.DOWN:
            print("Displaying interactive view...")
            # display images for interaction
            visualize.visualize_parallel([orig], [pred])
        if key == keys.LEFT:
            reverse = True
            i -= 2
            if i < 0:
                break
        if key == keys.ENTER:
            sys.exit()

    if reverse:
        idx -= 1
    else:
        idx += 1
