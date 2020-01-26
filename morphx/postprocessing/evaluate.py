# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import os
import glob
import numpy as np
import sklearn.metrics as sm
from tqdm import tqdm
from morphx.processing import clouds


def eval_dataset(input_path: str, gt_path: str, output_path: str, metrics: list,
                 total: bool = False, direct: bool = False, filters: bool = False):
    files = glob.glob(input_path + '*.pkl')
    gt_files = glob.glob(gt_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    reports = []
    total_pred_labels = np.array([])
    total_pred_node_labels = np.array([])
    total_gt_labels = np.array([])
    total_gt_node_labels = np.array([])

    total_pred_node_labels_f = np.array([])

    total_pred_labels_d = np.array([])
    total_pred_node_labels_d = np.array([])

    total_pred_node_labels_d_f = np.array([])

    for file in tqdm(files):
        # load HybridCloud and corresponding ground truth
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1]+1:-4]
        hc = clouds.load_cloud(file)
        gt_file = None
        for item in gt_files:
            if name in item:
                gt_file = item
        if gt_file is None:
            print("Ground truth for {} was not found. Skipping file.".format(name))
            continue
        gt_hc = clouds.load_cloud(gt_file)
        if len(hc.labels) != len(gt_hc.labels):
            raise ValueError("Length of ground truth label array doesn't match with length of predicted label array.")

        # Perform majority vote on existing predictions and set these as new labels
        hc.preds2labels_mv()
        # if hc.encoding is not None:
        #     target_names = [pair[0] for pair in sorted(hc.encoding.items(), key=lambda pair: pair[1])]
        # else:
        #     target_names = None

        for metric in metrics:
            # Get evaluation for vertices
            reports.append(metric(gt_hc.labels, hc.labels))
            # Get evaluation for skeletons
            reports.append(metric(gt_hc.node_labels, hc.node_labels))

            if total:
                total_pred_labels = np.append(total_pred_labels, hc.labels)
                total_gt_labels = np.append(total_gt_labels, gt_hc.labels)
                total_pred_node_labels = np.append(total_pred_node_labels, hc.node_labels)
                total_gt_node_labels = np.append(total_gt_node_labels, gt_hc.node_labels)

            if filters:
                # Apply filters to nodes
                hc.clean_node_labels()
                reports.append(metric(gt_hc.node_labels, hc.node_labels))
                total_pred_node_labels_f = np.append(total_pred_node_labels_f, hc.node_labels)

            if direct:
                # Map predictions without majority vote and produce same reports as above
                hc.preds2labels_direct()
                reports.append(metric(gt_hc.labels, hc.labels))
                reports.append(metric(gt_hc.node_labels, hc.node_labels))
                total_pred_labels_d = np.append(total_pred_labels_d, hc.labels)
                total_pred_node_labels_d = np.append(total_pred_node_labels_d, hc.node_labels)

                if filters:
                    hc.clean_node_labels()
                    reports.append(metric(gt_hc.node_labels, hc.node_labels))
                    total_pred_node_labels_d_f = np.append(total_pred_node_labels_d_f, hc.node_labels)

    # Perform evaluation on total label arrays (labels from all files sticked together)
    if total:
        for metric in metrics:
            reports.append(metric(total_gt_labels, total_pred_labels))
            reports.append(metric(total_gt_node_labels, total_pred_node_labels))

            if filters:
                reports.append(metric(total_gt_node_labels, total_pred_node_labels_f))

            if direct:
                reports.append(metric(total_gt_labels, total_pred_labels_d))
                reports.append(metric(total_gt_node_labels, total_pred_node_labels_d))

                if filters:
                    reports.append(metric(total_gt_node_labels, total_pred_node_labels_d_f))
