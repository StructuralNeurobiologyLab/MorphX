# -*- coding: utf-8 -*-
# MorphX - Toolkit for morphology exploration and segmentation
#
# Copyright (c) 2020 - now
# Max Planck Institute of Neurobiology, Martinsried, Germany
# Authors: Jonathan Klimesch

import ipdb
import os
import glob
import numpy as np
import sklearn.metrics as sm
from tqdm import tqdm
from morphx.processing import clouds


def eval_dataset(input_path: str, gt_path: str, output_path: str, metrics: list, report_name: str = 'Evaluation',
                 total: bool = False, direct: bool = False, filters: bool = False, drop_unpreds: bool = True):
    input_path = os.path.expanduser(input_path)
    gt_path = os.path.expanduser(gt_path)
    output_path = os.path.expanduser(output_path)

    files = glob.glob(input_path + '*.pkl')
    gt_files = glob.glob(gt_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    reports = []
    total_labels = {'pred': np.array([]), 'pred_node': np.array([]), 'gt': np.array([]), 'gt_node': np.array([]),
                    'pred_node_f': np.array([]), 'gt_node_f': np.array([]), 'pred_d': np.array([]),
                    'gt_d': np.array([]), 'pred_node_d': np.array([]), 'gt_node_d': np.array([]),
                    'pred_node_d_f': np.array([]), 'gt_node_d_f': np.array([])}

    for file in tqdm(files):
        slashs = [pos for pos, char in enumerate(file) if char == '/']
        name = file[slashs[-1] + 1:-4]

        # Find corresponding ground truth
        gt_file = None
        for item in gt_files:
            if name in item:
                gt_file = item
        if gt_file is None:
            print("Ground truth for {} was not found. Skipping file.".format(name))
            continue

        report = '\n\n### File: ' + name + ' ###\n\n'
        report += eval_single(file, gt_file, metrics, total_labels, direct=direct, filters=filters,
                              drop_unpreds=drop_unpreds)
        reports.append(report)

    # Perform evaluation on total label arrays (labels from all files sticked together)
    report = '\n\n### Evaluation of files at: ' + input_path + ' ###\n\n'
    if total:
        report += '\n\n### Total evaluation ###\n\n'
        for idx, metric in enumerate(metrics):
            report += '\n\n### Metric ' + str(idx) + ' ###\n\n'
            report += '\nVertex evaluation:\n\n'
            report += metric(total_labels['gt'], total_labels['pred'])
            report += '\n\nSkeleton evaluation:\n\n'
            report += metric(total_labels['gt_node'], total_labels['pred_node'])

            if filters:
                report += '\n\nSkeleton evaluation with filters:\n\n'
                report += metric(total_labels['gt_node_f'], total_labels['pred_node_f'])

            if direct:
                report += '\n\nVertex evaluation without majority vote:\n\n'
                report += metric(total_labels['gt_d'], total_labels['pred_d'])
                report += '\n\nSkeleton evaluation without majority vote:\n\n'
                report += metric(total_labels['gt_node_d'], total_labels['pred_node_d'])

                if filters:
                    report += '\n\nSkeleton evaluation without majority vote:\n\n'
                    report += metric(total_labels['gt_node_d_f'], total_labels['pred_node_d_f'])

    # Write reports to file
    file = output_path + report_name + '.txt'
    total_report = report
    for item in reports:
        total_report += item

    with open(file, 'w') as f:
        f.write(total_report)
    f.close()


def eval_single(file: str, gt_file: str, metrics: list, total: dict = None, direct: bool = False,
                filters: bool = False, drop_unpreds: bool = True) -> str:
    file = os.path.expanduser(file)
    gt_file = os.path.expanduser(gt_file)

    # load HybridCloud and corresponding ground truth
    hc = clouds.load_cloud(file)
    gt_hc = clouds.load_cloud(gt_file)
    if len(hc.labels) != len(gt_hc.labels):
        raise ValueError("Length of ground truth label array doesn't match with length of predicted label array.")

    report = ''

    for idx, metric in enumerate(metrics):
        report += '\n\n### Metric ' + str(idx) + ' ###\n\n'

        # Perform majority vote on existing predictions and set these as new labels
        hc.preds2labels_mv()

        hc_labels = hc.labels
        gt_labels = gt_hc.labels
        if drop_unpreds:
            mask = np.logical_and(hc.labels != -1, gt_hc.labels != -1)
            hc_labels = hc_labels[mask]
            gt_labels = gt_labels[mask]

        # Get evaluation for vertices
        report += '\nVertex evaluation:\n\n'
        report += metric(gt_labels, hc_labels)

        hc_node_labels = hc.node_labels
        gt_node_labels = gt_hc.node_labels
        if drop_unpreds:
            mask = np.logical_and(hc.node_labels != -1, gt_hc.node_labels != -1)
            hc_node_labels = hc_node_labels[mask]
            gt_node_labels = gt_node_labels[mask]

        # Get evaluation for skeletons
        report += '\n\nSkeleton evaluation:\n\n'
        report += metric(gt_node_labels, hc_node_labels)

        if total is not None:
            total['pred'] = np.append(total['pred'], hc_labels)
            total['gt'] = np.append(total['gt'], gt_labels)
            total['pred_node'] = np.append(total['pred_node'], hc_node_labels)
            total['gt_node'] = np.append(total['gt_node'], gt_node_labels)

        if filters:
            # Apply filters to nodes
            hc.clean_node_labels()

            hc_node_labels = hc.node_labels
            gt_node_labels = gt_hc.node_labels
            if drop_unpreds:
                mask = np.logical_and(hc.node_labels != -1, gt_hc.node_labels != -1)
                hc_node_labels = hc_node_labels[mask]
                gt_node_labels = gt_node_labels[mask]

            report += '\n\nSkeleton evaluation with filters:\n\n'
            report += metric(gt_node_labels, hc_node_labels)
            if total is not None:
                total['pred_node_f'] = np.append(total['pred_node_f'], hc_node_labels)
                total['gt_node_f'] = np.append(total['gt_node_f'], gt_node_labels)

        if direct:
            # Map predictions without majority vote and produce same reports as above
            hc.preds2labels_direct()

            hc_labels = hc.labels
            gt_labels = gt_hc.labels
            if drop_unpreds:
                mask = np.logical_and(hc.labels != -1, gt_hc.labels != -1)
                hc_labels = hc_labels[mask]
                gt_labels = gt_labels[mask]

            report += '\n\nVertex evaluation without majority vote:\n\n'
            report += metric(gt_labels, hc_labels)

            hc_node_labels = hc.node_labels
            gt_node_labels = gt_hc.node_labels
            if drop_unpreds:
                mask = np.logical_and(hc.node_labels != -1, gt_hc.node_labels != -1)
                hc_node_labels = hc_node_labels[mask]
                gt_node_labels = gt_node_labels[mask]

            report += '\n\nSkeleton evaluation without majority vote:\n\n'
            report += metric(gt_node_labels, hc_node_labels)
            if total is not None:
                total['pred_d'] = np.append(total['pred_d'], hc_labels)
                total['pred_node_d'] = np.append(total['pred_node_d'], hc_node_labels)
                total['gt_d'] = np.append(total['gt_d'], gt_labels)
                total['gt_node_d'] = np.append(total['gt_node_d'], gt_node_labels)

            if filters:
                hc.clean_node_labels()

                hc_node_labels = hc.node_labels
                gt_node_labels = gt_hc.node_labels
                if drop_unpreds:
                    mask = np.logical_and(hc.node_labels != -1, gt_hc.node_labels != -1)
                    hc_node_labels = hc_node_labels[mask]
                    gt_node_labels = gt_node_labels[mask]

                report += '\n\nSkeleton evaluation without majority vote with filters:\n\n'
                report += metric(gt_node_labels, hc_node_labels)
                if total is not None:
                    total['pred_node_d_f'] = np.append(total['pred_node_d_f'], hc_node_labels)
                    total['gt_node_d_f'] = np.append(total['gt_node_d_f'], gt_node_labels)

    return report


if __name__ == '__main__':
    # print(eval_single('~/thesis/gt/gt_poisson/ads/predictions/25000_5000/sso_24414208_c.pkl',
    #                   '~/thesis/gt/gt_poisson/ads/sso_24414208_c.pkl', [sm.classification_report], filters=True,
    #                   direct=True))

    eval_dataset('~/thesis/gt/gt_poisson/ads/predictions/25000_5000/', '~/thesis/gt/gt_poisson/ads/',
                 '~/thesis/gt/gt_poisson/ads/predictions/25000_5000/', [sm.classification_report], total=True)
