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
from typing import Tuple

import morphx.processing.objects
from morphx.processing import clouds


def eval_dataset(input_path: str, gt_path: str, output_path: str, metrics: list, report_name: str = 'Evaluation',
                 total: bool = False, direct: bool = False, filters: bool = False, drop_unpreds: bool = True):
    """ Apply different metrics to HybridClouds with predictions and compare these predictions with corresponding
        ground truth files with different filters or under different conditions.

    Args:
        input_path: Location of HybridClouds with predictions, saved as pickle files by a MorphX prediction mapper.
        gt_path: Location of ground truth files, one for each file at input_path. Must have the same names as their
            counterparts.
        output_path: Location where results of evaluation should be saved.
        metrics: List of metrics which should be applied to the predicted clouds.
        report_name: Name of the current evaluation. Is used as the filename in which the evaluation report gets saved.
        total: Combine the predictions of all files to apply the metrics to the total prediction array.
        direct: Flag for swichting off the majority vote which is normally applied if there are multiple predictions
            for one vertex. If this flag is set, only the first prediction is taken into account.
        filters: After mapping the vertex labels to the skeleton, this flag can be used to apply filters to the skeleton
            and append the evaluation of these filtered skeletons.
        drop_unpreds: Flag for dropping all vertices or nodes which don't have predictions and whose labels are thus set
            to -1. If this flag is not set, the number of vertices or nodes without predictions might be much higher
            than the one of the predicted vertices or nodes, which results in bad evaluation results.
    """
    input_path = os.path.expanduser(input_path)
    gt_path = os.path.expanduser(gt_path)
    output_path = os.path.expanduser(output_path)

    files = glob.glob(input_path + '*.pkl')
    gt_files = glob.glob(gt_path + '*.pkl')
    if not os.path.isdir(output_path):
        os.makedirs(output_path)

    reports = []
    # Arrays for concatenating the labels of all files for later total evaluation
    total_labels = {'pred': np.array([]), 'pred_node': np.array([]), 'gt': np.array([]), 'gt_node': np.array([])}

    # Build single file evaluation reports
    reports.append('\n\n### Single file evaluation ###\n')
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

        report = '\n\n### File: ' + name + ' ###\n'
        report += eval_single(file, gt_file, metrics, total_labels, direct=direct, filters=filters,
                              drop_unpreds=drop_unpreds)
        reports.append(report)

    # Perform evaluation on total label arrays (labels from all files sticked together), prediction
    # mappings or filters are already included
    report = '\n\n### Evaluation of files at: ' + input_path + ' ###\n\n'
    if total:
        report += '\n\n### Total evaluation ###\n'
        for idx, metric in enumerate(metrics):
            report += '\n\n### Metric ' + str(idx) + ' ###\n\n'
            if direct:
                mode = 'direct'
            else:
                mode = 'majority vote'
            report += '\nVertex evaluation ' + mode + ':\n\n'
            report += metric(total_labels['gt'], total_labels['pred'])
            if filters:
                mode += ' with filters'
            report += '\n\nSkeleton evaluation ' + mode + ':\n\n'
            report += metric(total_labels['gt_node'], total_labels['pred_node'])

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
    """ Apply different metrics to HybridClouds with predictions and compare these predictions with corresponding
        ground truth files with different filters or under different conditions.

    Args:
        file: HybridCloud with predictions, saved as pickle file by a MorphX prediction mapper.
        gt_file: Ground truth file corresponding to the HybridCloud given in file.
        metrics: List of metrics which should be applied to the predicted clouds.
        total: Use given dict to save processed predictions for later use (see eval_dataset).
        direct: Flag for swichting off the majority vote which is normally applied if there are multiple predictions
            for one vertex. If this flag is set, only the first prediction is taken into account.
        filters: After mapping the vertex labels to the skeleton, this flag can be used to apply filters to the skeleton
            and append the evaluation of these filtered skeletons.
        drop_unpreds: Flag for dropping all vertices or nodes which don't have predictions and whose labels are thus set
            to -1. If this flag is not set, the number of vertices or nodes without predictions might be much higher
            than the one of the predicted vertices or nodes, which results in bad evaluation results.

    Returns:
        Evaluation report as string.
    """
    file = os.path.expanduser(file)
    gt_file = os.path.expanduser(gt_file)

    # load HybridCloud and corresponding ground truth
    hc = morphx.processing.objects.load_pkl(file)
    gt_hc = morphx.processing.objects.load_pkl(gt_file)
    if len(hc.labels) != len(gt_hc.labels):
        raise ValueError("Length of ground truth label array doesn't match with length of predicted label array.")

    report = ''

    for idx, metric in enumerate(metrics):
        report += '\n\n### Metric ' + str(idx) + ' ###\n\n'

        # Perform majority vote on existing predictions and set these as new labels
        if direct:
            hc.preds2labels(False)
            mode = 'direct'
        else:
            hc.preds2labels()
            mode = 'majority vote'

        # Get evaluation for vertices
        gtl, hcl = handle_unpreds(gt_hc.labels, hc.labels, drop_unpreds)
        report += '\nVertex evaluation ' + mode + ':\n\n'
        report += metric(gtl, hcl)

        # Get evaluation for skeletons
        if filters:
            hc.clean_node_labels()
            mode += ' with filters'
        gtnl, hcnl = handle_unpreds(gt_hc.node_labels, hc.node_labels, drop_unpreds)

        report += '\n\nSkeleton evaluation ' + mode + ':\n\n'
        report += metric(gtnl, hcnl)

        if total is not None:
            total['pred'] = np.append(total['pred'], hcl)
            total['pred_node'] = np.append(total['pred_node'], hcnl)
            total['gt'] = np.append(total['gt'], gtl)
            total['gt_node'] = np.append(total['gt_node'], gtnl)

    return report


def handle_unpreds(gt: np.ndarray, hc: np.ndarray, drop: bool) -> Tuple[np.ndarray, np.ndarray]:
    if drop:
        mask = np.logical_and(hc != -1, gt != -1)
        return gt[mask], hc[mask]
    else:
        return gt, hc


if __name__ == '__main__':
    # print(eval_single('~/thesis/gt/gt_poisson/ads/predictions/25000_5000/sso_24414208_c.pkl',
    #                   '~/thesis/gt/gt_poisson/ads/sso_24414208_c.pkl', [sm.classification_report], filters=True,
    #                   direct=True))

    for radius in [25000]:
        for npoints in [5000]:
            eval_dataset(f'~/thesis/trainings/past/2020/01_14/2020_01_14_{radius}_{npoints}/predictions/',
                         '~/thesis/gt/gt_poisson/ads/',
                         f'~/thesis/trainings/past/2020/01_14/2020_01_14_{radius}_{npoints}/predictions/',
                         [sm.classification_report], total=True, report_name='Evaluation_new')
