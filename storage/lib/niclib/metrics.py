import time

import numpy as np
from scipy import ndimage
from niclib.medpy_hausdorff import hd as haussdorf_dist

def compute_confusion_matrix(y_true, y_pred):
    """
    Returns tuple tp, tn, fp, fn
    """

    assert y_true.size == y_pred.size

    true_pos = np.sum(np.logical_and(y_true, y_pred))
    true_neg = np.sum(np.logical_and(y_true == 0, y_pred == 0))

    false_pos = np.sum(np.logical_and(y_true == 0, y_pred))
    false_neg = np.sum(np.logical_and(y_true, y_pred == 0))

    return true_pos, true_neg, false_pos, false_neg


def compute_lesion_confusion_matrix(y_true, y_pred):
    # True positives
    lesions_true, num_lesions_true = ndimage.label(y_true)
    lesions_pred, num_lesions_pred = ndimage.label(y_pred)

    true_pos = 0.0
    for i in range(num_lesions_true):
        lesion_detected = np.logical_and(y_pred, lesions_true == (i + 1)).any()
        if lesion_detected: true_pos += 1
    true_pos = np.min([true_pos, num_lesions_pred])

    # False positives
    tp_labels = np.unique(y_true * lesions_pred)
    fp_labels = np.unique(np.logical_not(y_true) * lesions_pred)

    # [label for label in fp_labels if label not in tp_labels]
    false_pos = 0.0
    for fp_label in fp_labels:
        if fp_label not in tp_labels: false_pos += 1

    return true_pos, false_pos, num_lesions_true, num_lesions_pred


def compute_segmentation_metrics(y_true, y_pred, lesion_metrics=False, exclude=None):
    metrics = {}
    eps = np.finfo(np.float32).eps

    tp, tn, fp, fn = compute_confusion_matrix(y_true, y_pred)

    # Sensitivity and specificity
    metrics['sens'] = tp / (tp + fn + eps) # Correct % of the real lesion
    metrics['spec'] = tn / (tn + fp + eps) # Correct % of the healthy area identified

    # Predictive value
    metrics['ppv'] = tp / (tp + fp + eps) # Of all lesion voxels, % of really lesion
    metrics['npv'] = tn / (tn + fn + eps)  # Of all lesion voxels, % of really lesion

    # Lesion metrics
    if lesion_metrics:
        tpl, fpl, num_lesions_true, num_lesions_pred = compute_lesion_confusion_matrix(y_true, y_pred)
        metrics['l_tpf'] = tpl / num_lesions_true if num_lesions_true > 0 else np.nan
        metrics['l_fpf'] = fpl / num_lesions_pred if num_lesions_pred > 0 else np.nan

        metrics['l_ppv'] = tpl / (tpl + fpl + eps)
        metrics['l_f1'] = (2.0 * metrics['l_ppv'] * metrics['l_tpf']) / (metrics['l_ppv'] + metrics['l_tpf'] + eps)

    # Dice coefficient
    metrics['dsc'] = dice_coef(y_true, y_pred)

    # Relative volume difference
    metrics['avd'] = 2.0 * np.abs(np.sum(y_pred) - np.sum(y_true))/(np.sum(y_pred) + np.sum(y_true) + eps)

    # Haussdorf distance
    try: metrics['hd'] = haussdorf_dist(y_pred, y_true, connectivity=3)  # Why connectivity 3?
    except Exception: metrics['hd'] = np.nan

    if exclude is not None:
        [metrics.pop(metric, None) for metric in exclude]

    return metrics


def dice_coef(y_true, y_pred, smooth = 0.01):
    intersection = np.sum(np.logical_and(y_true, y_pred))

    if intersection > 0:
        return (2.0 * intersection + smooth) / (np.sum(y_true) + np.sum(y_pred) + smooth)
    else:
        return 0.0

def compute_avg_std_metrics_list(metrics_list):
    metrics_avg_std = dict()

    assert len(metrics_list) > 0

    for metric_name in metrics_list[0].keys():
        metric_values = [metrics[metric_name] for metrics in metrics_list]

        metrics_avg_std.update({
            '{}_avg'.format(metric_name): np.nanmean(metric_values),
            '{}_std'.format(metric_name): np.nanstd(metric_values)})

    return  metrics_avg_std



def compute_clinical_metrics(y_true, y_pred):

    vol_true, vol_pred = np.sum(y_true), np.sum(y_pred)

    np.polyfit(x, y, deg)

    pass

















