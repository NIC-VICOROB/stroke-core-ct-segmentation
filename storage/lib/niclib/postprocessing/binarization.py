import copy
import itertools
import time
from threading import Thread

import numpy as np
from scipy import ndimage

from niclib.metrics import compute_segmentation_metrics, compute_avg_std_metrics_list

from niclib.inout.terminal import printProgressBar
import numpy as np
from scipy import ndimage
from abc import ABC, abstractmethod

class Binarizer(ABC):
    @abstractmethod
    def binarize(self, probs):
        pass

class ThreshSizeBinarizer(Binarizer):
    def __init__(self, thresh=0.5, min_lesion_vox=10):
        self.thresh = thresh
        self.min_lesion_vox = min_lesion_vox

    def binarize(self, probs):
        """
        Generates final class prediction by thresholding according to threshold and filtering by minimum lesion size
        """

        # Apply threshold
        y_prob = probs > self.thresh

        # Get connected components information
        y_prob_labelled, nlesions = ndimage.label(y_prob)
        if nlesions > 0:
            label_list = np.arange(1, nlesions + 1)
            lesion_volumes = ndimage.labeled_comprehension(y_prob, y_prob_labelled, label_list, np.sum, float, 0)

            # Set to 0 invalid lesions
            lesions_to_ignore = [idx + 1 for idx, lesion_vol in enumerate(lesion_volumes) if lesion_vol < self.min_lesion_vox]
            y_prob_labelled[np.isin(y_prob_labelled, lesions_to_ignore)] = 0

        # Generate binary mask and return
        y_binary = (y_prob_labelled > 0).astype('uint8')

        return y_binary

def thresh_size_search_single(result_set, images, thresholds, lesion_sizes, compute_lesion_metrics=False):
    true_vols, prob_vols = [], []
    for img in images:
        true_vols.append(img.labels[0])
        prob_vols.append(result_set[img.id] if img.id in result_set else None)

    # Generate result filename and try to load_samples results
    metrics_list = list()
    metrics_names = list()

    for n, (thresh, lesion_size) in enumerate(itertools.product(thresholds, lesion_sizes)):
        printProgressBar(n, len(thresholds)*len(lesion_sizes), suffix=" parameters evaluated")

        metrics_iter = list()
        for lesion_probs, true_vol in zip(prob_vols, true_vols):
            if lesion_probs is not None:
                rec_vol = ThreshSizeBinarizer(thresh, lesion_size).binarize(lesion_probs)
            else:
                continue

            metrics_iter.append(
                compute_segmentation_metrics(true_vol, rec_vol, lesion_metrics=compute_lesion_metrics))

        m_avg_std = compute_avg_std_metrics_list(metrics_iter)

        metrics_list.append(m_avg_std)
        metrics_names.append("th={}_ls={}".format(thresh, lesion_size))

    printProgressBar(len(thresholds)*len(lesion_sizes), len(thresholds)*len(lesion_sizes), suffix=" parameters evaluated")
    return metrics_list, metrics_names


def process_sample_metrics(true_vol, lesion_probs, thresh, lesion_size, compute_lesion_metrics, result_metrics, sample_idx):
    rec_vol = ThreshSizeBinarizer(thresh, lesion_size).binarize(lesion_probs)
    result_metrics[sample_idx] = compute_segmentation_metrics(true_vol, rec_vol, lesion_metrics=compute_lesion_metrics)


def thresh_size_search(result_set, images, thresholds, lesion_sizes, compute_lesion_metrics=False):
    # 6x faster than the inefficient one

    true_vols, prob_vols = [], []
    for img in images:
        true_vols.append(img.labels[0])
        prob_vols.append(result_set[img.id] if img.id in result_set else None)

    # Generate result filename and try to load_samples results
    metrics_list = list()
    metrics_names = list()
    for n, (thresh, lesion_size) in enumerate(itertools.product(thresholds, lesion_sizes)):
        printProgressBar(n, len(thresholds) * len(lesion_sizes), suffix=" parameters evaluated")

        threads = []
        metrics_iter = [None] * len(prob_vols)
        for sample_idx, (lesion_probs, true_vol) in enumerate(zip(prob_vols, true_vols)):
            if lesion_probs is None:
                continue

            process = Thread(target=process_sample_metrics, args=[true_vol, lesion_probs, thresh, lesion_size, compute_lesion_metrics, metrics_iter, sample_idx])
            process.start()
            threads.append(process)

        # Ensure every volume has been processed and remove none entries from results
        for process in threads:
            process.join()
        metrics_iter = [m for m in metrics_iter if m is not None] # in case incomplete prob set

        # Compute average for the specific thresh and lesion size and store
        m_avg_std = compute_avg_std_metrics_list(metrics_iter)

        metrics_list.append(m_avg_std)
        metrics_names.append("th={}_ls={}".format(thresh, lesion_size))

    printProgressBar(len(thresholds) * len(lesion_sizes), len(thresholds) * len(lesion_sizes),
                     suffix=" parameters evaluated")
    return metrics_list, metrics_names

def compute_set_metrics_dict(result_set_dict, images, thresh, lesion_size, compute_lesion_metrics=False):
    true_vols, prob_vols = [], []
    for img in images:
        true_vols.append(img.labels[0])
        prob_vols.append(result_set_dict[img.id] if img.id in result_set_dict else None)

    # Generate result filename and try to load_samples results
    threads = []
    metrics_iter = [None] * len(prob_vols)
    for sample_idx, (lesion_probs, true_vol) in enumerate(zip(prob_vols, true_vols)):
        if lesion_probs is None:
            continue

        process = Thread(target=process_sample_metrics,
                         args=[true_vol, lesion_probs, thresh, lesion_size, compute_lesion_metrics, metrics_iter,
                               sample_idx])
        process.start()
        threads.append(process)
    # Ensure every volume has been processed and remove none entries from results
    for process in threads:
        process.join()


    # Format resulting metrics with id
    metrics_dict = dict()
    for n, metrics in enumerate(metrics_iter):
        if metrics is None:
            continue

        assert images[n].id in result_set_dict
        metrics_dict.update({images[n].id: metrics})

    return metrics_dict