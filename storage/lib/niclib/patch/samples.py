import copy
import numpy as np
import logging as log
import itertools as iter

from niclib.dataset import NIC_Image
from niclib.patch.volume import pad_volume


def split_train_val(dataset, validation_split=None, idxs=None):
    assert validation_split is not None or idxs is not None  # One is not None
    assert validation_split is None or idxs is None  # Other is None
    if validation_split is not None:
        assert isinstance(validation_split, float) and 0.0 < validation_split < 1.0
    if idxs is not None:
        assert isinstance(idxs, list)

    if len(dataset.val) > 0:
        log.debug("Existing validation set found -> num_train={}, num_val={}" \
                  .format(len(dataset.train), len(dataset.val)))
        return dataset

    # Last volumes will be relocated to val set
    N = len(dataset.train)
    if validation_split is not None:
        num_val_volumes = np.int(np.ceil(N * validation_split))
        val_idxs = np.asarray(range(N - num_val_volumes, N), dtype=int)
    else:  # idxs is not None
        val_idxs = np.asarray(idxs, dtype=int)
        val_idxs = val_idxs[val_idxs < len(dataset.train)]

    train_idxs = np.delete(np.arange(N, dtype=int), val_idxs)

    dataset.val = [dataset.train[idx] for idx in val_idxs]
    dataset.train = [dataset.train[idx] for idx in train_idxs]

    crossval_id = '{}_to_{}'.format(val_idxs[0], val_idxs[-1])

    disp_msg = "Splitted train into train+val -> num_train={}, num_val={}, num_test={}, crossval_id={}"
    log.debug(disp_msg.format(len(dataset.train), len(dataset.val), len(dataset.test), crossval_id))

    return dataset, crossval_id


def compute_set_statistics(set_in):
    num_samples = len(set_in)
    num_modalities = set_in[0].data.shape[0]

    mean = np.zeros((num_samples, num_modalities,))
    std = np.zeros((num_samples, num_modalities,))

    for idx in range(len(set_in)):
        mean[idx], std[idx] = compute_sample_statistics(set_in[idx])

    return mean, std


def normalise_set(set_in, mean=None, std=None):
    return_stats = False
    if mean is None and std is None:
        mean, std = compute_set_statistics(set_in)
        return_stats = True

    for i in range(len(set_in)):
        set_in[i] = normalise_sample(set_in[i], mean=mean[i], std=std[i])

    return set_in if not return_stats else set_in, mean, std


def compute_sample_statistics(sample_in):
    assert isinstance(sample_in, NIC_Image)
    num_modalities = sample_in.data.shape[0]

    mean = np.zeros((num_modalities,))
    std = np.zeros((num_modalities,))
    for modality in range(num_modalities):
        mean[modality] = np.mean(sample_in.data[modality])
        std[modality] = np.std(sample_in.data[modality])

    return mean, std


def normalise_sample(sample_in, mean=None, std=None):
    if mean is None and std is None:
        mean, std = compute_sample_statistics(sample_in)

    for modality in range(sample_in.data.shape[0]):
        sample_in.data[modality] -= mean[modality]
        sample_in.data[modality] /= std[modality]

    return sample_in


