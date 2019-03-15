import logging as log
import numpy as np
import math
import itertools as iter
import scipy.ndimage as ndimage
import copy
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import sys

from niclib.utils import get_resampling_indexes


def extract_patch_with_slice(volume, patch_slice):
    assert volume.ndim == len(patch_slice)

    patch_out = copy.deepcopy(volume[patch_slice])
    if patch_out.shape[-1] == 1: # 2D/3D compatibility
        patch_out = patch_out.squeeze(axis=-1)

    return patch_out

def extract_patches_at_locations(volume, patch_slices):
    # Infer patch shape from the patch_slices
    patch_shape = volume[patch_slices[0]].shape

    patches = np.zeros((len(patch_slices),) + patch_shape)
    for i, patch_slice in enumerate(patch_slices):
        patches[i] = copy.copy(volume[patch_slice])

    # If last dimension is 1 means 2D case (remove dimension)
    if patches.shape[-1] == 1:
        patches = patches.squeeze(axis=-1)

    return patches


def zeropad_patches(patches, desired_shape):
    """
    Zeropad patches to match desired_shape

    :patches: as (#, x, y, z, class)
    :desired_shape: 3D tuple desired x,y,z
    """

    #log.debug("Zeropad patches: patches shape {}, desired_shape {}".format(patches.shape, desired_shape))

    assert patches.ndim == 5
    patch_shape = patches.shape[1:-1]  # Extract only x,y,z

    pad = []
    for dim in range(len(desired_shape)):
        pad.append([int(math.ceil((desired_shape[dim] - patch_shape[dim]) / 2.0)),
                    int(math.floor((desired_shape[dim] - patch_shape[dim]) / 2.0))])

    # Check if we need to do padding >.<
    if np.sum(pad) > 0:
        # Pre-append 0,0 to not pad batch dimension and postappend 0,0 to not pad class dimension (only x,y,z)
        padding = ((0, 0), (pad[0][0], pad[0][1]), (pad[1][0], pad[1][1]), (pad[2][0], pad[2][1]), (0, 0))
        patches = np.pad(patches, padding, 'constant', constant_values=0)

    #log.debug("out: patches shape {}".format(patches.shape))

    return patches

def augment_patches(patch_sets_in, goal_num_patches):
    patch_set_ref = patch_sets_in[0] # Use first patch set for meta-computations
    if goal_num_patches <= patch_set_ref.shape[0]:
        return patch_sets_in

    num_patches_in = len(patch_set_ref)
    num_augments_per_patch = int(math.ceil(np.minimum(goal_num_patches / num_patches_in, 5)))
    num_patches_augment = int(goal_num_patches - num_patches_in)

    # Allocate space for output array and copy non augmented patches
    num_augmented_so_far = 0
    patch_sets_augmented = []
    for patch_set in patch_sets_in:
        patch_set_aug = np.zeros((int(goal_num_patches),) + patch_set.shape[1:])
        patch_set_aug[:num_patches_in] = patch_set
        patch_sets_augmented += [patch_set_aug, ]
    num_augmented_so_far += num_patches_in

    # Augment and add remaining copies
    augment_funcs = get_augment_functions(x_axis=1, y_axis=2)  # (nmods, x, y, z)
    sampling_idxs = get_resampling_indexes(num_patches_in, num_patches_augment // num_augments_per_patch)
    func_idxs = get_resampling_indexes(len(augment_funcs), num_augments_per_patch)
    for idx in sampling_idxs:
        for func_idx in func_idxs:
            augment_func = augment_funcs[func_idx]
            for set_idx in range(len(patch_sets_augmented)):
                patch_sets_augmented[set_idx][num_augmented_so_far] = augment_func(patch_sets_in[set_idx][idx])
            num_augmented_so_far += 1

    # log.debug("AUGMENTED Data patches: {}, Label patches: {}".format(patch_set_reference.shape, label_patches.shape))
    for set_idx in range(len(patch_sets_augmented)):
        patch_sets_augmented[set_idx] = patch_sets_augmented[set_idx][:num_augmented_so_far]

    patch_sets_augmented = tuple(patch_sets_augmented)
    return patch_sets_augmented



def normalise_patches(patch_sets, mean, std):
    unpack_on_return = False
    if type(patch_sets) is not tuple:
        patch_sets = (patch_sets,)
        unpack_on_return = True

    assert patch_sets[0].shape[1] == len(mean) == len(std)

    for i, patch_set in enumerate(patch_sets):
        for modality in range(patch_set.shape[1]):
            patch_sets[i][modality] -= mean[modality]
            patch_sets[i][modality] /= std[modality]

    return patch_sets if not unpack_on_return else patch_sets[0]


def normalise_patch(patch, mean, std):
    assert patch.shape[0] == len(mean) == len(std)
    for modality in range(patch.shape[0]):
        patch[modality] -= mean[modality]
        patch[modality] /= std[modality]
    return patch


def elastic_transform_patches(data_patch, label_patch,  x_axis, y_axis, alpha_factor=2.5, sigma_factor=0.12, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """

    random_state = np.random.RandomState(None)
    shape = data_patch.shape[1:]

    # Elastic deformation
    alpha = np.mean(shape)*alpha_factor
    sigma = np.mean(shape)*sigma_factor

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))

    data_patch_out = np.copy(data_patch)
    for idx, modality in enumerate(data_patch):
        data_patch_out[idx] = map_coordinates(data_patch[idx], indices, order=1, mode='reflect').reshape(shape)

    if label_patch is not None:
        label_patch_out = np.copy(label_patch)
        for idx, modality in enumerate(label_patch):
            label_patch_out[idx] = map_coordinates(label_patch[idx], indices, order=1, mode='reflect').reshape(shape)
    else:
        label_patch_out = None

    # Random flipping
    coin = np.random.uniform()
    if np.logical_and(coin >= 0.0, coin <= 0.333):
        data_patch_out = np.flip(data_patch_out.astype(np.float32), axis=x_axis)
        label_patch_out = np.flip(
            label_patch_out.astype(np.float32), axis=x_axis) if label_patch_out is not None else None
    elif np.logical_and(coin >= 0.333, coin <= 0.666):
        data_patch_out = np.flip(data_patch_out.astype(np.float32), axis=y_axis)
        label_patch_out = np.flip(
            label_patch_out.astype(np.float32), axis=y_axis) if label_patch_out is not None else None
    else:
        pass

    return data_patch_out, label_patch_out

def get_augment_functions_2d(x_axis=1, y_axis=2):
    augment_funcs = {
        0: lambda patches: elastic_transform_patches(patches[0], patches[1], x_axis, y_axis),
        1: lambda patches: elastic_transform_patches(patches[0], patches[1], x_axis, y_axis),
        2: lambda patches: elastic_transform_patches(patches[0], patches[1], x_axis, y_axis),
        3: lambda patches: elastic_transform_patches(patches[0], patches[1], x_axis, y_axis),
        4: lambda patches: elastic_transform_patches(patches[0], patches[1], x_axis, y_axis)
    }

    return augment_funcs

def get_augment_functions(x_axis=1, y_axis=2):
    augment_funcs = {
        0: lambda patch: (np.rot90(patch[0].astype(np.float32), k=1, axes=(x_axis, y_axis)), np.rot90(patch[1].astype(np.float32), k=1, axes=(x_axis, y_axis))),
        1: lambda patch: (np.rot90(patch[0].astype(np.float32), k=2, axes=(x_axis, y_axis)), np.rot90(patch[1].astype(np.float32), k=2, axes=(x_axis, y_axis))),
        2: lambda patch: (np.rot90(patch[0].astype(np.float32), k=3, axes=(x_axis, y_axis)),np.rot90(patch[1].astype(np.float32), k=3, axes=(x_axis, y_axis))),
        3: lambda patch: (np.flip(patch[0].astype(np.float32), axis=x_axis), np.flip(patch[1].astype(np.float32), axis=x_axis)),
        4: lambda patch: (np.flip(patch[0].astype(np.float32), axis=y_axis), np.flip(patch[1].astype(np.float32), axis=x_axis)),
    }

    return augment_funcs