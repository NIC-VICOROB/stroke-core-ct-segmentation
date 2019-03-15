import logging as log
import numpy as np
import math
import itertools as iter
import scipy.ndimage as ndimage
import copy
import sys

from niclib.utils import get_resampling_indexes

def get_patch_slices(centers, patch_shape, step=None):
    assert len(patch_shape) == centers.shape[1] == 3

    ### Compute patch sides for slicing
    half_sizes = [[dim // 2, dim // 2] for dim in patch_shape]
    for i in range(len(half_sizes)):  # If even dimension, subtract 1 to account for assymetry
        if patch_shape[i] % 2 == 0: half_sizes[i][1] -= 1

    # Actually create slices
    patch_locations = []
    for count, center in enumerate(centers):
        patch_slice = (slice(None),  # slice(None) selects all modalities
                       slice(center[0] - half_sizes[0][0], center[0] + half_sizes[0][1] + 1, step),
                       slice(center[1] - half_sizes[1][0], center[1] + half_sizes[1][1] + 1, step),
                       slice(center[2] - half_sizes[2][0], center[2] + half_sizes[2][1] + 1, step))
        patch_locations.append(patch_slice)

    return patch_locations


def filter_slices_out_of_bounds(slice_sets, vol_shape):
    unpack_return = False
    if isinstance(slice_sets, list):
        slice_sets = (slice_sets, )
        unpack_return = True

    assert type(slice_sets) is tuple

    # First slice set in tuple is limit case for out of bounds (typically biggest patch shape)
    slice_reference_set = slice_sets[0]

    valid_slices = []
    for slice_count, patch_slice in enumerate(slice_reference_set):
        for i, dim_slice in enumerate(patch_slice):
            if i > 0 and (dim_slice.start < 0 or dim_slice.stop > vol_shape[i]):
                break
        else:
            valid_slices.append(slice_count)

    slice_sets_out = ()
    for slice_set_in in slice_sets:
        slice_set_out = [slice_set_in[valid_slice] for valid_slice in valid_slices]
        slice_sets_out += (slice_set_out,)

    return slice_sets_out if not unpack_return else slice_sets_out[0]

def get_in_bounds_slices_index(slices, volume_shape):
    valid_slices_idx = []
    for slice_count, patch_slice in enumerate(slices):
        for i, dim_slice in enumerate(patch_slice):
            if i > 0 and (dim_slice.start < 0 or dim_slice.stop > volume_shape[i]):
                break
        else:
            valid_slices_idx.append(int(slice_count))

    return valid_slices_idx

def get_foreground_slices_index(slices, volume_in, min_fg_percentage):
    volume = volume_in if volume_in.ndim == 3 else volume_in[0] # Take modality 0

    volume_foreground = (volume > np.min(volume))
    if np.all(volume_foreground) or np.all(np.invert(volume_foreground)):
        raise(ValueError, "Detected all foreground/background volume (there should be a bit of both right?)")

    valid_idxs = list()
    for idx, s in enumerate(slices):
        vol_slice_foreground = volume_foreground[s[1:]] # Slices are 4D

        if np.any(vol_slice_foreground): # At least one foreground voxel
            if min_fg_percentage == 0.0:
                valid_idxs.append(idx)
            else:
                # A background voxel has been detected, check for ratio
                min_foreground_vox = min_fg_percentage * float(vol_slice_foreground.size)
                if np.count_nonzero(vol_slice_foreground) > min_foreground_vox:
                    valid_idxs.append(idx)

    #log.debug("Filter BG: filtering {} slices out, leaving {}".format(len(slices) - len(valid_idxs), len(valid_idxs)))
    return valid_idxs