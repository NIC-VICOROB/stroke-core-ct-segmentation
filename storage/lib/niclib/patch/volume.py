import nibabel as nib
import logging as log
import os
import itertools
import numpy as np
import math

def read_volume(filename):

    try:
        nib_file = load_nib_file(filename)
    except Exception as e:
        filename = filename[:-3] if filename.endswith('.gz') else filename + '.gz'
        nib_file = load_nib_file(filename)

    return nib_file.get_data()

def load_nib_file(filename):
    return nib.load(filename)

def remove_zeropad_volume(volume, patch_shape):
    # Get padding amount per each dimension
    selection = []
    for dim_size in patch_shape:
        slice_start = dim_size // 2
        slice_stop = -slice_start if slice_start != 0 else None
        selection += [slice(slice_start, slice_stop)]
    volume = volume[selection]
    return volume

def pad_volume(volume, patch_shape):
    assert len(patch_shape) == (len(volume.shape) - 1)
    pad_size = [patch_dim // 2 for patch_dim in patch_shape]
    padding = ((0, 0), (pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2]))
    return np.pad(volume, padding, 'constant', constant_values=0).astype(np.float32)

def get_brain_mask(vol):
    if len(vol.shape) > 3:
        vol = vol[0] # Take only first modality
    return np.sum(vol, axis=0) > 0


def reconstruct_volume(patches, patch_shape, original_vol, original_centers, extraction_step, num_classes) :
    expected_shape = original_vol.shape[1:]
    assert len(expected_shape) == len(extraction_step) == len(patch_shape) == 3

    if len(patches.shape) != 5:
        patches = np.expand_dims(patches, axis=3)  # 2D/3D compatibility (None, x, y, new:z, c)

    ### Pad predicted patches to match input patch size (just as they where extracted from original volume)
    output_patch_shape = patches.shape[1:]

    ### Compute patch sides for slicing
    half_sizes = [[dim // 2, dim // 2] for dim in output_patch_shape]
    for i in range(len(half_sizes)):     # If even dimension, subtract 1 to account for assymetry
        if output_patch_shape[i] % 2 == 0: half_sizes[i][1] -= 1

    ### Voting space
    vote_img = np.zeros(expected_shape + (num_classes,), dtype=np.float32)
    count_img = np.zeros(expected_shape + (num_classes,), dtype=np.float32)

    # Create counting patch
    counting_patch = np.ones(output_patch_shape)
    for count, coord in enumerate(original_centers):
        selection = [slice(coord[0] - half_sizes[0][0], coord[0] + half_sizes[0][1] + 1), # x
                     slice(coord[1] - half_sizes[1][0], coord[1] + half_sizes[1][1] + 1), # y
                     slice(coord[2] - half_sizes[2][0], coord[2] + half_sizes[2][1] + 1), # z
                     slice(None)]  #selects all classes

        vote_img[selection] += patches[count]
        count_img[selection] += counting_patch

    count_img[count_img == 0.0] = 1.0  # Avoid division by 0
    volume_probs = np.divide(vote_img, count_img)

    lesion_probs = volume_probs[:, :, :, 1] # Get probability of lesion
    return lesion_probs