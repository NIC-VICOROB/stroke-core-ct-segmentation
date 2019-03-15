import copy
import math
import numpy as np

def zeropad_set(samples_in, patch_shape):
    samples_out = list()
    for sample in samples_in:
        samples_out.append(zeropad_sample(sample, patch_shape))
    return samples_out

def zeropad_sample(sample_in, patch_shape):
    sample = copy.deepcopy(sample_in)

    sample.data = pad_volume(sample.data, patch_shape)
    sample.foreground = pad_volume(sample.foreground, patch_shape)
    try: sample.labels = pad_volume(sample.labels, patch_shape)
    except AttributeError: pass  # Some test samples don't have labels
    return sample

def remove_zeropad_volume(volume, patch_shape):
    # Get padding amount per each dimension
    selection = []
    for dim_size in patch_shape:
        slice_start = int(math.ceil(dim_size / 2.0))
        slice_stop = -slice_start if slice_start != 0 else None
        selection += [slice(slice_start, slice_stop)]
    volume = volume[tuple(selection)]
    return volume

def pad_volume(volume, patch_shape):
    pad_size = [int(math.ceil(patch_dim / 2.0)) for patch_dim in patch_shape]
    padding = [(pad_size[0], pad_size[0]), (pad_size[1], pad_size[1]), (pad_size[2], pad_size[2])]
    if len(volume.shape) == 4: # also includes modality
        padding = [(0,0)] + padding
    return np.pad(volume, tuple(padding), 'constant', constant_values=0).astype(np.float32)
