from abc import ABC, abstractmethod

import torch

from niclib.patch.patches import *
from niclib.patch.samples import *
import itertools as iter
from abc import ABC, abstractmethod

from niclib.volume import zeropad_sample
from niclib.patch.centers import *
from niclib.patch.slices import *
from niclib.patch.sampling import *
from niclib.inout.terminal import printProgressBar
from niclib.dataset import NIC_Dataset


class NIC_Instruction(ABC):
    @abstractmethod
    def extract_from(self, images):
        pass

class NIC_InstructionGenerator(ABC):
    @abstractmethod
    def generate_instructions(self, images):
        pass


class PatchSegmentationInstruction(NIC_Instruction):
    def __init__(self, case_id=-1, data_patch_slice=None, label_patch_slice=None, augment_func=None, normalise=True):
        self.case_id = case_id

        self.data_patch_slice = data_patch_slice
        self.label_patch_slice = label_patch_slice

        self.normalise_data_patch = normalise
        self.augment_func = augment_func

    def extract_from(self, images):
        # Obtain image from list
        sample = NIC_Dataset.get_by_id(self.case_id, images) if isinstance(images, list) else images
        assert isinstance(sample, NIC_Image)
        assert self.case_id == sample.id

        # Extract patches
        data_patch = extract_patch_with_slice(sample.data, self.data_patch_slice)

        do_extract_label = self.label_patch_slice is not None and sample.labels is not None
        label_patch = extract_patch_with_slice(sample.labels, self.label_patch_slice) if do_extract_label else None

        # Normalize data patch
        if self.normalise_data_patch:
            data_patch = normalise_patch(data_patch, sample.statistics['mean'], sample.statistics['std_dev'])

        # Augment patches
        if self.augment_func is not None:
            augment_func = get_augment_functions(x_axis=1, y_axis=2)[self.augment_func]
            data_patch, label_patch = augment_func((data_patch, label_patch if do_extract_label else None))

        dpatch = torch.tensor(np.ascontiguousarray(data_patch, dtype=np.float32))
        lpatch = torch.tensor(np.ascontiguousarray(label_patch, dtype=np.float32))
        return dpatch, lpatch


class AutoencoderInstruction(PatchSegmentationInstruction):
    def extract_from(self, images):
        dpatch, lpatch = super().extract_from(images)
        return dpatch, dpatch


class PatchInstructionGenerator(NIC_InstructionGenerator):
    def __init__(self, in_shape, out_shape, sampler, augment_to=None, autoencoder=False):
        assert isinstance(sampler, NICSampler)
        self.in_shape = in_shape
        self.out_shape = out_shape
        self.sampler = sampler
        self.augment_positives = augment_to
        self.autoencoder = autoencoder

    def generate_instructions(self, images):
        assert isinstance(images, list) and all([isinstance(image, NIC_Image) for image in images])

        set_instructions = []
        for idx, image in enumerate(images):
            printProgressBar(idx, len(images), suffix='samples processed')

            centers = self.sampler.get_centers(image)
            if isinstance(centers, tuple):  # Sampling that have two sets of centers
                pos_centers, unif_centers = centers
                lesion_instructions = get_instructions_from_centers(image.id, pos_centers, self.in_shape, self.out_shape,
                                                                 augment_to=self.augment_positives, autoencoder=self.autoencoder)
                unif_instructions = get_instructions_from_centers(image.id, unif_centers, self.in_shape, self.out_shape,
                                                                  augment_to=None, autoencoder=self.autoencoder)
                image_instructions = lesion_instructions + unif_instructions
            else:
                image_instructions = get_instructions_from_centers(image.id, centers, self.in_shape, self.out_shape,
                                                                   augment_to=self.augment_positives, autoencoder=self.autoencoder)
            set_instructions += image_instructions
        printProgressBar(len(images), len(images), suffix='samples processed')

        return set_instructions

def get_instructions_from_centers(sample_id, centers, patch_shape, output_shape, augment_to=None, autoencoder=False):
    data_slices = get_patch_slices(centers, patch_shape)
    label_slices = get_patch_slices(centers, output_shape)

    sample_instructions = list()
    for data_slice, label_slice in zip(data_slices, label_slices):
        if not autoencoder:
            instruction = PatchSegmentationInstruction(
                case_id=sample_id, data_patch_slice=data_slice, label_patch_slice=label_slice, normalise=True)
        else:
            instruction = AutoencoderInstruction(
                sample_id=sample_id, data_patch_slice=data_slice, label_patch_slice=label_slice, normalise=False)
        sample_instructions.append(instruction)

    if augment_to is not None:
        sample_instructions = augment_instructions(sample_instructions, goal_num_instructions=augment_to)

    return sample_instructions



def augment_instructions(original_instructions, goal_num_instructions):
    augment_funcs = get_augment_functions(x_axis=1, y_axis=2)  # (modality, x, y, z)

    num_patches_in = len(original_instructions)
    num_augments_per_patch = np.minimum( int(math.ceil(goal_num_instructions / num_patches_in)), len(augment_funcs))
    goal_num_augmented_patches = int(goal_num_instructions - num_patches_in)

    # Augment and add remaining copies
    sampling_idxs = get_resampling_indexes(num_patches_in, goal_num_augmented_patches // num_augments_per_patch)
    func_idxs = get_resampling_indexes(len(augment_funcs), num_augments_per_patch)

    augmented_instructions = list()
    for sampling_idx, func_idx in iter.product(sampling_idxs, func_idxs):
        aug_instr = copy.copy(original_instructions[sampling_idx])
        aug_instr.augment_func = func_idx
        augmented_instructions.append(aug_instr)

    final_instructions = original_instructions + augmented_instructions

    return final_instructions