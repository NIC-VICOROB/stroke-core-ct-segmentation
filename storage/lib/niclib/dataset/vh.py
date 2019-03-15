import os
import nibabel as nib
import numpy as np
import itertools as iter

from niclib.dataset import NIC_Image, NIC_Dataset

class VH(NIC_Dataset):
    def __init__(self, dataset_path, load_test=True, symmetric_modalities=False):
        super().__init__()

        self.dataset_path = os.path.expanduser(dataset_path)

        self.train_names = ['VPL07_0_RBF', 'VPL10_0_FNG', 'VPL13_0_BFN', 'VPL16_0_AVX', 'VPL19_0_AVX', 'VPL08_0_RBF',
                            'VPL11_0_BFN', 'VPL14_0_BFN', 'VPL17_0_AVX', 'VPL20_0_NTZ', 'VPL09_0_RBF', 'VPL12_0_BFN',
                            'VPL15_0_BFN', 'VPL18_0_AVX']

        self.test_names = ['VPI01_0', 'VPI03_0', 'VPI05_0', 'VPI07_0', 'VPI10_0', 'VPL02_0_COP', 'VPL04_0_COP',
                           'VPI02_0', 'VPI04_0', 'VPI06_0', 'VPI08_0', 'VPL01_0_AVX', 'VPL03_0_COP', 'VPL06_0_RBF']


        self.modalities = ['FLAIR', 'T1']
        if symmetric_modalities:
            self.modalities += ['sym_{}'.format(mod) for mod in self.modalities]
            print(self.modalities)

    def load(self):
        dataset_path = self.dataset_path
        modalities = self.modalities

        print("Loading VH_synth dataset with modalities {}...".format(modalities))

        # Training loading
        for case_name in self.train_names:
            case_folder = os.path.join(dataset_path, 'train/{}/'.format(case_name))

            initialized, image_data, labels, nib_nifty = False, None, None, None
            for m_idx, modality in enumerate(modalities):
                filepath = case_folder + '{}.nii.gz'.format(modality)

                nib_file = nib.load(filepath)
                vol = nib_file.get_data()

                if not initialized:
                    image_data = np.zeros((len(modalities),) + vol.shape)
                    nib_nifty = nib_file
                    labels = np.zeros((1,) + vol.shape)
                    initialized = True
                image_data[m_idx] = vol

            labels[0] = nib.load(case_folder + 'lesionMask.nii.gz').get_data() > 0.0

            sample = NIC_Image(
                sample_id=case_name,
                nib_file=nib_nifty,
                image_data=image_data,
                foreground=(image_data[0] > 0.0),
                labels=labels)
            self.add_train(sample)

        # Testing loading
        for case_name in self.test_names:
            case_folder = os.path.join(dataset_path, 'test/{}/'.format(case_name))

            initialized, image_data, labels, nib_nifty = False, None, None, None
            for m_idx, modality in enumerate(modalities):
                filepath = case_folder + '{}.nii.gz'.format(modality)

                nib_file = nib.load(filepath)
                vol = nib_file.get_data()

                if not initialized:
                    image_data = np.zeros((len(modalities),) + vol.shape)
                    nib_nifty = nib_file
                    labels = np.zeros((1,) + vol.shape)
                    initialized = True
                image_data[m_idx] = vol

            labels[0] = nib.load(case_folder + 'lesionMask.nii.gz').get_data()

            sample = NIC_Image(
                sample_id=case_name,
                nib_file=nib_nifty,
                image_data=image_data,
                foreground=(image_data[0] > 0.0),
                labels=labels)
            self.add_test(sample)
