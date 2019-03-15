import os
import nibabel as nib
import numpy as np
import itertools as iter

from niclib.dataset import NIC_Image, NIC_Dataset


class Isles2018(NIC_Dataset):
    def __init__(self, dataset_path, num_volumes=(94, 62), modalities=('CT_SS', 'CT_CBF', 'CT_CBV', 'CT_MTT', 'CT_Tmax'), load_testing=True):
        super().__init__()
        self.dataset_path = os.path.expanduser(dataset_path)
        self.num_volumes = num_volumes
        self.modalities = modalities
        self.load_testing = load_testing

    def load(self):
        dataset_path = self.dataset_path
        num_volumes = self.num_volumes
        modalities = self.modalities
        load_testing = self.load_testing

        print("Loading ISLES2018 dataset with modalities {}...".format(modalities))
        pattern = ['training/case_{}/', 'testing/case_{}/']
        # Training loading
        for case_idx in range(num_volumes[0]):
            case_folder = os.path.join(dataset_path, pattern[0].format(str(case_idx + 1)))

            initialized, image_data, labels, nib_nifty = False, None, None, None
            for m_idx, modality in enumerate(modalities):
                filepath = case_folder + '{}.nii'.format(modality)

                try:
                    nib_file = nib.load(filepath)
                except Exception as e:
                    filepath_gz = filepath + '.gz'
                    nib_file = nib.load(filepath_gz)

                vol = nib_file.get_data()
                if not initialized:
                    image_data = np.zeros((len(modalities),) + vol.shape)
                    nib_nifty = nib_file
                    labels = np.zeros((1,) + vol.shape)
                    initialized = True
                image_data[m_idx] = vol
            labels[0] = nib.load(case_folder + 'OT.nii').get_data()

            sample = NIC_Image(
                sample_id=case_idx + 1,
                nib_file=nib_nifty,
                image_data=image_data,
                foreground=(image_data[0] > 0.0),
                labels=labels)
            self.add_train(sample)

        # Testing loading
        if not load_testing:
            return

        for case_idx in range(num_volumes[1]):
            case_folder = os.path.join(dataset_path, pattern[1].format(str(case_idx + 1)))

            initialized, image_data, labels, nib_nifty = False, None, None, None
            for m_idx, modality in enumerate(modalities):
                filepath = case_folder + '{}.nii'.format(modality)

                try: nib_file = nib.load(filepath)
                except Exception as e: nib_file = nib.load(filepath[:-3] if filepath.endswith('.gz') else filepath + '.gz')

                vol = nib_file.get_data()

                if not initialized:
                    image_data = np.zeros((len(modalities),) + vol.shape)
                    nib_nifty = nib_file
                    initialized = True
                image_data[m_idx] = vol

            sample = NIC_Image(
                sample_id=case_idx + 1,
                nib_file=nib_nifty,
                image_data=image_data,
                foreground=(image_data[0] > 0.0),
                labels=None)
            self.add_test(sample)

