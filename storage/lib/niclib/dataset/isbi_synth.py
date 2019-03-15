import os
import nibabel as nib
import numpy as np
import itertools as iter

from niclib.dataset import NIC_Image, NIC_Dataset

class ISBIsynth(NIC_Dataset):
    def __init__(self, dataset_path, load_synth=False):
        super().__init__()

        self.dataset_path = os.path.expanduser(dataset_path)

        self.train_names = ['01_01', '01_03', '02_01', '02_03', '03_01', '03_03', '04_01', '04_03', '05_01', '05_03', '01_02', '01_04', '02_02', '02_04', '03_02', '03_04', '04_02', '04_04', '05_02', '05_04']

        self.test_names = None

        self.do_load_synth = load_synth
        self.modalities = ['FLAIR', 'T1']
        self.modalities_synth = ['FLAIR_syn_3tissues', 'T1_syn_3tissues']


    def load(self):
        dataset_path = self.dataset_path
        modalities = self.modalities if not self.do_load_synth else self.modalities_synth

        print("Loading ISBI_synth dataset with modalities {}...".format(modalities))

        # Training loading
        for case_name in self.train_names:
            case_folder = os.path.join(dataset_path, '{}/'.format(case_name))

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
            self.add_train(sample)
