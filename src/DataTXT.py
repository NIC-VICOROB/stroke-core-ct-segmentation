import os
import numpy as np
import nibabel as nib
import json

from niclib.dataset import NIC_Dataset, NIC_Image
from preprocess import augment_symmetric_modality, skull_strip_ct_isles18


class DataTXT(NIC_Dataset):
    def __init__(self, txt_path, symmetry=False, skull_stripping=False, storage_path='/storage/', has_gt=True):
        super().__init__()
        assert os.path.isfile(txt_path)
        self.txt_path = txt_path
        self.base_path = os.path.split(txt_path)[0]

        self.storage_path = storage_path
        self.symmetry_path = os.path.join(self.storage_path, 'symmetric_images')

        self.do_skull_strip = skull_stripping

        self.has_gt = has_gt

        # Read and check image paths
        self.dataset_filepaths = []
        with open(self.txt_path) as txt:
            case_filepaths = []

            for line in txt:
                if len(line) < 3:  # Empty line
                    if len(case_filepaths) > 0:
                        self.dataset_filepaths.append(case_filepaths)
                        case_filepaths = []
                else:
                    full_path = os.path.join(self.base_path, line).strip()
                    if os.path.isfile(full_path):
                        case_filepaths.append(full_path)

        # Assert files exist
        for case in self.dataset_filepaths:
            assert all([os.path.isfile(img_file) for img_file in case])

        # Check that
        self.symmetric_dict = None
        if symmetry is True:
            with open(os.path.join(storage_path, 'symmetry.txt'), 'r') as config_file:
                self.symmetric_dict = json.loads(config_file.read())  # use `json.dumps` to do the reverse

            # Assert that every modality has been preprocessed
            print("Checking images have symmetric version...")
            for i, case in enumerate(self.dataset_filepaths):
                if case[0] not in self.symmetric_dict:
                    additional_mods = case[1:-1] if self.has_gt else case[1:]
                    print("\nProcessing case {} - {}".format(i, case[0]))
                    sym_filepaths = augment_symmetric_modality(
                        filepath_in=case[0], path_out=self.symmetry_path,
                        additional_filepaths=additional_mods, prefix='sym_{}'.format(i))

                    # Update lookup table
                    for orig, sym in zip(case, sym_filepaths):
                        self.symmetric_dict.update({orig: sym})
                    with open(os.path.join(storage_path, 'symmetry.txt'), 'w') as config_file:
                        config_file.write(json.dumps(self.symmetric_dict))

    def load(self):
        print("DataTXT: Loading from {}".format(self.txt_path))

        for case_filepaths in self.dataset_filepaths:
            original_filepaths = case_filepaths[:-1] if self.has_gt else case_filepaths
            gt_filepath = case_filepaths[-1]

            # 1. Load original modalities
            original_images = [nib.load(fp).get_data() for fp in original_filepaths] # Omit gt?

            if self.do_skull_strip:
                original_images[0] = skull_strip_ct_isles18(original_images[0], original_images[1:])

            # Add also the symmetric modalities
            symmetric_images = []
            if self.symmetric_dict is not None:
                symmetric_images = [nib.load(self.symmetric_dict[fp] + '.gz').get_data() for fp in original_filepaths]
                if self.do_skull_strip:
                    symmetric_images[0] = skull_strip_ct_isles18(symmetric_images[0], symmetric_images[1:])

            # get all images in one list
            all_images = original_images + symmetric_images

            # 2. Load ground_truth
            gt = nib.load(gt_filepath).get_data()
            gt = np.expand_dims(gt, axis=0)

            # 3. Put everything in dataset
            self.add_train(NIC_Image(
                sample_id=str(len(self.train) + 1),
                nib_file=nib.load(case_filepaths[0]),
                image_data=np.stack(all_images, axis=0),
                foreground=original_images[0] > 0.0,
                labels=gt))

        print("DataTXT: Loaded {} cases with {} modalities".format(len(self.train), len(self.train[0].data)))