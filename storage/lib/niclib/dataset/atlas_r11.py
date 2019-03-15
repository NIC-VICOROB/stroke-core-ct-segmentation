import os
import nibabel as nib
import numpy as np
import itertools as iter

from niclib.inout.terminal import printProgressBar
from dataset import NIC_Image, NIC_Dataset


class Atlas_R11(NIC_Dataset):
    def __init__(self, path, modalities=('t1w',), nvols=229):
        super().__init__()
        print("Loading ATLAS dataset...")

        dataset_path = os.path.expanduser(path)
        mni_brain_mask = nib.load(os.path.join(os.path.expanduser('~/atlases/'), 'atlas_brain_mask.nii.gz')).get_data()

        """
        img = nib.Nifti1Image(mni_brain_mask.astype('uint8'), np.eye(4))
        img.to_filename('atlas_brain_mask.nii.gz')
        raise NotImplementedError
        """

        loaded_samples = 0
        for root, subdirs, files in os.walk(dataset_path):
            if any(['t1w' in filename for filename in files]):
                printProgressBar(loaded_samples, nvols, suffix='samples loaded')
                loaded_samples += 1

                t1_path = [os.path.join(root, filename) for filename in files if 't1w' in filename][0]
                lesion_paths = [os.path.join(root, filename) for filename in files if 'LesionSmooth' in filename]

                sample_id = '{}_{}_{}'.format(
                    t1_path[t1_path.find('Site') + 4],
                    t1_path[t1_path.find('/t0') + 3],
                    os.path.basename(t1_path).split('_')[0])

                # Load volume to check dimensions (not the same for all train samples)
                nib_file = nib.load(t1_path)
                vol = nib_file.get_data()

                data = np.zeros((len(modalities),) + vol.shape, dtype='float32')
                labels = np.zeros((1,) + vol.shape, dtype='float32')

                # DATA
                data[0] = vol * mni_brain_mask
                foreground_mask = mni_brain_mask

                # LABELS
                for lesion_file in lesion_paths:
                    labels = np.logical_or(nib.load(lesion_file).get_data() > 0, labels)

                sample = NIC_Image(sample_id, nib_file, data, foreground_mask, labels)
                self.train.append(sample)

                if loaded_samples > nvols:
                    break

        printProgressBar(nvols, nvols, suffix='samples loaded')
