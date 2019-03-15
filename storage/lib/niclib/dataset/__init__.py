import itertools
from abc import ABC, abstractmethod
import numpy as np
import nibabel as nib

class NIC_Image:
    def __init__(self, sample_id, nib_file, image_data, foreground, labels, as_type='float16'):
        assert isinstance(nib_file, nib.Nifti1Image) or isinstance(nib_file, nib.Nifti2Image)

        self.id = sample_id
        self.nib = {'affine': nib_file.affine, 'header': nib_file.header} # Affine, header
        self.data = image_data
        self.foreground = foreground
        self.labels = labels
        self.statistics = {'mean': [np.mean(modality) for modality in self.data],
                           'std_dev': [np.std(modality) for modality in self.data]}

class NIC_Dataset(ABC):
    def __init__(self):
        self.train = []
        self.test = []

    @abstractmethod
    def load(self):
        pass

    def add_train(self, image_in):
        assert isinstance(image_in, NIC_Image)
        for image_train in self.train:
            assert image_train.id != image_in.id
        self.train.append(image_in)

    def add_test(self, image_in):
        assert isinstance(image_in, NIC_Image)
        for image_test in self.test:
            assert image_test.id != image_in.id
        self.test.append(image_in)

    @staticmethod
    def get_by_id(wanted_id, images):
        if isinstance(images, NIC_Dataset):
            for image in itertools.chain(images.train, images.test):
                assert isinstance(image, NIC_Image)
                if image.id == wanted_id:
                    return image
        elif isinstance(images, list):
            for image in images:
                assert isinstance(image, NIC_Image)
                if image.id == wanted_id:
                    return image
        else:
            raise (ValueError, "Given images are not a valid instance of NICdataset or a list of NICimages")

        raise(ValueError, "Desired id not found in given images")