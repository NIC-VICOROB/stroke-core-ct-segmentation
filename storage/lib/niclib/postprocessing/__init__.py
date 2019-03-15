import numpy as np
from abc import ABC, abstractmethod

class NIC_Postprocessing(ABC):
    @abstractmethod
    def postprocess(self, vol, params=None):
        pass


class Denormalizer(NIC_Postprocessing):
    def __init__(self, out_range=(0, 255)):
        self.out_range = out_range

    def postprocess(self, vol, params=None):
        """
        Denormalizes an image (useful for image to image translation
        :param vol: volume to postprocess (modalities as first dimension)
        :param params: dictionary with keys 'mean' and 'std_dev' with a list with mean and std for each modality
        :return: denormalized volume in the specified range
        """

        vol_out = denormalize_volume(vol, stats=params[0])
        if self.out_range is not None:
            vol_out = np.rint((self.out_range[1] * (vol_out - np.min(vol_out))) / (np.max(vol_out) - np.min(vol_out)))
        return vol_out


def denormalize_volume(vol, stats):
    for modality in range(len(stats['mean'])):
        vol[modality] = (vol[modality] * stats['std_dev'][modality]) + stats['mean'][modality]
    return vol