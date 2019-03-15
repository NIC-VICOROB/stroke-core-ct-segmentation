import os
import time

import torch
import numpy as np
import nibabel as nib

from niclib.dataset import NIC_Image
from niclib.evaluation.prediction import NIC_Predictor
from niclib.postprocessing.binarization import ThreshSizeBinarizer


class TestingPrediction:
    def __init__(self, predictor, out_path, binarizer=None, save_probs=False, save_seg=True):
        assert isinstance(predictor, NIC_Predictor)
        self.test_predictor = predictor
        self.binarizer = binarizer

        if not os.path.exists(out_path):
            os.mkdir(out_path)
        self.out_path = out_path

        self.do_save_probs = save_probs
        self.do_save_seg = save_seg

    def predict_test_set(self, model, test_images):
        assert isinstance(test_images, list) and all([isinstance(img, NIC_Image) for img in test_images])

        for test_img in test_images:
            # Prediction
            probs_img = self.test_predictor.predict_sample(model, test_img)
            probs_out = nib.Nifti1Image(probs_img, test_img.nib['affine'], test_img.nib['header'])

            probs_filepath = os.path.join(self.out_path, '{}_test_probs.nii'.format(test_img.id))
            if self.do_save_probs:
                print('Saving {}'.format(probs_filepath))
                nib.save(probs_out, probs_filepath)

            if self.binarizer is not None:
                # Binary segmentation
                binary_segmentation = self.binarizer.binarize(probs_img)
                # Adaptation to SMIR.ch specification
                binary_segmentation = binary_segmentation.astype('uint16')
                binary_segmentation = np.multiply(binary_segmentation, test_img.foreground)
                seg_out = nib.Nifti1Image(binary_segmentation, test_img.nib['affine'], test_img.nib['header'])

                # Storage
                seg_filepath = os.path.join(self.out_path, '{}_test_seg.nii'.format(test_img.id))

                if self.do_save_seg:
                    print('Saving {}'.format(seg_filepath))
                    nib.save(seg_out, seg_filepath)