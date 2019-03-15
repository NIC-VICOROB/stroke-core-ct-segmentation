from abc import ABC, abstractmethod

import torch
import numpy as np

from niclib.dataset import NIC_Image
from niclib.network.generator import PatchGeneratorBuilder
#from niclib.patch.instructions import PatchExtractInstruction

from niclib.volume import zeropad_sample, remove_zeropad_volume

from niclib.inout.terminal import printProgressBar
from niclib.network.training import ElapsedTimeEstimator


class NIC_Predictor(ABC):
    @abstractmethod
    def predict_sample(self, model, sample):
        pass

class PatchPredictor(NIC_Predictor):
    """
    Predicts a whole volume using patches with the provided model
    """

    def __init__(self, instruction_generator, num_classes, zeropad_shape=None, uncertainty_dropout=0.5, uncertainty_passes=0, uncertainty_dotype='alpha', lesion_class=None, device=torch.device('cuda')):
        assert isinstance(instruction_generator, PatchGeneratorBuilder)
        self.zeropad_shape = zeropad_shape
        self.instr_gen = instruction_generator

        self.num_classes = num_classes
        self.lesion_class = lesion_class
        self.device = device

        self.uncertainty_passes = uncertainty_passes
        self.uncertainty_dropout = uncertainty_dropout
        self.uncertainty_dotype = uncertainty_dotype

    def predict_sample(self, model, sample_in):
        assert isinstance(sample_in, NIC_Image)
        print("Predicting sample with id:{}".format(sample_in.id))

        sample = zeropad_sample(sample_in, self.zeropad_shape)

        batch_size = self.instr_gen.bs
        sample_generator, instructions = self.instr_gen.build_patch_generator(sample, return_instructions=True)

        voting_img = np.zeros((self.num_classes, ) + sample.data[0].shape, dtype=np.float32)
        counting_img = np.zeros_like(voting_img)

        model.eval()
        model.to(self.device)

        if self.uncertainty_passes > 1:
            try:
                model.activate_dropout_testing(p_out=self.uncertainty_dropout, dotype=self.uncertainty_dotype)
                print("Activated uncertainty dropout with p={}".format(self.uncertainty_dropout))
            except AttributeError as ae:
                print(str(ae), "Dropout at test time not configured for this model")
                self.uncertainty_passes = 1

        with torch.no_grad():  # Turns off autograd (faster exec)
            eta = ElapsedTimeEstimator(total_iters=len(sample_generator))
            for batch_idx, (x, y) in enumerate(sample_generator):
                printProgressBar(batch_idx, len(sample_generator),
                                 suffix=' patches predicted - ETA {}'.format(eta.update(batch_idx + 1)))

                # Send generated x,y batch to GPU
                if isinstance(x, list):
                    for i in range(len(x)):
                        x[i] = x[i].to(self.device)
                else:
                    x = x.to(self.device)


                if isinstance(y, list):
                    for i in range(len(y)):
                        y[i] = y[i].to(self.device)
                else:
                    y = y.to(self.device)

                y_pred = model(x)
                if self.uncertainty_passes > 1:
                    for i in range(1, self.uncertainty_passes):
                        y_pred = y_pred + model(x)
                    y_pred = y_pred / self.uncertainty_passes

                y_pred = y_pred.cpu().numpy()
                if len(y_pred.shape) == 4:  # Add third dimension to 2D patches
                    y_pred = np.expand_dims(y_pred, axis=-1)

                batch_slice = slice(batch_idx*batch_size, (batch_idx + 1)*batch_size)
                batch_instructions = instructions[batch_slice]

                assert len(y_pred) == len(batch_instructions)
                for patch_pred, patch_instruction in zip(y_pred, batch_instructions):
                    voting_img[patch_instruction.data_patch_slice] += patch_pred
                    counting_img[patch_instruction.data_patch_slice] += np.ones_like(patch_pred)
            printProgressBar(len(sample_generator), len(sample_generator), suffix=' patches predicted - {} s.'.format(eta.get_elapsed_time()))

        if self.uncertainty_passes > 1:
            model.deactivate_dropout_testing()

        counting_img[counting_img == 0.0] = 1.0 # Avoid division by 0
        volume_probs = np.divide(voting_img, counting_img)

        if self.lesion_class is not None:
            volume_probs = volume_probs[self.lesion_class]
        else:
            volume_probs = np.squeeze(volume_probs, axis=0)

        volume_probs = remove_zeropad_volume(volume_probs, self.zeropad_shape)

        assert np.array_equal(volume_probs.shape, sample_in.foreground.shape), (volume_probs.shape, sample_in.foreground.shape)

        return volume_probs


