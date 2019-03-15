import torch
import os
import copy

from niclib.network.training import EarlyStoppingTrain
from niclib.network.generator import PatchGeneratorBuilder
from niclib.postprocessing import NIC_Postprocessing

from niclib.inout.results import *
from niclib.metrics import *

from niclib.utils import *
from niclib.inout.metrics import print_metrics_list

class SimpleValidation:
    def __init__(self, model_definition, images, split_ratio, model_trainer, train_instr_gen, val_instr_gen, checkpoint_pathfile, log_pathfile, test_predictor, results_path, pred_images=None, pretrained_pathfile=None):
        assert isinstance(model_trainer, EarlyStoppingTrain)  # TODO make abstract trainer class
        assert isinstance(train_instr_gen, PatchGeneratorBuilder)
        assert isinstance(val_instr_gen, PatchGeneratorBuilder)

        assert checkpoint_pathfile.endswith('.pt')
        assert log_pathfile.endswith('.csv')

        self.model_definition = model_definition
        self.images = images
        self.pred_images = pred_images
        self.split_ratio = split_ratio

        self.pretrained_pathfile = pretrained_pathfile
        self.checkpoint_pathfile = checkpoint_pathfile
        self.log_pathfile = log_pathfile

        self.trainer = model_trainer
        self.train_instr_gen = train_instr_gen
        self.val_instr_gen = val_instr_gen

        self.predictor = test_predictor

        if not os.path.exists(results_path):
            os.mkdir(results_path)
        self.results_path = results_path

    def run_eval(self):
        start_idx_val, stop_idx_val = get_val_split_indexes(images=self.images, split_ratio=self.split_ratio)

        print("\n" + "-" * 75 +"\n Running eval on val images {} to {} \n".format(start_idx_val, stop_idx_val) + "-" * 75 + "\n", sep='')

        model_fold = copy.deepcopy(self.model_definition)
        if self.pretrained_pathfile is not None:
            print("Loading pre-trained model at {}".format(self.pretrained_pathfile))
            model_fold = torch.load(self.pretrained_pathfile)

        train_images = self.images[:start_idx_val] + self.images[stop_idx_val:]
        val_images = self.images[start_idx_val:stop_idx_val]
        if self.trainer.do_train:
            print("Building training generator...")
            train_gen = self.train_instr_gen.build_patch_generator(images=train_images)
            print("Building validation generator...")
            val_gen = self.val_instr_gen.build_patch_generator(images=val_images)
            print("Generators with {} training and {} validation patches".format(
                len(train_gen)*self.trainer.bs, len(val_gen)*self.trainer.bs))

            self.trainer.train(model_fold, train_gen, val_gen, self.checkpoint_pathfile, self.log_pathfile)
        else:
            torch.save(model_fold, self.checkpoint_pathfile)

        print("Loading trained model {}".format(self.checkpoint_pathfile))
        model_fold = torch.load(self.checkpoint_pathfile, self.trainer.device)

        # Predict set
        pred_images = self.pred_images if self.pred_images is not None else val_images
        for n, sample in enumerate(pred_images):
            predicted = self.predictor.predict_sample(model_fold, sample)
            save_image_probs(self.results_path + '{}_pred.nii.gz'.format(sample.id), sample, predicted)