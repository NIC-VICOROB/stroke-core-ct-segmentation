import sys

sys.path.append('niclib')

import os
import json
from socket import gethostname

from niclib.architecture.SUNet import SUNETx5

from niclib.network.generator import *
from niclib.network.optimizers import TorchOptimizer
from niclib.network.loss_functions import *
from niclib.network.training import EarlyStoppingTrain

from niclib.evaluation.prediction import PatchPredictor
from niclib.evaluation.testing import TestingPrediction

from niclib.utils import *

torch.set_default_tensor_type('torch.FloatTensor')


from DataTXT import DataTXT # It works, no worries
from evaluation import DockerValidation
from art import tprint

def launch_code():
    # Read config
    with open('/storage/config.txt', 'r') as config_file:
        config_dict = json.loads(config_file.read())  # use `json.dumps` to do the reverse

    # Launch script
    if config_dict['execution'] == 'training':
        tprint('Training')
        run_training(params=config_dict)
    elif config_dict['execution'] == 'inference':
        tprint('Inference')
        run_inference(params=config_dict)
    else:
        raise Exception('Execution type not recognised')



def run_training(params=None):
    checkpoints_dir = '/storage/models/'
    models_dict_pathfile = os.path.join('/storage/', 'models.txt')

    checkpoint_path = os.path.join(checkpoints_dir, '{}.pt'.format(params['model_name']))

    # 1st load dataset
    dataset = DataTXT(
        txt_path=params['dataset_path'],
        symmetry=params['symmetric_modalities'],
        skull_stripping=params['skull_stripping'],
        has_gt=True)
    dataset.load()
    num_modalities = len(dataset.train[0].data)

    pretrained_path = None
    with open(models_dict_pathfile, 'r') as models_file:
        pretrained_dict = json.loads(models_file.read())  # use `json.dumps` to do the reverse
        print(pretrained_dict)

        if params['pretrained_name'] not in {'None'}:
            # Load path and associated options
            pretrained_path = pretrained_dict[params['pretrained_name']]['path']
            pretrain_sym = pretrained_dict[params['pretrained_name']]['symmetric_modalities']
            pretrained_num_mods = pretrained_dict[params['pretrained_name']]['num_modalities']

            assert pretrain_sym == params['symmetric_modalities'], \
                "Pretrained model and current one don't have same symmetric augmentation"

            assert pretrained_num_mods == num_modalities, \
                "Pretrained model and current one don't have same number of modalities augmentation"

    # --------------------------------------------------------------
    #    CONFIGURATION
    # --------------------------------------------------------------
    visible_dict = {}
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_dict.get(gethostname(), '0')  # Return '0' by default
    print("Using GPU {}".format(os.environ['CUDA_VISIBLE_DEVICES']))


    cfg = {
        # Dataset
        'architecture': SUNETx5(in_ch=num_modalities, out_ch=2, nfilts=8, ndims=2),

        # Patches
        'npatches': 4000, # 4000
        'patch_shape': (64, 64, 1),

        # Training
        'do_train': True,
        'max_epochs': 50, # 50
        'batch_size': 32,
        'train_loss': NIC_binary_xent_gdl(type_weight='Simple'),
        'early_stop': {'patience': 8, 'metric': 'l1_er'},

        # Evaluation
        'num_folds': 5,
        'crossval_folds': None,

        # Prediction
        'patch_shape_pred': (64, 64, 1),
        'uncertainty': {'runs': 0, 'rate': 0.1, 'type': 'alpha'}
    }

    # if params is not None:
    #     print("Setting parameters in configuration queue")
    #     recursive_update(cfg, params)  # Overwrite cfg with keys present in paramas (others left untouched)


    # --------------------------------------------------------------
    #    Experiment
    # --------------------------------------------------------------

    # 2. Model
    model_def = copy.deepcopy(cfg['architecture'])

    model_parameters = filter(lambda p: p.requires_grad, model_def.parameters())
    nparams = sum([np.prod(p.size()) for p in model_parameters])
    print("Setting model instance from {} architecture with {} parameters".format(model_def.__class__.__name__, nparams))

    # 3. Training and validation sets
    train_gen = PatchGeneratorBuilder(
        batch_size=cfg['batch_size'],
        zeropad_shape=cfg['patch_shape'],
        instruction_generator=PatchInstructionGenerator(
            in_shape=cfg['patch_shape'],
            out_shape=cfg['patch_shape'],
            sampler=HybridLesionSampling(
                in_shape=cfg['patch_shape'], num_min_max_lesion=(cfg['npatches']//3, cfg['npatches']), num_uniform=cfg['npatches']),
            augment_to=cfg['npatches']),
        num_workers=4,
        shuffle=True)

    val_gen = PatchGeneratorBuilder(
        batch_size=cfg['batch_size'],
        zeropad_shape=cfg['patch_shape'],
        instruction_generator=PatchInstructionGenerator(
            in_shape=cfg['patch_shape'],
            out_shape=cfg['patch_shape'],
            sampler=HybridLesionSampling(
                in_shape=cfg['patch_shape'], num_min_max_lesion=(cfg['npatches']//3, cfg['npatches']), num_uniform=cfg['npatches']),
            augment_to=cfg['npatches']),
        num_workers=4,
        shuffle=True)

    trainer = EarlyStoppingTrain(
        do_train=cfg['do_train'],
        device=torch.device('cuda'),
        max_epochs=cfg['max_epochs'],
        batch_size=cfg['batch_size'],
        loss_func=cfg['train_loss'],
        optimizer=TorchOptimizer(torch.optim.Adadelta, opts={'rho':0.95}),
        train_metrics={'acc':nic_binary_accuracy},
        eval_metrics={
            'acc': nic_binary_accuracy,
            'dice': nic_binary_dice,
            'l1_er': nic_binary_l1_er},
        early_stopping_metric=cfg['early_stop']['metric'],
        early_stopping_patience=cfg['early_stop']['patience'])



    validation = DockerValidation(
        model_definition=model_def,
        images=dataset.train,
        split_ratio=0.25,
        model_trainer=trainer,
        train_instr_gen=train_gen,
        val_instr_gen=val_gen,
        checkpoint_pathfile=checkpoint_path,
        pretrained_pathfile=pretrained_path,
        log_pathfile=None,
    )

    # 4. EXECUTION
    # Run validation
    validation.run_eval()

    print("Finished training")

    print("Registering trained model...")
    model_dict_entry = {
        'path': checkpoint_path,
        'num_modalities': num_modalities,
        'symmetric_modalities': params['symmetric_modalities'],
    }

    with open(models_dict_pathfile, 'r') as config_file:
        models_dict = json.loads(config_file.read())
    models_dict.update({'{}'.format(params['model_name']):model_dict_entry})
    with open(models_dict_pathfile, 'w') as config_file:
        config_file.write(json.dumps(models_dict))

    print(model_dict_entry)

def run_inference(params=None):
    formatted_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    results_path = '/results/results_{}_{}/'.format(formatted_datetime, params['pretrained_name'])

    models_dict_pathfile = os.path.join('/storage/', 'models.txt')
    with open(models_dict_pathfile, 'r') as models_file:
        pretrained_dict = json.loads(models_file.read())  # use `json.dumps` to do the reverse

    # Load path and associated options
    pretrained_path = pretrained_dict[params['pretrained_name']]['path']
    pretrained_sym = pretrained_dict[params['pretrained_name']]['symmetric_modalities']
    pretrained_num_mods = pretrained_dict[params['pretrained_name']]['num_modalities']


    # 1st load dataset
    dataset = DataTXT(
        txt_path=params['dataset_path'],
        symmetry=pretrained_sym,
        skull_stripping=params['skull_stripping'],
        has_gt=False)
    dataset.load()

    # assert that num modalities is respected
    for case in dataset.train:
        assert len(case.data) == pretrained_num_mods, \
            "Trained number of modalities doesn't match inference images"

    # 2nd model and rest
    model_def = torch.load(pretrained_path)

    model_parameters = filter(lambda p: p.requires_grad, model_def.parameters())
    nparams = sum([np.prod(p.size()) for p in model_parameters])
    print("Setting model instance from {} architecture with {} parameters".format(model_def.__class__.__name__, nparams))

    visible_dict = {}
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_dict.get(gethostname(), '0')  # Return '0' by default
    print("Using GPU {}".format(os.environ['CUDA_VISIBLE_DEVICES']))

    patch_shape_pred = (64, 64, 1)

    predictor = PatchPredictor(
        num_classes=2,
        lesion_class=1,
        uncertainty_passes=3,
        uncertainty_dropout=0.1,
        zeropad_shape = patch_shape_pred,
        instruction_generator=PatchGeneratorBuilder(
            batch_size=64,
            shuffle=False,
            zeropad_shape=None,
            instruction_generator=PatchInstructionGenerator(
                in_shape=patch_shape_pred,
                out_shape=patch_shape_pred,
                sampler=UniformSampling(
                    in_shape=patch_shape_pred, extraction_step=(3, 3, 1)),
                augment_to=None)))

    test_pred = TestingPrediction(predictor=predictor, out_path=results_path, save_probs=True, save_seg=False)

    if not os.path.isdir(results_path):
        os.mkdir(results_path)
    test_pred.predict_test_set(model_def, dataset.train)



if __name__ == '__main__':
    launch_code()

    
