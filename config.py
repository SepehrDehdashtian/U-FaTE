# config.py

import os
import shutil
import datetime
import argparse
import json
import configparser
import hal.utils.misc as misc
import re
from ast import literal_eval as make_tuple
import time

def parse_args():
    result_path = "results/"
    now = datetime.datetime.now().strftime('%m%d-%H%M%S')
    # result_path = os.path.join(result_path, now)

    parser = argparse.ArgumentParser(description='Controllable Representation Learning')

    # the following two parameters can only be provided at the command line.
    parser.add_argument("-c", "--config", "--args-file", dest="config_file", default="args.txt", help="Specify a config file", metavar="FILE")
    args, remaining_argv = parser.parse_known_args()
    parser.add_argument('--result-path', type=str, default=result_path, metavar='', help='full path to store the results')

    # ======================= Project Settings =====================================
    parser.add_argument('--project-name', type=str, default='myproject', metavar='', help='name of the project')
    parser.add_argument('--save-dir', type=str, default="", metavar='', help='save the trained models here')
    parser.add_argument('--feat-dir', type=str, default="", metavar='', help='save the features here')
    parser.add_argument('--figs-dir', type=str, default="", metavar='', help='save the figures here')
    parser.add_argument('--main-figs-dir', type=str, default="", metavar='', help='save the figures of different experiments here')
    parser.add_argument('--no-progress-bar', type=str, default="false", metavar='', help='turn off progress bar')
    parser.add_argument('--result-subdir', type=str, default="", metavar='', help='subdir inside results folder')
    parser.add_argument('--logs-dir', type=str, default="", metavar='', help='save the training log files here')
    parser.add_argument('--monitor', type=json.loads, default={}, metavar='', help='metric based on which we save models')
    parser.add_argument('--checkpoint-max-history', type=int, default=-1, metavar='', help='max checkpopint history')
    parser.add_argument('-s', '--save', '--save-results', type=misc.str2bool, dest="save_results",default='Yes', metavar='', help='save the arguments and the results')
    parser.add_argument('--mode', type=str, default="Train", metavar='', help='whether we are training encoder or evaluating encoder')

    parser.add_argument('--exp-name', type=str, default='', metavar='', help='Tensorboard subdir')
    parser.add_argument('--wandb-group', type=str, default='', metavar='', help='Weights and Biases name for the experiment group')
    parser.add_argument('--tune-flag', type=misc.str2bool, default='No', metavar='', help='Shows that whether it is a tuning process or not')
    
    # ======================= Data Settings =====================================
    parser.add_argument('--dataroot', type=str, default=None, help='path of the data root')
    parser.add_argument('--available-domains', type=misc.str2list, default=None, help='Available Domains', required=False)
    parser.add_argument('--dataset-root-test', type=str, default=None, help='path of the data')
    parser.add_argument('--dataset-root-train', type=str, default=None, help='path of the data')
    parser.add_argument('--dataset-test', type=str, default=None, help='name of training dataset')
    parser.add_argument('--dataset-train', type=str, default=None, help='name of training dataset')
    parser.add_argument('--split-test', type=float, default=None, help='test split')
    parser.add_argument('--test-flag', type=str, default="", help='test dataloader')
    parser.add_argument('--split-train', type=float, default=None, help='train split')
    parser.add_argument('--test-dev-percent', type=float, default=None, metavar='', help='percentage of dev in test')
    parser.add_argument('--train-dev-percent', type=float, default=None, metavar='', help='percentage of dev in train')
    parser.add_argument('--resume', type=str, default=None, help='full path of models pretrained checkpoint')
    parser.add_argument('--pretrained-checkpoint', type=str, default=None, help='full path of models to resume training')
    parser.add_argument('--nclasses', type=int, default=None, metavar='', dest='noutputs', help='number of classes for classification')
    parser.add_argument('--noutputs', type=int, default=None, metavar='', help='number of outputs, i.e. number of classes for classification')
    parser.add_argument('--input-filename-test', type=str, default=None, help='input test filename for filelist and folderlist')
    parser.add_argument('--label-filename-test', type=str, default=None, help='label test filename for filelist and folderlist')
    parser.add_argument('--input-filename-train', type=str, default=None, help='input train filename for filelist and folderlist')
    parser.add_argument('--label-filename-train', type=str, default=None, help='label train filename for filelist and folderlist')
    parser.add_argument('--loader-input', type=str, default=None, help='input loader')
    parser.add_argument('--loader-label', type=str, default=None, help='label loader')
    parser.add_argument('--dataset', type=str, default=None, metavar='', help='name of the dataset')
    parser.add_argument('--dataset-options', type=json.loads, default=None, metavar='', help='additional model-specific parameters')
    parser.add_argument('--alpha', type=float, default=0., metavar='', dest='alpha', help='gaussian data parameter')
    parser.add_argument('--transform-trn', type=json.loads, default={}, metavar='', help='training data transforms')
    parser.add_argument('--transform-val', type=json.loads, default={}, metavar='', help='validation data transforms')
    parser.add_argument('--transform-tst', type=json.loads, default={}, metavar='', help='testing data transforms')
    parser.add_argument('--cache-size', type=int, default=None, help='lmdb data loader cache size')
    parser.add_argument('--dataset-type', type=str, default=None, help='dataset type')

    # ======================= Network Model Settings ============================
    parser.add_argument('--model-type', type=json.loads, default={}, help='type of network')
    parser.add_argument('--model-options', type=json.loads, default={}, metavar='', help='additional model-specific parameters, i.e. \'{"nstack": 1}\'')
    parser.add_argument('--loss-type', type=json.loads, default={}, help='loss method')
    parser.add_argument('--conditional-fairness', type=int, default=0, help='conditional fairness or not')
    parser.add_argument('--parameters-tuning', type=json.loads, default={}, metavar='', help='hyperparameters tuning dict')
    parser.add_argument('--loss-options', type=json.loads, default={}, metavar='', help='loss-specific parameters, i.e. \'{"wsigma": 1}\'')
    parser.add_argument('--metric-target', type=json.loads, default=None, help='metric target names')
    parser.add_argument('--metric-target-options', type=json.loads, default=None, help='metric targets options')
    parser.add_argument('--metric-control', type=json.loads, default=None, help='metric control names')
    parser.add_argument('--metric-control-options', type=json.loads, default=None, help='metric control options')
    parser.add_argument('--metric-control-ys', type=json.loads, default=None, help='metric control names')
    parser.add_argument('--metric-control-ys-options', type=json.loads, default=None, help='metric control options')
    parser.add_argument('--resolution-high', type=int, default=None, help='image resolution height')
    parser.add_argument('--resolution-wide', type=int, default=None, help='image resolution width')
    parser.add_argument('--ndim', type=int, default=None, help='number of feature dimensions')
    parser.add_argument('--nunits', type=int, default=None, help='number of units in hidden layers')
    parser.add_argument('--dropout', type=float, default=None, help='dropout parameter')
    parser.add_argument('--length-scale', type=float, default=None, help='length scale')
    parser.add_argument('--precision', type=int, default=32, help='model precision')
    parser.add_argument('--enc-weights', type=str, default=None, help='path to encoder weights. Only for testing')
    parser.add_argument('--train-adv', action="store_true", help='train adversary also. Only for testing')
    parser.add_argument('--lam', type=float, default=None, help='lambda for closed form solvers')
    parser.add_argument('--build-kernel', type=str, default=None, help='Name of the function that build our kernel')
    parser.add_argument('--norm-kernel-input', type=misc.str2bool, default=False, help='A flag to show whether we want to normalize the input of our kernel or not')
    parser.add_argument('--log-z', type=misc.str2bool, default=True, help='A flag to show whether we want to save Z, S, Y to the log file or not')


    # ======================= Common Control Settings ================================
    parser.add_argument('--control-type', type=str, default=None, help='control method')
    parser.add_argument('--control-options', type=json.loads, default={'type':None}, metavar='', help='control type options')
    parser.add_argument('--control-criterion', type=str, default=None, help='control criterion type')
    parser.add_argument('--control-criterion-options', type=json.loads, default={}, metavar='', help='control-criterion-specific parameters')
    parser.add_argument('--rff-flag', type=misc.str2bool, default='Yes',metavar='', help='Using RFF or full kernel matrix')
    parser.add_argument('--rff-dim', type=int, default=None, help='Random Fourier Feature dimensionality')
    parser.add_argument('--evaluation-type', type=str, default="EvaluateEncoder", help='evaluate method')

    parser.add_argument('--kernel-alpha', type=str, default=None, help='kernel choice for z in dep')
    parser.add_argument('--kernel-alpha-options', type=json.loads, default={}, metavar='', help='kernel-z-specific parameters in dep')
    parser.add_argument('--kernel-beta', type=str, default=None, help='kernel choice for s in dep')
    parser.add_argument('--kernel-beta-options', type=json.loads, default={}, metavar='', help='kernel-s-specific parameters in dep')
    parser.add_argument('--kernel-alpha-lin', type=str, default=None, help='kernel choice for z in dep')
    parser.add_argument('--kernel-alpha-lin-options', type=json.loads, default={}, metavar='',
                        help='kernel-z-specific parameters in dep')
    parser.add_argument('--kernel-beta-lin', type=str, default=None, help='kernel choice for s in dep')
    parser.add_argument('--kernel-beta-lin-options', type=json.loads, default={}, metavar='',
                        help='kernel-s-specific parameters in dep')

    parser.add_argument('--kernel-z', type=str, default=None, help='kernel choice for z')
    parser.add_argument('--kernel-z-options', type=json.loads, default={}, metavar='', help='kernel-z-specific parameters')
    parser.add_argument('--kernel-x-s', type=str, default=None, help='kernel choice for x')
    parser.add_argument('--kernel-x-s-options', type=json.loads, default={}, metavar='', help='kernel-x-specific parameters')
    parser.add_argument('--kernel-x', type=str, default=None, help='kernel choice for x')
    parser.add_argument('--kernel-x-options', type=json.loads, default={}, metavar='', help='kernel-x-specific parameters')
    parser.add_argument('--kernel-y', type=str, default=None, help='kernel choice for y')
    parser.add_argument('--kernel-y-options', type=json.loads, default={}, metavar='', help='kernel-y-specific parameters')
    parser.add_argument('--kernel-s', type=str, default=None, help='kernel choice for s')
    parser.add_argument('--kernel-s-options', type=json.loads, default={}, metavar='', help='kernel-s-specific parameters')
    parser.add_argument('--control-niters', type=int, default=None, help='number of iterations for control model')
    parser.add_argument('--tau', type=float, default=None, help='Tau to trade-off target and control loss')
    parser.add_argument('--beta', type=float, default=None, help='Beta to trade-off target and laplacian')
    parser.add_argument('--eps', type=float, default=None, help='constraint value')
    parser.add_argument('--enc-update-freq', type=int, default=None, help='frequency of encoder training')
    parser.add_argument('--damping', type=float, default=None, metavar='', dest='damping',
                        help='damping factor for constrained optimization')
    parser.add_argument('--tau-lr', type=float, default=None, help='learning rate for tau')
    parser.add_argument('--dim-z', type=int, default=None, help='Embedding dimensionality')
    parser.add_argument('--gamma', type=float, default=None, help='gamma to trade-off target and kernel control loss')
    parser.add_argument('--tgt', type=str, default=None, help='Target attribute index, if applicable')
    parser.add_argument('--sens', type=str, default=None, help='Sensitive attribute index, if applicable')
    parser.add_argument('--pretrain-epochs', type=int, default=0, help='number of iterations for feature extractor and encoder models')
    parser.add_argument('--control-epoch', type=int, default=None, help='number of iterations for control model')
    parser.add_argument('--control-epoch-optnet', type=int, default=None, help='number of iterations for control model in optnet')
    parser.add_argument('--control-pretrain', type=bool, default=False,help='whether to pretrain control for validation')
    parser.add_argument('--fairness-type', type=str, default=None, help='DP or EO or EoO')
    parser.add_argument('--fairness-options', type=json.loads, default={}, metavar='', help='fairness metric options')

    # ======================= Control: HSIC Settings ================================
    parser.add_argument('--hsic-type', type=str, default=None, help='conditional or not')
    parser.add_argument('--kernel-type', type=str, default=None, help='control method')
    parser.add_argument('--kernel-options', type=json.loads, default={}, metavar='', help='kernel-specific parameters')

    # ======================= Control: HGR Settings ================================
    parser.add_argument('--hgrkde-type', type=str, default=None, help='conditional or not')
    parser.add_argument('--control-model-1', type=str, default=None, help='1st control model for HGR')
    parser.add_argument('--control-model-options-1', type=json.loads, default={}, metavar='', help='1st control-model-specific parameters')
    parser.add_argument('--control-model-2', type=str, default=None, help='2nd control model for HGR')
    parser.add_argument('--control-model-options-2', type=json.loads, default={}, metavar='', help='2nd control-model-specific parameters')

    # ======================= Control: ARL Settings ================================
    parser.add_argument('--control-model', type=str, default=None, help='control model')
    parser.add_argument('--control-model-options', type=json.loads, default={}, metavar='',
                        help='control-model-specific parameters')
    parser.add_argument('--num_adv_train_iters', type=int, default=None, help='number of iterations at test time')
    parser.add_argument('--extra-advtgt-epochs', type=int, default=0,
                        help='number of extra epochs for tuning target and adversary')
    parser.add_argument('--num_adv_train_epochs', type=int, default=None, help='number of iterations at test time')

    # ======================= Training Settings ================================
    parser.add_argument('--ngpu', type=int, default=1, help='number of gpus to use')
    parser.add_argument('--batch-size-test', type=int, default=None, help='batch size for testing')
    parser.add_argument('--batch-size-train', type=int, default=None, help='batch size for training')
    parser.add_argument('--nepochs', type=int, default=None, help='number of epochs to train')
    parser.add_argument('--epoch-number', type=int, default=None, help='epoch number')
    parser.add_argument('--nthreads', type=int, default=4, help='number of threads for data loading')
    parser.add_argument('--manual-seed', type=int, default=0, help='manual seed for randomness')
    parser.add_argument('--data-seed', type=int, default=41, help='manual seed for randomness in data')
    parser.add_argument('--check-val-every-n-epochs', type=int, default=1, help='validation every n epochs')
    parser.add_argument("--local_rank", default=0, type=int)

    # ======================= Hyperparameter Settings ===========================
    parser.add_argument('--learning-rate', type=float, default=None, help='learning rate')
    parser.add_argument('--optim-method', type=json.loads, default={}, help='the optimization routine ')
    parser.add_argument('--optim-options', type=json.loads, default={}, metavar='', help='optimizer-specific parameters, i.e. \'{"lr": 0.001}\'')
    parser.add_argument('--scheduler-method', type=str, default=None, help='cosine, step, exponential, plateau')
    parser.add_argument('--scheduler-options', type=json.loads, default={}, metavar='', help='optimizer-specific parameters')

    # ======================= Visualizer Settings ===========================
    parser.add_argument('--visualizer', type=str, default='VisualizerTensorboard', help='VisualizerTensorboard or VisualizerVisdom')
    parser.add_argument('--same-env', type=misc.str2bool, default='Yes', metavar='',help='does not add date and time to the visdom environment name')


    # ======================= Zero-Shot Models ===========================
    parser.add_argument('--range', type=str, default=None, metavar='',help='indicate the start and end of slice from the openCLIP csv file for 120 VLMs.')
    parser.add_argument('--timm-log-dir', type=str, default=None, metavar='',help='shows where to log the features extracted from timm models')
    

    config_file_copy = args.config_file
    if os.path.exists(args.config_file):
        config = configparser.ConfigParser()
        config.read([args.config_file])
        defaults = dict(config.items("Arguments"))
        parser.set_defaults(**defaults)

    args = parser.parse_args(remaining_argv)
    args.config_file = config_file_copy

    # Parse some arguments from string to bool
    args.no_progress_bar = misc.str2bool(args.no_progress_bar)

    # add date and time to the name of Visdom environment and the result
    if args.visualizer == 'VisualizerVisdom':
        if args.env == '':
            args.env = args.model_type
        if not args.same_env:
            args.env += '_' + now

    
    # # Read from queue
    # with open(args.queue_file, 'r') as f:
    #     line = f.readlines(args.line)
    #     cache_file = os.path.join('cache', args.queue_file, args.line)
    #     if os.path.exists(cache_file):
    #         quit()
    #     else:
    #         os.makedirs(cache_file)

    args.result_path = os.path.join(args.result_path, args.result_subdir)

    control_name = args.result_subdir.split("-")[0]

    dataset_specific = args.dataset
    if "gaussian" in dataset_specific.lower():
        dataset_specific += f"_alp_{args.alpha:1.3f}"
    elif dataset_specific in ["CelebAHQFeatures", "CFDFeatures", "FairFaceFeatures", "UTKFacesFeatures"]:
        dataset_specific += f"_tgt_{args.tgt}_sens_{args.sens}"
    elif dataset_specific == "FolkTables":
        state = args.dataset_options["state"]
        dataset_specific += f"_{state}_tgt_{args.tgt}_sens_{args.sens}"
    elif dataset_specific in ["HomeOffice", "HomeOfficeSwAV"]:
        dataset_specific += f"_src_{args.available_domains[0]}_tgt_{args.available_domains[1]}"

    eoo = "" if args.conditional_fairness == 0 else "EOO"
    # folder_name = f"{args.mode}_" + control_name + eoo + "_" + dataset_specific
    folder_name = f"{args.mode}_" + args.project_name + eoo + "_" + dataset_specific

    if args.tau is not None:
        folder_name += f"_tau_{args.tau:1.4f}"

    if args.eps is not None:
        folder_name += f"_eps_{args.eps:1.3f}"

    if args.damping is not None:
        folder_name += f"_damp_{args.damping:1.2f}"

    if args.beta is not None:
        folder_name += f"_beta_{args.beta:1.2f}"

    if args.alpha is not None:
        folder_name += f"_alpha_{args.alpha:1.2f}"

    folder_name += f"_seed_{args.manual_seed:1d}"

    folder_name += f"_{int(time.time())}"

    if "target" in args.model_type.keys():
        if args.model_type["target"] == "ClosedFormTgt":
            args.model_options["target"]["lam"] = args.lam
            # folder_name += f"_lam_{args.model_options['target']['lam']:1.1f}"

    if "adversary" in args.model_type:
        if args.model_type["adversary"] == "ClosedFormAdv":
            args.model_options["adversary"]["lam"] = args.lam


    if args.save_dir == "":
        args.save_dir = os.path.join(args.result_path, folder_name, "Save")
    if args.logs_dir == "":
        args.logs_dir = os.path.join(args.result_path, folder_name, "Logs")
    if args.feat_dir == "":
        args.feat_dir = os.path.join(args.result_path, folder_name, "Feat")
    if args.figs_dir == "":
        args.figs_dir = os.path.join(args.result_path, folder_name, "Figs")
    if args.main_figs_dir == "":
        args.main_figs_dir = os.path.join(args.result_path, "Figs")
    

    # refine tuple arguments: this section converts tuples that are
    #                         passed as string back to actual tuples.
    pattern = re.compile('^\(.+\)')

    for arg_name in vars(args):
        arg_value = getattr(args, arg_name)
        if isinstance(arg_value, str) and pattern.match(arg_value):
            setattr(args, arg_name, make_tuple(arg_value))
            print(arg_name, arg_value)
        elif isinstance(arg_value, dict):
            dict_changed = False
            for key, value in arg_value.items():
                if isinstance(value, str) and pattern.match(value):
                    dict_changed = True
                    arg_value[key] = make_tuple(value)
            if dict_changed:
                setattr(args, arg_name, arg_value)

    # We need to convert tau from string of integers separated by
    # commas to a list of integers
    if args.tgt is not None:
        args.tgt = [int(_) if _.isdigit() else _ for _ in args.tgt.strip().split(",")]
    if args.sens is not None:
        args.sens = [int(_) if _.isdigit() else _ for _ in args.sens.strip().split(",")]

    # If tau==0, we are not using an adversary. Why do we need to train
    # the adversary then? For some loss functions, this would save a lot
    # of time.
    if args.tau == 0:
        args.control_epoch = args.nepochs

    # Sometimes, you want to have separate learning rates for the
    # model and the adversary.
    for k, v in args.optim_options.items():
        if "lr" not in v:
            args.optim_options[k]["lr"] = args.learning_rate


    # Create the required directories
    os.makedirs(os.path.join(args.result_path, folder_name), exist_ok=True)
    os.makedirs(os.path.join(args.figs_dir), exist_ok=True)
    os.makedirs(os.path.join(args.main_figs_dir), exist_ok=True)
    os.makedirs(os.path.join(args.logs_dir), exist_ok=True)




    args.batch_size = args.batch_size_train



    """
    # ======================= Save source code ===========================
    # Save the source code as a zip file
    now = datetime.datetime.now()
    ts = "{:04d}-{:02d}-{:02d}-{:02d}-{:02d}-{:02d}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                            now.second)
    # Get your current location
    src_dir = os.getcwd()
    # Path to save the source code
    checkpoint_dir = os.path.join(args.result_path, folder_name)
    # Name of zip folder
    src_name = "src"
    # Create a temporary folder to keep the source
    dst_dir = os.path.join(checkpoint_dir, src_name)

    os.makedirs(dst_dir, exist_ok=True)

    curr_dirname = os.getcwd().split('/')[-1]
    # file formats that you want
    file_formats = ('.txt', '.py', '.ini')

    # Create a new folder, and copy the required files to it.
    for root, subdirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith(file_formats) and checkpoint_dir not in root:
                root_ = root.split(curr_dirname)[1][1:]
                src_path = os.path.join(root, f)
                dst_path = os.path.join(dst_dir, root_)
                os.makedirs(dst_path, exist_ok=True)
                # Copy the required files
                shutil.copy(src=src_path, dst=dst_path)

    # Write the time stamp
    with open(os.path.join(dst_dir, 'timestamp'), 'w') as f:
        f.write(ts+'\n')

    # Zip the folder
    shutil.make_archive(dst_dir, format='zip', root_dir=checkpoint_dir, base_dir=src_name)
    # Remove the temporary folder
    shutil.rmtree(dst_dir)
    """
    # ======================= Saved source code ===========================
    
    # Write this args to a file
    with open(os.path.join(args.result_path, folder_name, "args.txt"), 'w') as f:
        f.write(json.dumps(vars(args), indent=4))


    args.folder_full_path = os.path.join(args.result_path, folder_name)


    return args
