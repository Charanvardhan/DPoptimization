#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

'''
Taining of a WideResNet on CIFAR-10 with DP-SGD
'''

import os,json
import argparse
import yaml
from math import ceil
import torch
import torchvision
from src.models.wideresnet import WideResNet
import torchvision.transforms as transforms
from src.models.resnet import resnet20
import torch.nn as nn
import torch.optim as optim
from src.opacus_augmented.privacy_engine_augmented import PrivacyEngineAugmented
from opacus.validators import ModuleValidator
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import numpy as np
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import Subset
from itertools import product 
from opacus import GradSampleModule
# from EMA import EMA
from src.models.EMA_without_class import create_ema, update
import pickle
from src.models.augmented_grad_samplers import AugmentationMultiplicity
from math import ceil
from src.utils.utils import (init_distributed_mode,initialize_exp,bool_flag,accuracy,get_noise_from_bs,get_epochs_from_bs,print_params,)
import json
from src.models.prepare_models import prepare_data_cifar, prepare_augmult_cifar
from torch.nn.parallel import DistributedDataParallel as DDP
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP

import warnings

warnings.simplefilter("ignore")

DEFAULT_DATA_ROOT = "/path/to/CIFAR10"   # Replace with the correct default path
DEFAULT_DUMP_PATH = "/home/myid/crm54892/results/wrn"  # Replace with a valid directory path
DEFAULT_EXP_NAME = "sgdm5"


def train(
    model,
    ema,
    train_loader,
    optimizer,
    epoch,
    max_nb_steps,
    device,
    privacy_engine,
    K,
    logger,
    losses,
    train_acc,
    epsilons,
    grad_sample_gradients_norms_per_epoch,
    test_loader,
    is_main_worker,
    args,
    norms2_before_sigma,
    nb_steps,
    log_data,
):
    """
    Trains the model for one epoch. If it is the last epoch, it will stop at max_nb_steps iterations.
    If the model is being shadowed for EMA, we update the model at every step.
    """
    # nb_steps = nb_steps
    model.train()
    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)
    if is_main_worker:print(f"steps_per_epoch:{steps_per_epoch}")
    losses_epoch, train_acc_epoch, grad_sample_norms = [], [], []
    nb_examples_epoch = 0
    max_physical_batch_size_with_augnentation = (args.max_physical_batch_size if K == 0 else args.max_physical_batch_size // K)
    with BatchMemoryManager(data_loader=train_loader,max_physical_batch_size=max_physical_batch_size_with_augnentation,optimizer=optimizer) as memory_safe_data_loader:
        for i, (images, target) in enumerate(memory_safe_data_loader):
            nb_examples_epoch+=len(images)
            optimizer.zero_grad(set_to_none=True)
            images = images.to(device)
            target = target.to(device)
            assert K == args.transform
            l = len(images)
            ##Using Augmentation multiplicity
            if K:
                images_duplicates = torch.repeat_interleave(images, repeats=K, dim=0)
                target = torch.repeat_interleave(target, repeats=K, dim=0)
                transform = transforms.Compose([transforms.RandomCrop(size=(32, 32), padding=4, padding_mode="reflect"),transforms.RandomHorizontalFlip(p=0.5),])
                images = transforms.Lambda(lambda x: torch.stack([transform(x_) for x_ in x]))(images_duplicates)
                assert len(images) == args.transform * l

            # compute output
            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()

            # measure accuracy and record loss
            acc = accuracy(preds, labels)
            losses_epoch.append(loss.item())
            train_acc_epoch.append(acc)

            loss.backward()
            is_updated = not (optimizer._check_skip_next_step(pop_next=False))  # check if we are at the end of a true batch

            ## Logging gradient statistics on the main worker
            if is_main_worker:
                per_param_norms = [g.grad_sample.view(len(g.grad_sample), -1).norm(2, dim=-1) for g in model.parameters() if g.grad_sample is not None]
                per_sample_norms = (torch.stack(per_param_norms, dim=1).norm(2, dim=1).cpu().tolist())
                grad_sample_norms += per_sample_norms[:l]  # in case of poisson sampling we dont want the 0s

        
            optimizer.step()
            if is_updated:
                nb_steps += 1  # ?
                if ema:
                    update(model, ema, nb_steps)
                if is_main_worker:
                    losses.append(np.mean(losses_epoch))
                    train_acc.append(np.mean(train_acc_epoch))
                    grad_sample_gradients_norms_per_epoch.append(np.mean(grad_sample_norms))
                    losses_epoch, train_acc_epoch = [],[]
                    if nb_steps % args.freq_log == 0:
                        print(f"epoch:{epoch},step:{nb_steps}")
                        m2 = max(np.mean(norms2_before_sigma)-1/args.batch_size,0)
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "nb_steps":nb_steps,
                                    "train_acc": np.mean(train_acc[-args.freq_log :]),
                                    "loss": np.mean(losses[-args.freq_log :]),
                                    "grad_sample_gradients_norms": np.mean(grad_sample_norms),
                                    "grad_sample_gradients_norms_lowerC": np.mean(np.array(grad_sample_norms)<args.max_per_sample_grad_norm),
                                    #"norms2_before_sigma":list(norms2_before_sigma),
                                   # "grad_sample_gradients_norms_hist":list(np.histogram(grad_sample_norms,bins=np.arange(100), density=True)[0]),
                                }
                            )
                        )
                        norms2_before_sigma=[]
                        grad_sample_norms = []
                    if nb_steps % args.freq_log_val == 0:
                        test_acc_ema, train_acc_ema = (
                            test(ema, test_loader, train_loader, device)
                            if ema
                            else test(model, test_loader, train_loader, device)
                        )
                        print(f"epoch:{epoch},step:{nb_steps}")
                        logger.info(
                            "__log:"
                            + json.dumps(
                                {
                                    "test_acc_ema": test_acc_ema,
                                    "train_acc_ema": train_acc_ema,
                                }
                            )
                        )
                nb_examples_epoch=0
                if nb_steps >= max_nb_steps:
                    break
        epsilon = privacy_engine.get_epsilon(args.delta)
        if is_main_worker:
            epsilons.append(epsilon)
            log_data["training_log"].append({"epoch": epoch, "epsilon": epsilon})
        return nb_steps, norms2_before_sigma


def test(model, test_loader, train_loader, device):
    """
    Test the model on the testing set and the training set
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    losses = []
    test_top1_acc = []
    train_top1_acc = []

    with torch.no_grad():
        for images, target in test_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            losses.append(loss.item())
            test_top1_acc.append(acc)

    test_top1_avg = np.mean(test_top1_acc)

    with torch.no_grad():
        for images, target in train_loader:
            images = images.to(device)
            target = target.to(device)

            output = model(images)
            loss = criterion(output, target)
            preds = np.argmax(output.detach().cpu().numpy(), axis=1)
            labels = target.detach().cpu().numpy()
            acc = accuracy(preds, labels)

            # losses.append(loss.item())
            train_top1_acc.append(acc)
    train_top1_avg = np.mean(train_top1_acc)
    # print(f"\tTest set:"f"Loss: {np.mean(losses):.6f} "f"Acc: {top1_avg * 100:.6f} ")
    return (test_top1_avg, train_top1_avg)


def save_to_json(data, filename="training_log.json"):
    """Save data to a JSON file, appending to it if it exists."""
    if os.path.exists(filename):
        with open(filename, "r+") as file:
            existing_data = json.load(file)
            existing_data.append(data)
            file.seek(0)
            json.dump(existing_data, file, indent=4)
    else:
        with open(filename, "w") as file:
            json.dump([data], file, indent=4)


def run_single_experiment(config, exp_id):  ## for non poisson, divide bs by world size

    args = argparse.Namespace(**config)
    args.data_root = getattr(args, "data_root", DEFAULT_DATA_ROOT)
    args.dump_path = getattr(args, "dump_path", DEFAULT_DUMP_PATH)
    args.exp_name = getattr(args, "exp_name", DEFAULT_EXP_NAME)
    args.exp_id = getattr(args, "exp_id", "default_exp_id")  # Default value if exp_id is missing
    args.delta = float(getattr(args, "delta", 1e-5))  # Ensure delta is a float


    if not os.path.isdir(args.dump_path):
        os.makedirs(args.dump_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # init_distributed_mode(args)#Handle single and multi-GPU / multi-node]
    init_distributed_mode(args)
    logger = initialize_exp(args)
    # File for logging JSON output
    # log_file_path = os.path.join("/home/myid/crm54892/tan/results/", f"firstrun_output.json")
    log_data = {"experiment_id": exp_id, "hyperparameters": config, "training_log": [], "final_results": {}}
    # model = resnet20()
    model = WideResNet(16,10,4)
    model = model.to(device)
    # model.cuda()
    print_params(model)
    if args.multi_gpu:
        print("using multi GPU DPDDP")
        model = DPDDP(model)
    rank = args.global_rank
    is_main_worker = rank == 0
    weights = model.module.parameters() if args.multi_gpu else model.parameters()
    # Creating the datasets and dataloader for cifar10
    train_dataset, train_loader, test_loader = prepare_data_cifar("/home/myid/crm54892/data", args.batch_size, args.proportion)
    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer_type = config.get("optimizer", "sgd").lower()
    if optimizer_type == "sgd":
        optimizer = optim.SGD(weights, lr=args.lr, momentum=args.momentum)
    elif optimizer_type == "adam":
        optimizer = optim.Adam(weights, lr=args.lr)
    elif optimizer_type == "rmsprop":
        optimizer = optim.RMSprop(weights, lr=args.lr, momentum=args.momentum)
    else:
        raise ValueError(f"Unsupported optimizer type: {optimizer_type}")

    print(optimizer_type)
    # Creating the privacy engine
    privacy_engine = PrivacyEngineAugmented(GradSampleModule.GRAD_SAMPLERS)
    sigma = get_noise_from_bs(args.batch_size, args.ref_noise, args.ref_B)

    ##We use our PrivacyEngine Augmented to take into accoung the eventual augmentation multiplicity
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=sigma,
        max_grad_norm=args.max_per_sample_grad_norm,
        poisson_sampling=args.poisson_sampling,
        K=args.transform
    )
    ## Changes the grad samplers to work with augmentation multiplicity
    prepare_augmult_cifar(model,args.transform)
    ema = None
    # we create a shadow model
    print("shadowing de model with EMA")
    ema = create_ema(model)
    train_acc,test_acc,epsilons,losses,top1_accs,grad_sample_gradients_norms_per_step = (0, 0, [], [], [], [])
    norms2_before_sigma = []

    E = get_epochs_from_bs(args.batch_size, args.ref_nb_steps, len(train_dataset))
    if is_main_worker: print(f"E:{E},sigma:{sigma}, BATCH_SIZE:{args.batch_size}, noise_multiplier:{sigma}, EPOCHS:{E}")
    nb_steps = 0
    for epoch in range(E):
        if nb_steps >= args.ref_nb_steps:
            break
        nb_steps, norms2_before_sigma = train(
            model,
            ema,
            train_loader,
            optimizer,
            epoch,
            args.ref_nb_steps,
            device,
            privacy_engine,
            args.transform,
            logger,
            losses,
            top1_accs,
            epsilons,
            grad_sample_gradients_norms_per_step,
            test_loader,
            is_main_worker,
            args,
            norms2_before_sigma,
            nb_steps,
            log_data
        )
        if is_main_worker:
            print(f"epoch:{epoch}, Current loss:{losses[-1]:.2f},nb_steps:{nb_steps}, top1_acc of model (not ema){top1_accs[-1]:.2f},average gradient norm:{grad_sample_gradients_norms_per_step[-1]:.2f}")
            epoch_log = {
            "epoch": epoch,
            "current_loss": losses[-1] if losses else None,
            "nb_steps": nb_steps,
            "top1_acc_model": top1_accs[-1] if top1_accs else None,
            "average_grad_norm": grad_sample_gradients_norms_per_step[-1] if grad_sample_gradients_norms_per_step else None,
            "epsilon": epsilons[-1] if epsilons else None
            }
            log_data["training_log"].append(epoch_log)

    if is_main_worker:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size)
        # Final evaluations and logging
        final_train_acc, final_test_acc = test(ema, test_loader, train_loader, device)
        log_data["final_results"]["ema"] = {
            "final_train_acc": final_train_acc,
            "final_test_acc": final_test_acc,
            "final_epsilon": epsilons[-1] if epsilons else None,
            "average_grad_norms": np.mean(grad_sample_gradients_norms_per_step) if grad_sample_gradients_norms_per_step else None,
            "final_epsilon": epsilons[-1] if epsilons else None
            }
    

        final_train_acc_non_ema, final_test_acc_non_ema = test(model, test_loader, train_loader, device)
        log_data["final_results"]["non_ema"] = {
            "final_train_acc": final_train_acc_non_ema,
            "final_test_acc": final_test_acc_non_ema,
            "final_epsilon": epsilons[-1] if epsilons else None
        }

    return log_data




# def parse_args():
#     parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")

#     parser.add_argument("--batch_size",default=256,type=int,help="Batch size for simulated training. It will automatically set the noise $\sigma$ s.t. $B/\sigma = B_{ref}/sigma_{ref}$")
#     parser.add_argument("--max_physical_batch_size",default=128,type=int,help="max_physical_batch_size for BatchMemoryManager",)
#     parser.add_argument("--WRN_depth",default=16,type=int)
#     parser.add_argument("--WRN_k",default=4,type=int,help="k of resnet block",)

#     parser.add_argument("--lr","--learning_rate",default=4,type=float,metavar="LR",help="initial learning rate",dest="lr",)

#     parser.add_argument("--momentum",default=0,type=float,help="SGD momentum",)
#     parser.add_argument("--experiment",default=0,type=int,help="seed for initializing training. ")
#     parser.add_argument("-c","--max_per_sample_grad_norm",type=float,default=1,metavar="C",help="Clip per-sample gradients to this norm (default 1.0)",)
#     parser.add_argument("--delta",type=float,default=1e-5,metavar="D",help="Target delta (default: 1e-5)",)
#     parser.add_argument("--ref_noise",type=float,default=3,help="reference noise used with reference batch size and number of steps to create our physical constant",)
#     parser.add_argument("--ref_B",type=int,default=4096,help="reference batch size used with reference noise and number of steps to create our physical constant",)
#     parser.add_argument("--nb_groups",type=int,default=16,help="number of groups for the group norms",)
#     parser.add_argument("--ref_nb_steps",default=2500,type=int,help="reference number of steps used with reference noise and batch size to create our physical constant",)
#     parser.add_argument("--data_root",type=str,default="",help="Where CIFAR10 is/will be stored",)
#     parser.add_argument("--dump_path",type=str,default="/home/myid/crm54892/tan/results/",help="Where results will be stored",)
#     parser.add_argument("--transform",type=int,default=0,help="using augmentation multiplicity",)

#     parser.add_argument("--freq_log", type=int, default=20, help="every each freq_log steps, we log",)

#     parser.add_argument("--freq_log_val",type=int,default=100,help="every each freq_log steps, we log val and ema acc",)

#     parser.add_argument("--poisson_sampling",type=bool_flag,default=True,help="using Poisson sampling",)


#     parser.add_argument("--proportion",default=1,type=float,help="proportion of the training set to use for training",)

#     parser.add_argument("--exp_name", type=str, default="bypass")

#     parser.add_argument("--init", type=int, default=0)
#     parser.add_argument("--order1", type=int, default=0)
#     parser.add_argument("--order2", type=int, default=0)

#     parser.add_argument("--local_rank", type=int, default=-1)
#     parser.add_argument("--master_port", type=int, default=-1)
#     parser.add_argument("--debug_slurm", type=bool_flag, default=False)
#     return parser.parse_args()

def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 DP Training")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the YAML configuration file")
    return parser.parse_args()


# def load_config(config_file):
#     """Loads hyperparameters from a YAML config file, allowing for lists of values for tuning."""
#     with open(config_file, 'r') as file:
#         config = yaml.safe_load(file)
#     return config


def load_config(config_file):
    """Load specific runs from a YAML config file."""
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    return config.get('runs', {})


def generate_combinations(config):
    """Generates all combinations of hyperparameters from a config with lists of values."""
    keys, values = zip(*config.items())
    all_combinations = [dict(zip(keys, v)) for v in product(*values)]
    return all_combinations


def main():
    config_file = "config_sgdm5.yaml"  # Path to YAML file
    num_repeats = 5  # Number of times to repeat each experiment
    all_configs = load_config(config_file)  # Load predefined runs
    print(f"Total specific runs to execute: {len(all_configs)} with {num_repeats} repetitions each.")

    # Store results for all experiments, including errors
    all_results = []

    # Run each specified experiment in the YAML file
    for run_name, config in all_configs.items():
        for repeat in range(num_repeats):
            try:
                exp_id = f"{run_name}_repeat{repeat+1}"
                print(f"Running experiment {exp_id} with config: {config}")
                result = run_single_experiment(config, exp_id=exp_id)
                all_results.append(result)
            except RuntimeError as e:
                error_message = str(e)
                print(f"Error in experiment {exp_id}: {error_message}")
                # Log error in result
                all_results.append({
                    "experiment_id": exp_id,
                    "hyperparameters": config,
                    "error": error_message
                })
                continue

    # Save all results into a single JSON file (including errors)
    output_file = os.path.join(DEFAULT_DUMP_PATH, f"{DEFAULT_EXP_NAME}_all_experiments.json")
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=4)
    print(f"All experiment results, including errors, saved to {output_file}")

if __name__ == "__main__":
    main()


# batch_size: [256, 128]
# lr: [0.05, 0.01, 0.001,0.05, 0.0001]
# momentum: [0.0, 0.9]
# max_per_sample_grad_norm: [1.0]
# delta: [1e-5]
# ref_noise: [3.0]
# ref_B: [4096]
# data_root: "/path/to/CIFAR10"
# dump_path: "/path/to/results"
# transform: [0]
# freq_log: [20]
# freq_log_val: [100]
# poisson_sampling: [false]
# proportion: [1.0]
# exp_name: "exp_run"
# ema: [False]