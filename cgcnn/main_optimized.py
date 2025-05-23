import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
import datetime
import pickle
import json
import csv

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR, CosineAnnealingLR, ExponentialLR

# Add new imports for Optuna and LMDB
import optuna
from optuna.trial import TrialState
import lmdb

from cgcnn.data import CIFData
from cgcnn.data import collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

parser = argparse.ArgumentParser(
    description="Crystal Graph Convolutional Neural Networks"
)
parser.add_argument(
    "data_options",
    metavar="OPTIONS",
    nargs="+",
    help="dataset options, started with the path to root dir, then other options",
)
parser.add_argument(
    "--task",
    choices=["regression", "classification"],
    default="regression",
    help="complete a regression or classification task (default: regression)",
)
parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA")
parser.add_argument(
    "-j",
    "--workers",
    default=0,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 0)",
)
parser.add_argument(
    "--epochs",
    default=30,
    type=int,
    metavar="N",
    help="number of total epochs to run (default: 30)",
)
parser.add_argument(
    "--start-epoch",
    default=0,
    type=int,
    metavar="N",
    help="manual epoch number (useful on restarts)",
)
parser.add_argument(
    "-b",
    "--batch-size",
    default=256,
    type=int,
    metavar="N",
    help="mini-batch size (default: 256)",
)
parser.add_argument(
    "--lr",
    "--learning-rate",
    default=0.01,
    type=float,
    metavar="LR",
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--lr-milestones",
    default=[100],
    nargs="+",
    type=int,
    metavar="N",
    help="milestones for scheduler (default: [100])",
)
parser.add_argument("--momentum", default=0.9, type=float, metavar="M", help="momentum")
parser.add_argument(
    "--weight-decay",
    "--wd",
    default=0,
    type=float,
    metavar="W",
    help="weight decay (default: 0)",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="print frequency (default: 10)",
)
parser.add_argument(
    "--resume",
    default="",
    type=str,
    metavar="PATH",
    help="path to latest checkpoint (default: none)",
)
train_group = parser.add_mutually_exclusive_group()
train_group.add_argument(
    "--train-ratio",
    default=None,
    type=float,
    metavar="N",
    help="number of training data to be loaded (default none)",
)
train_group.add_argument(
    "--train-size",
    default=None,
    type=int,
    metavar="N",
    help="number of training data to be loaded (default none)",
)
valid_group = parser.add_mutually_exclusive_group()
valid_group.add_argument(
    "--val-ratio",
    default=0.1,
    type=float,
    metavar="N",
    help="percentage of validation data to be loaded (default 0.1)",
)
valid_group.add_argument(
    "--val-size",
    default=None,
    type=int,
    metavar="N",
    help="number of validation data to be loaded (default 1000)",
)
test_group = parser.add_mutually_exclusive_group()
test_group.add_argument(
    "--test-ratio",
    default=0.1,
    type=float,
    metavar="N",
    help="percentage of test data to be loaded (default 0.1)",
)
test_group.add_argument(
    "--test-size",
    default=None,
    type=int,
    metavar="N",
    help="number of test data to be loaded (default 1000)",
)

parser.add_argument(
    "--optim",
    default="SGD",
    type=str,
    metavar="SGD",
    help="choose an optimizer, SGD or Adam, (default: SGD)",
)
parser.add_argument(
    "--atom-fea-len",
    default=64,
    type=int,
    metavar="N",
    help="number of hidden atom features in conv layers",
)
parser.add_argument(
    "--h-fea-len",
    default=128,
    type=int,
    metavar="N",
    help="number of hidden features after pooling",
)
parser.add_argument(
    "--n-conv", default=3, type=int, metavar="N", help="number of conv layers"
)
parser.add_argument(
    "--n-h",
    default=1,
    type=int,
    metavar="N",
    help="number of hidden layers after pooling",
)

# Add new arguments for hyperparameter optimization and data caching
parser.add_argument(
    "--hp-opt",
    action="store_true",
    help="Enable hyperparameter optimization with Optuna",
)
parser.add_argument(
    "--n-trials",
    default=150,
    type=int,
    help="Number of Optuna trials for hyperparameter optimization",
)
parser.add_argument(
    "--study-name", default="cgcnn_study", type=str, help="Name of the Optuna study"
)
parser.add_argument(
    "--trial-epochs",
    default=75,
    type=int,
    help="Number of epochs to run per trial (default: 75)",
)

# Add scheduler arguments
parser.add_argument(
    "--scheduler",
    default="step",
    type=str,
    choices=["step", "cosine", "exponential"],
    help="LR scheduler type (default: step)",
)
parser.add_argument(
    "--lr-gamma",
    default=0.1,
    type=float,
    help="Gamma for learning rate scheduler (default: 0.1)",
)
parser.add_argument(
    "--scheduler-opt",
    action="store_true",
    help="Enable scheduler optimization with Optuna",
)

# LMDB parameters (for cache creation, not used during optimization)
parser.add_argument(
    "--create-lmdb-cache",
    action="store_true",
    help="Create an LMDB cache before training (not used during hyperparameter optimization)",
)
parser.add_argument(
    "--lmdb-path", default="./lmdb_cache", type=str, help="Path to LMDB cache directory"
)
parser.add_argument(
    "--lmdb-map-size",
    default=1e11,
    type=float,
    help="Maximum size (in bytes) for LMDB database",
)

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

if args.task == "regression":
    best_mae_error = 1e10
else:
    best_mae_error = 0.0


def tensor_to_numpy(obj):
    """
    Recursively convert PyTorch tensors to NumPy arrays so they can be serialized.

    Args:
        obj: Object to convert

    Returns:
        Object with tensors converted to NumPy arrays
    """
    if isinstance(obj, torch.Tensor):
        return obj.cpu().detach().numpy()
    elif isinstance(obj, dict):
        return {k: tensor_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [tensor_to_numpy(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(tensor_to_numpy(item) for item in obj)
    else:
        return obj


def numpy_to_tensor(obj):
    """
    Recursively convert NumPy arrays back to PyTorch tensors.

    Args:
        obj: Object to convert

    Returns:
        Object with NumPy arrays converted to tensors
    """
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    elif isinstance(obj, dict):
        return {k: numpy_to_tensor(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [numpy_to_tensor(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(numpy_to_tensor(item) for item in obj)
    else:
        return obj


class LMDBCachedDataset(CIFData):
    """LMDB-cached version of CIFData that only reads, never writes."""

    def __init__(self, lmdb_path, *args, **kwargs):
        """
        Initialize LMDBCachedDataset.

        Args:
            lmdb_path (str): Path to LMDB cache
            *args, **kwargs: Arguments to pass to CIFData
        """
        # Initialize regular CIFData as a fallback
        super().__init__(*args, **kwargs)

        # Open LMDB environment in read-only mode
        self.lmdb_path = lmdb_path
        self.env = None
        try:
            self.env = lmdb.open(
                lmdb_path,
                readonly=True,  # Read-only mode
                lock=False,  # Disable locking
                max_readers=1000,  # Support many readers
                max_dbs=0,  # Use default DB
            )
            print(f"Successfully opened LMDB cache at {lmdb_path}")
        except Exception as e:
            print(f"Error opening LMDB cache: {e}")
            self.env = None

    def __getitem__(self, idx):
        """
        Get a crystal structure by index, using cache if available.

        Args:
            idx (int): Index of the crystal

        Returns:
            tuple: Crystal structure, target, and crystal ID
        """
        # Try to get from cache first
        if self.env is not None:
            try:
                # Extract ID for this index - if it exists
                cif_id = str(self.id_prop_data[idx][0])

                # Create a temporary transaction
                with self.env.begin() as txn:
                    # Try to get the cached data
                    cached_data = txn.get(cif_id.encode())

                    if cached_data is not None:
                        # Found in cache, deserialize and return
                        result = pickle.loads(cached_data)
                        return numpy_to_tensor(result)
            except Exception as e:
                # If any error occurs, fall back to original method
                print(f"Cache read error, falling back to CIFData: {e}")
                pass

        # If we get here, either no cache, cache miss, or cache error
        # Fall back to original implementation
        return super().__getitem__(idx)

    def close(self):
        """Close LMDB environment."""
        if self.env is not None:
            self.env.close()
            self.env = None

    def __del__(self):
        """Ensure environment is closed."""
        self.close()


def create_lmdb_cache(data_options, lmdb_path, map_size=1e11):
    """
    Create an LMDB cache from a CIFData dataset.

    Args:
        data_options (list): Options to pass to CIFData
        lmdb_path (str): Path to create LMDB cache
        map_size (float): Maximum size of the database
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(lmdb_path), exist_ok=True)

    # Initialize original dataset
    dataset = CIFData(*data_options)
    print(f"Creating LMDB cache for {len(dataset)} structures at {lmdb_path}")

    # Open LMDB environment for writing
    env = lmdb.open(lmdb_path, map_size=int(map_size))

    # Process all structures
    with env.begin(write=True) as txn:
        for idx in range(len(dataset)):
            try:
                # Get original structure
                result = dataset[idx]

                # Get structure ID for the key
                cif_id = str(dataset.id_prop_data[idx][0])

                # Convert tensors to numpy for serialization
                serializable_result = tensor_to_numpy(result)

                # Serialize and store
                txn.put(cif_id.encode(), pickle.dumps(serializable_result))

                # Print progress
                if (idx + 1) % 100 == 0:
                    print(f"Cached {idx + 1}/{len(dataset)} structures")

            except Exception as e:
                print(f"Error caching structure {idx}: {e}")

    # Close environment
    env.close()
    print(f"Completed LMDB cache creation at {lmdb_path}")


def train_and_evaluate(trial=None, args=None):
    """
    Train and evaluate a CGCNN model, optionally with hyperparameter optimization.

    Args:
        trial (optuna.Trial): Optuna trial object for hyperparameter optimization
        args (argparse.Namespace): Command line arguments

    Returns:
        float: Validation metric (MAE for regression, AUC for classification)
    """
    global best_mae_error

    # For timing data collection
    epoch_times = []
    trial_start_time = time.time()
    trial_id = trial.number if trial is not None else "no_trial"

    # If using Optuna, override hyperparameters
    if trial is not None:
        # Sample hyperparameters
        args.batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        args.lr = trial.suggest_float("lr", 5e-5, 5e-1, log=True)
        args.atom_fea_len = trial.suggest_categorical(
            "atom_fea_len", [32, 64, 128, 256, 512]
        )
        args.h_fea_len = trial.suggest_categorical(
            "h_fea_len", [64, 128, 256, 512, 1024]
        )
        args.n_conv = trial.suggest_int("n_conv", 1, 7)
        args.n_h = trial.suggest_int("n_h", 1, 5)
        args.optim = trial.suggest_categorical("optim", ["SGD", "Adam"])

        if args.optim == "SGD":
            args.momentum = trial.suggest_float("momentum", 0.5, 0.99)

        args.weight_decay = trial.suggest_float("weight_decay", 1e-7, 1e-2, log=True)

        # Learning rate scheduler options
        if args.scheduler_opt:
            args.scheduler = trial.suggest_categorical(
                "scheduler", ["step", "cosine", "exponential"]
            )
            if args.scheduler == "step":
                # For step scheduler, suggest milestones
                milestone1 = trial.suggest_int("lr_milestone_1", 10, 100)
                args.lr_milestones = [milestone1]

                # Optionally add a second milestone
                if trial.suggest_categorical("use_second_milestone", [True, False]):
                    milestone2 = trial.suggest_int(
                        "lr_milestone_2", milestone1 + 10, 200
                    )
                    args.lr_milestones.append(milestone2)

                args.lr_gamma = trial.suggest_float("lr_gamma", 0.1, 0.5)
            elif args.scheduler == "exponential":
                args.lr_gamma = trial.suggest_float("lr_gamma", 0.9, 0.99)

    # Create dataset - during optimization trials, use regular CIFData to avoid cache issues
    if trial is not None or not os.path.exists(args.lmdb_path):
        dataset = CIFData(*args.data_options)
    else:
        # For regular training or final model, use cached dataset if available
        dataset = LMDBCachedDataset(args.lmdb_path, *args.data_options)

    # Get data loaders
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True,
    )

    # Obtain target value normalizer
    if args.task == "classification":
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({"mean": 0.0, "std": 1.0})
    else:
        if len(dataset) < 500:
            warnings.warn(
                "Dataset has less than 500 data points. Lower accuracy is expected. "
            )
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # Build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=True if args.task == "classification" else False,
    )
    if args.cuda:
        model.cuda()

    # Define loss func and optimizer
    if args.task == "classification":
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "Adam":
        optimizer = optim.Adam(
            model.parameters(), args.lr, weight_decay=args.weight_decay
        )
    else:
        raise NameError("Only SGD or Adam is allowed as --optim")

    # Optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_mae_error = checkpoint["best_mae_error"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            normalizer.load_state_dict(checkpoint["normalizer"])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # Setup learning rate scheduler based on type
    if hasattr(args, "scheduler"):
        if args.scheduler == "step":
            scheduler = MultiStepLR(
                optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
            )
        elif args.scheduler == "cosine":
            # For Optuna, use trial epochs, otherwise full epochs
            if trial is not None:
                t_max = min(args.trial_epochs, args.epochs)
            else:
                t_max = args.epochs
            scheduler = CosineAnnealingLR(optimizer, T_max=t_max)
        elif args.scheduler == "exponential":
            scheduler = ExponentialLR(optimizer, gamma=args.lr_gamma)
    else:
        # Default scheduler from original code
        scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    # For Optuna, use fewer epochs to speed up search
    if trial is not None:
        epochs = min(args.trial_epochs, args.epochs)
    else:
        epochs = args.epochs

    # Training loop
    for epoch in range(args.start_epoch, epochs):
        # Time each epoch
        epoch_start_time = time.time()

        # Train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer, args)

        # Evaluate on validation set
        mae_error = validate(val_loader, model, criterion, normalizer, args)

        # Record epoch timing
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        epoch_times.append(
            {
                "trial_id": trial_id,
                "epoch": epoch,
                "duration": epoch_duration,
                "mae_error": float(mae_error),
                "timestamp": datetime.datetime.now().isoformat(),
            }
        )

        # Save timing data after each epoch
        if trial is not None:
            timing_file = f"trial_{trial_id}_timing.csv"
            file_exists = os.path.isfile(timing_file)
            with open(timing_file, "a", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "trial_id",
                        "epoch",
                        "duration",
                        "mae_error",
                        "timestamp",
                    ],
                )
                if not file_exists:
                    writer.writeheader()
                writer.writerow(epoch_times[-1])

        if mae_error != mae_error:
            print("Exit due to NaN")
            if trial is not None:
                return float("inf")  # For Optuna
            sys.exit(1)

        scheduler.step()

        # Remember the best mae_error and save checkpoint
        if args.task == "regression":
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)

        # Don't save checkpoints during hyperparameter search
        if trial is None:
            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_mae_error": best_mae_error,
                    "optimizer": optimizer.state_dict(),
                    "normalizer": normalizer.state_dict(),
                    "args": vars(args),
                },
                is_best,
            )

        # Report intermediate values to Optuna if using it
        if trial is not None:
            trial.report(mae_error, epoch)

            # Handle pruning based on intermediate results
            if trial.should_prune():
                # Save timing data before pruning
                trial_duration = time.time() - trial_start_time
                with open("trial_durations.csv", "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([trial_id, "pruned", epoch, trial_duration])
                raise optuna.exceptions.TrialPruned()

    # Calculate total trial duration
    trial_duration = time.time() - trial_start_time

    # Save overall trial timing data
    if trial is not None:
        with open("trial_durations.csv", "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([trial_id, "completed", epochs, trial_duration])

    # Test best model
    if trial is None:  # Skip testing during hyperparameter search
        print("---------Evaluate Model on Test Set---------------")
        best_checkpoint = torch.load("model_best.pth.tar")
        model.load_state_dict(best_checkpoint["state_dict"])
        test_error = validate(
            test_loader, model, criterion, normalizer, args, test=True
        )

    # Close LMDB environment if using cached dataset
    if hasattr(dataset, "close"):
        dataset.close()

    return best_mae_error if args.task == "regression" else best_mae_error


def train(train_loader, model, criterion, optimizer, epoch, normalizer, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    if args.task == "regression":
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    # Switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (
                Variable(input[0].cuda(non_blocking=True)),
                Variable(input[1].cuda(non_blocking=True)),
                input[2].cuda(non_blocking=True),
                [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
            )
        else:
            input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        # Normalize target
        if args.task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # Compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
        if args.task == "regression":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target
            )
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # Compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == "regression":
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        mae_errors=mae_errors,
                    )
                )
            else:
                print(
                    "Epoch: [{0}][{1}/{2}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accu {accu.val:.3f} ({accu.avg:.3f})\t"
                    "Precision {prec.val:.3f} ({prec.avg:.3f})\t"
                    "Recall {recall.val:.3f} ({recall.avg:.3f})\t"
                    "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                    "AUC {auc.val:.3f} ({auc.avg:.3f})".format(
                        epoch,
                        i,
                        len(train_loader),
                        batch_time=batch_time,
                        data_time=data_time,
                        loss=losses,
                        accu=accuracies,
                        prec=precisions,
                        recall=recalls,
                        f1=fscores,
                        auc=auc_scores,
                    )
                )


def validate(val_loader, model, criterion, normalizer, args, test=False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if args.task == "regression":
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()
    if test:
        test_targets = []
        test_preds = []
        test_cif_ids = []

    # Switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (
                    Variable(input[0].cuda(non_blocking=True)),
                    Variable(input[1].cuda(non_blocking=True)),
                    input[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input[3]],
                )
        else:
            with torch.no_grad():
                input_var = (Variable(input[0]), Variable(input[1]), input[2], input[3])
        if args.task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        # Compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # Measure accuracy and record loss
        if args.task == "regression":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target
            )
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == "regression":
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        mae_errors=mae_errors,
                    )
                )
            else:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accu {accu.val:.3f} ({accu.avg:.3f})\t"
                    "Precision {prec.val:.3f} ({prec.avg:.3f})\t"
                    "Recall {recall.val:.3f} ({recall.avg:.3f})\t"
                    "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                    "AUC {auc.val:.3f} ({auc.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        accu=accuracies,
                        prec=precisions,
                        recall=recalls,
                        f1=fscores,
                        auc=auc_scores,
                    )
                )

    if test:
        star_label = "**"
        import csv

        with open("test_results.csv", "w") as f:
            writer = csv.writer(f)
            for cif_id, target, pred in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target, pred))
    else:
        star_label = "*"
    if args.task == "regression":
        print(
            " {star} MAE {mae_errors.avg:.3f}".format(
                star=star_label, mae_errors=mae_errors
            )
        )
        return mae_errors.avg
    else:
        print(" {star} AUC {auc.avg:.3f}".format(star=star_label, auc=auc_scores))
        return auc_scores.avg


class Normalizer(object):
    """Normalize a Tensor and restore it later."""

    def __init__(self, tensor):
        """tensor is taken as a sample to calculate the mean and std"""
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor):
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor):
        return normed_tensor * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def mae(prediction, target):
    """
    Computes the mean absolute error between prediction and target

    Parameters
    ----------

    prediction: torch.Tensor (N, 1)
    target: torch.Tensor (N, 1)
    """
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction, target):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if not target_label.shape:
        target_label = np.asarray([target_label])
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average="binary"
        )
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


def main():
    global args, best_mae_error

    # Create LMDB cache if requested
    if args.create_lmdb_cache:
        print("Creating LMDB cache...")
        create_lmdb_cache(args.data_options, args.lmdb_path, args.lmdb_map_size)
        print("LMDB cache creation complete. Exiting.")
        return

    # Create timing tracking files
    if not os.path.exists("trial_durations.csv"):
        with open("trial_durations.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                ["trial_id", "status", "epochs_completed", "duration_seconds"]
            )

    # If hyperparameter optimization is enabled, run Optuna
    if args.hp_opt:
        print("Starting hyperparameter optimization with Optuna...")

        # Record start time
        overall_start_time = time.time()

        # Create a new study or load an existing one
        study_storage = f"sqlite:///{args.study_name}.db"
        study = optuna.create_study(
            study_name=args.study_name,
            storage=study_storage,
            direction="minimize" if args.task == "regression" else "maximize",
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(n_startup_trials=15, multivariate=True),
        )

        # Set up pruning (early stopping for unpromising trials) - less aggressive now
        pruner = optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=15, interval_steps=3
        )
        study.pruner = pruner

        # Run optimization
        study.optimize(
            lambda trial: train_and_evaluate(trial, args), n_trials=args.n_trials
        )

        # Record overall optimization time
        overall_duration = time.time() - overall_start_time

        # Print optimization results
        print("Hyperparameter optimization completed!")
        print(f"Number of finished trials: {len(study.trials)}")
        print(f"Total optimization time: {overall_duration:.2f} seconds")

        trial = study.best_trial
        print(f"Best trial: {trial.number}")
        print(f"Best value: {trial.value:.3f}")
        print("Best hyperparameters:")

        for key, value in trial.params.items():
            print(f"    {key}: {value}")

        # Save best hyperparameters to file
        with open("best_params.json", "w") as f:
            json.dump(trial.params, f, indent=4)

        # Also save details about all trials for analysis
        all_trials_data = []
        for i, trial in enumerate(study.trials):
            if trial.state == optuna.trial.TrialState.COMPLETE:
                trial_data = {
                    "number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "datetime_start": trial.datetime_start.isoformat(),
                    "datetime_complete": trial.datetime_complete.isoformat(),
                    "duration": (
                        trial.datetime_complete - trial.datetime_start
                    ).total_seconds(),
                }
                all_trials_data.append(trial_data)

        with open("all_trials.json", "w") as f:
            json.dump(all_trials_data, f, indent=4)

        # Save overall optimization time
        with open("optimization_summary.json", "w") as f:
            json.dump(
                {
                    "total_trials": len(study.trials),
                    "completed_trials": len(all_trials_data),
                    "pruned_trials": len(study.trials) - len(all_trials_data),
                    "best_trial": trial.number,
                    "best_value": trial.value,
                    "total_duration": overall_duration,
                    "average_trial_duration": overall_duration / len(study.trials)
                    if len(study.trials) > 0
                    else 0,
                },
                f,
                indent=4,
            )

        # Train a final model using the best hyperparameters
        print("\nTraining final model with best hyperparameters...")
        for key, value in trial.params.items():
            setattr(args, key, value)

        train_and_evaluate(args=args)

    else:
        # Run normal training without hyperparameter optimization
        train_and_evaluate(args=args)


if __name__ == "__main__":
    main()
