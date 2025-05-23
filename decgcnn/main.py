import argparse
import sys
import os
import shutil
import time
import warnings
from random import sample

import pandas as pd
import numpy as np
from sklearn import metrics
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F

from decgcnn.model import CrystalGraphConvNet
from decgcnn.data import collate_pool, get_train_val_test_loader
from decgcnn.data import CIFData


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
parser.add_argument(
    "--train-ratio",
    default=None,
    type=float,
    metavar="N",
    help="number of training data to be loaded (default none)",
)
parser.add_argument(
    "--train-size",
    default=None,
    type=int,
    metavar="N",
    help="number of training data to be loaded (default none)",
)
parser.add_argument(
    "--val-ratio",
    default=0.1,
    type=float,
    metavar="N",
    help="percentage of validation data to be loaded (default 0.1)",
)
parser.add_argument(
    "--val-size",
    default=1000,
    type=int,
    metavar="N",
    help="number of validation data to be loaded (default 1000)",
)
parser.add_argument(
    "--test-ratio",
    default=0.1,
    type=float,
    metavar="N",
    help="percentage of test data to be loaded (default 0.1)",
)
parser.add_argument(
    "--test-size",
    default=1000,
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
    default=3,
    type=int,
    metavar="N",
    help="number of hidden layers after pooling",
)

parser.add_argument("--layersKept", default=0, type=int)

args = parser.parse_args(sys.argv[1:])

args.cuda = not args.disable_cuda and torch.cuda.is_available()

# feature_list = [
#        'spacegroup_num','crystal_system_int','structural complexity per atom','structural complexity per cell','mean absolute deviation in relative bond length',
#        'max relative bond length','maximum neighbor distance variation','range neighbor distance variation','mean neighbor distance variation','avg_dev neighbor distance variation',
#        'mean absolute deviation in relative cell size','std_dev Average bond length','mean Average bond angle','std_dev Average bond angle','mean BOOP Q l=1','std_dev BOOP Q l=1',
#        'std_dev BOOP Q l=3','std_dev BOOP Q l=4','mean BOOP Q l=5','std_dev BOOP Q l=5','std_dev BOOP Q l=6','std_dev BOOP Q l=7','std_dev BOOP Q l=8','std_dev BOOP Q l=9','std_dev BOOP Q l=10',
#        'mean CN_VoronoiNN','std_dev CN_VoronoiNN','vpa']

feature_list = [
    "spacegroup_num",
    "crystal_system_int",
    "structural complexity per atom",
    "structural complexity per cell",
    "mean absolute deviation in relative bond length",
    "max relative bond length",
    "min relative bond length",
    "maximum neighbor distance variation",
    "range neighbor distance variation",
    "mean neighbor distance variation",
    "avg_dev neighbor distance variation",
    "mean absolute deviation in relative cell size",
    "mean Average bond length",
    "std_dev Average bond length",
    "mean Average bond angle",
    "std_dev Average bond angle",
    #        'mean BOOP Q l=1','std_dev BOOP Q l=1','mean BOOP Q l=2','std_dev BOOP Q l=2',
    #        'mean BOOP Q l=3','std_dev BOOP Q l=3','mean BOOP Q l=4','std_dev BOOP Q l=4',
    #        'mean BOOP Q l=5','std_dev BOOP Q l=5','mean BOOP Q l=6','std_dev BOOP Q l=6',
    #        'mean BOOP Q l=7','std_dev BOOP Q l=7','mean BOOP Q l=8','std_dev BOOP Q l=8',
    #        'mean BOOP Q l=9','std_dev BOOP Q l=9','mean BOOP Q l=10','std_dev BOOP Q l=10',
    "mean CN_VoronoiNN",
    "std_dev CN_VoronoiNN",
    "density",
    "vpa",
    "packing fraction",
    "a",
    "b",
    "c",
    "alpha",
    "beta",
    "gamma",
    "natoms",
]


if args.task == "regression":
    best_mae_error = 1e10
else:
    best_mae_error = 0.0

print(args.layersKept)


def main():
    global args, best_mae_error

    # load data
    cif_path = args.data_options[0]
    descriptor_path = args.data_options[1]
    print(cif_path, descriptor_path, len(feature_list))
    features = pd.read_csv(descriptor_path)
    features_dict = {}
    for _, row in features.iterrows():
        features_dict[str(row["material_id"])] = []
        for fea_name in features.columns:
            if fea_name in feature_list:
                features_dict[str(row["material_id"])].append(row[fea_name])
    #    print (len(features_dict.keys()))
    dataset = CIFData(cif_path, features_dict)
    collate_fn = collate_pool
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        train_size=args.train_size,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        val_size=args.val_size,
        test_ratio=args.test_ratio,
        test_size=args.test_size,
        pin_memory=args.cuda,
        return_test=True,
    )

    # obtain target value normalizer
    if args.task == "classification":
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({"mean": 0.0, "std": 1.0})
    else:
        if len(dataset) < 500:
            warnings.warn(
                "Dataset has less than 500 data points. Lower accuracy is expected. "
            )
            print(len(dataset))
            #            for i in range (len(dataset)):
            #                print (i)
            #                sample_data_list.append(dataset[i])
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        #            print (sample_data_list)
        else:
            sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]
        #        print (collate_pool(sample_data_list))
        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # build model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[2].shape[-1]
    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        human_fea_len=len(feature_list),
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=True if args.task == "classification" else False,
    )

    old_para = []
    for name, para in model.named_parameters():
        copied = para.data.clone()
        old_para.append(copied)
    old_para.reverse()
    #    print (len(old_para))
    #    for para in old_para:
    #        print (para)
    if args.cuda:
        model.cuda()

    # define loss func and optimizer
    if args.task == "classification":
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            #            args.start_epoch = checkpoint['epoch']
            #            best_mae_error = 1e+3
            #            best_mae_error = checkpoint['best_mae_error']
            model.load_state_dict(checkpoint["state_dict"])
            num_of_layers = 0
            for para in model.named_parameters():
                num_of_layers += 1
            #                print (para)
            layers = 0
            kept_para = []
            optimized_para = []
            for name, para in model.named_parameters():
                if layers < args.layersKept:
                    kept_para.append(para.data)
                    #                    mean = torch.mean(para)
                    #                    std = torch.std(para)
                    #                    para = (para - mean) / (std)
                    #                    para.data = para.detach()
                    old_para.pop()
                else:
                    ori_para = old_para.pop()
                    para.data = ori_para
                    optimized_para.append(para)
                layers += 1
            #                print (para)
            #            for name, para in model.named_parameters():
            #                print (name, para)
            #            optimizer.load_state_dict(checkpoint['optimizer'])
            #            normalizer.load_state_dict(checkpoint['normalizer'])
            print(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

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

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, normalizer)

        # evaluate on validation set
        mae_error = validate(val_loader, model, criterion, normalizer)

        if mae_error != mae_error:
            print("Exit due to NaN")
            sys.exit(1)

        scheduler.step()

        # remember the best mae_eror and save checkpoint
        if args.task == "regression":
            is_best = mae_error < best_mae_error
            best_mae_error = min(mae_error, best_mae_error)
        else:
            is_best = mae_error > best_mae_error
            best_mae_error = max(mae_error, best_mae_error)
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

    # test best model
    print("---------Evaluate Model on Test Set---------------")
    best_checkpoint = torch.load("model_best.pth.tar")
    model.load_state_dict(best_checkpoint["state_dict"])
    validate(test_loader, model, criterion, normalizer, test=True)


# for para in model.parameters():
#     print (para)


def train(train_loader, model, criterion, optimizer, epoch, normalizer):
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

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (
                Variable(input[0].cuda(non_blocking=True)),
                Variable(input[1].cuda(non_blocking=True)),
                Variable(input[2].cuda(non_blocking=True)),
                input[3].cuda(non_blocking=True),
                [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]],
            )
        else:
            input_var = (
                Variable(input[0]),
                Variable(input[1]),
                Variable(input[2]),
                input[3],
                input[4],
            )
        # normalize target
        if args.task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        # compute output
        model.cuda()
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        if args.task == "regression":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error, target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target
            )
            losses.update(loss.data.cpu()[0], target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # recover kept layers
        #        recover = 0
        #        for para in model.parameters():
        #            recover += 1
        #            if recover > args.layersKept:
        #                break
        #            para.data = kept_para[recover-1]

        # measure elapsed time
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


def validate(val_loader, model, criterion, normalizer, test=False):
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

    # switch to evaluate mode
    model.eval()

    end = time.time()
    for i, (input, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            input_var = (
                Variable(input[0].cuda(non_blocking=True), volatile=True),
                Variable(input[1].cuda(non_blocking=True), volatile=True),
                Variable(input[2].cuda(non_blocking=True), volatile=True),
                input[3].cuda(non_blocking=True),
                [crys_idx.cuda(non_blocking=True) for crys_idx in input[4]],
            )
        else:
            input_var = (
                Variable(input[0], volatile=True),
                Variable(input[1], volatile=True),
                Variable(input[2], volatile=True),
                input[3],
                input[4],
            )
        if args.task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()
        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True), volatile=True)
        else:
            target_var = Variable(target_normed, volatile=True)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
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
            losses.update(loss.data.cpu()[0], target.size(0))
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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == "regression":
                print("\t")
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
        if star_label == "**":
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


def adjust_learning_rate(optimizer, epoch, k):
    """Sets the learning rate to the initial LR decayed by 10 every k epochs"""
    assert type(k) is int
    lr = args.lr * (0.1 ** (epoch // k))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


if __name__ == "__main__":
    main()
