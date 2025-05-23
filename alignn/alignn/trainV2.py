"""Module for enhanced training script with MAE reporting and normalized loss."""

from torch.nn.parallel import DistributedDataParallel as DDP
from functools import partial
from typing import Any, Dict, Union
import torch
import random
from sklearn.metrics import mean_absolute_error
import pickle as pk
import numpy as np
from torch import nn
from alignn.data import get_train_val_loaders
from alignn.config import TrainingConfig
from alignn.models.alignn_atomwise import ALIGNNAtomWise
from alignn.models.ealignn_atomwise import eALIGNNAtomWise
from alignn.models.alignn import ALIGNN
from jarvis.db.jsonutils import dumpjson
import json
import pprint
import os
import warnings
import time
from sklearn.metrics import roc_auc_score
from alignn.utils import (
    group_decay,
    setup_optimizer,
    print_train_val_loss,
)
import dgl
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)

figlet_alignn = """
    _    _     ___ ____ _   _ _   _
   / \  | |   |_ _/ ___| \ | | \ | |
  / _ \ | |    | | |  _|  \| |  \| |
 / ___ \| |___ | | |_| | |\  | |\  |
/_/   \_\_____|___\____|_| \_|_| \_|
            Enhanced v2.0
"""


def compute_mae_loss(predictions, targets, criterion=None):
    """Compute MAE between predictions and targets."""
    if criterion is None:
        return torch.abs(predictions - targets).mean()
    else:
        # If using L1Loss, it's already MAE
        if isinstance(criterion, nn.L1Loss):
            return criterion(predictions, targets)
        # If using MSE, compute MAE separately
        else:
            return torch.abs(predictions - targets).mean()


def print_train_val_loss_enhanced(
    epoch,
    running_loss,
    running_loss1,
    running_loss2,
    running_loss3,
    running_loss4,
    running_loss5,
    val_loss,
    val_loss1,
    val_loss2,
    val_loss3,
    val_loss4,
    val_loss5,
    train_ep_time,
    val_ep_time,
    train_mae_dict=None,
    val_mae_dict=None,
    saving_msg="",
    normalized=False,
):
    """Enhanced printing function with MAE reporting."""
    print(f"\n{'=' * 80}")
    print(f"Epoch {epoch} Summary:")
    print(f"{'=' * 80}")

    # Time information
    print(f"Time - Train: {train_ep_time:.2f}s, Val: {val_ep_time:.2f}s")

    # Loss information
    if normalized:
        loss_label = "Loss (avg per sample)"
        # When normalized, these are average losses per sample
        # To get total dataset loss, multiply by dataset size
    else:
        loss_label = "Loss (sum over batches)"

    print(f"\n{loss_label}:")
    print(f"  Total    - Train: {running_loss:.4f}, Val: {val_loss:.4f}")

    if running_loss1 > 0:
        print(f"  Graph    - Train: {running_loss1:.4f}, Val: {val_loss1:.4f}")
    if running_loss2 > 0:
        print(f"  Atomwise - Train: {running_loss2:.4f}, Val: {val_loss2:.4f}")
    if running_loss3 > 0:
        print(f"  Gradient - Train: {running_loss3:.4f}, Val: {val_loss3:.4f}")
    if running_loss4 > 0:
        print(f"  Stress   - Train: {running_loss4:.4f}, Val: {val_loss4:.4f}")
    if running_loss5 > 0:
        print(f"  Additional - Train: {running_loss5:.4f}, Val: {val_loss5:.4f}")

    # MAE information
    if train_mae_dict or val_mae_dict:
        print("\nMAE:")
        if train_mae_dict and val_mae_dict:
            for key in train_mae_dict:
                if (
                    train_mae_dict[key] is not None
                    and val_mae_dict.get(key) is not None
                ):
                    print(
                        f"  {key:10s} - Train: {train_mae_dict[key]:.4f}, Val: {val_mae_dict[key]:.4f}"
                    )

    if saving_msg:
        print(f"\n{saving_msg}")
    print(f"{'=' * 80}")


def calculate_mae_from_results(results, target_type="out"):
    """Calculate MAE from a list of result dictionaries."""
    targets = []
    predictions = []

    for res in results:
        if f"target_{target_type}" in res and f"pred_{target_type}" in res:
            target = res[f"target_{target_type}"]
            pred = res[f"pred_{target_type}"]

            # Flatten if needed
            if isinstance(target, list):
                if len(target) > 0 and isinstance(target[0], list):
                    target = [item for sublist in target for item in sublist]
                targets.extend(target)
            else:
                targets.append(target)

            if isinstance(pred, list):
                if len(pred) > 0 and isinstance(pred[0], list):
                    pred = [item for sublist in pred for item in sublist]
                predictions.extend(pred)
            else:
                predictions.append(pred)

    if len(targets) > 0 and len(predictions) > 0:
        return mean_absolute_error(targets, predictions)
    return None


def train_dgl_custom(
    config: Union[TrainingConfig, Dict[str, Any]],
    model: nn.Module = None,
    train_val_test_loaders=[],
    rank=0,
    world_size=0,
):
    """Enhanced training entry point with MAE reporting and normalized loss."""

    # Check for custom flags
    report_mae = getattr(config, "report_mae", False)
    normalize_loss = getattr(config, "normalize_loss", False)

    if rank == 0:
        if type(config) is dict:
            try:
                print("Trying to convert dictionary.")
                config = TrainingConfig(**config)
            except Exception as exp:
                print("Check", exp)
    print("config:", config.dict())
    print(f"\nEnhanced Training Mode:")
    print(f"  Report MAE: {report_mae}")
    print(f"  Normalize Loss: {normalize_loss}")

    if not os.path.exists(config.output_dir):
        os.makedirs(config.output_dir)

    classification = False
    tmp = config.dict()
    f = open(os.path.join(config.output_dir, "config.json"), "w")
    f.write(json.dumps(tmp, indent=4))
    f.close()
    global tmp_output_dir
    tmp_output_dir = config.output_dir
    pprint.pprint(tmp)

    if config.classification_threshold is not None:
        classification = True

    TORCH_DTYPES = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat": torch.bfloat16,
    }
    torch.set_default_dtype(TORCH_DTYPES[config.dtype])

    line_graph = False
    if config.compute_line_graph > 0:
        line_graph = True

    if world_size > 1:
        use_ddp = True
    else:
        use_ddp = False
        device = "cpu"
        if torch.cuda.is_available():
            device = torch.device("cuda")

    if not train_val_test_loaders:
        (
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ) = get_train_val_loaders(
            dataset=config.dataset,
            target=config.target,
            n_train=config.n_train,
            n_val=config.n_val,
            n_test=config.n_test,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            test_ratio=config.test_ratio,
            batch_size=config.batch_size,
            atom_features=config.atom_features,
            neighbor_strategy=config.neighbor_strategy,
            standardize=config.atom_features != "cgcnn",
            line_graph=line_graph,
            id_tag=config.id_tag,
            pin_memory=config.pin_memory,
            workers=config.num_workers,
            save_dataloader=config.save_dataloader,
            use_canonize=config.use_canonize,
            filename=config.filename,
            cutoff=config.cutoff,
            max_neighbors=config.max_neighbors,
            output_features=config.model.output_features,
            classification_threshold=config.classification_threshold,
            target_multiplication_factor=config.target_multiplication_factor,
            standard_scalar_and_pca=config.standard_scalar_and_pca,
            keep_data_order=config.keep_data_order,
            output_dir=config.output_dir,
            use_lmdb=config.use_lmdb,
            dtype=config.dtype,
        )
    else:
        train_loader = train_val_test_loaders[0]
        val_loader = train_val_test_loaders[1]
        test_loader = train_val_test_loaders[2]
        prepare_batch = train_val_test_loaders[3]

    # Get dataset sizes for normalization
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset) if test_loader else 0

    if normalize_loss:
        print(f"\nDataset sizes for normalization:")
        print(f"  Train: {train_size}")
        print(f"  Val: {val_size}")
        print(f"  Test: {test_size}")

    if use_ddp:
        device = torch.device(f"cuda:{rank}")
    prepare_batch = partial(prepare_batch, device=device)

    if classification:
        config.model.classification = True

    _model = {
        "alignn_atomwise": ALIGNNAtomWise,
        "ealignn_atomwise": eALIGNNAtomWise,
        "alignn": ALIGNN,
    }

    if config.random_seed is not None:
        random.seed(config.random_seed)
        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)
        torch.cuda.manual_seed_all(config.random_seed)
        try:
            import torch_xla.core.xla_model as xm

            xm.set_rng_state(config.random_seed)
        except ImportError:
            pass
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(config.random_seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = str(":4096:8")
        torch.use_deterministic_algorithms(True)

    if model is None:
        net = _model.get(config.model.name)(config.model)
    else:
        net = model

    print(figlet_alignn)
    print("Model parameters", sum(p.numel() for p in net.parameters()))
    print("CUDA available", torch.cuda.is_available())
    print("CUDA device count", int(torch.cuda.device_count()))

    try:
        gpu_stats = torch.cuda.get_device_properties(0)
        max_memory = round(gpu_stats.total_memory / 1024 / 1024 / 1024, 3)
        from platform import system as platform_system

        platform_system = platform_system()
        statistics = (
            f"   GPU: {gpu_stats.name}. Max memory: {max_memory} GB"
            + f". Platform = {platform_system}.\n"
            f"   Pytorch: {torch.__version__}. CUDA = "
            + f"{gpu_stats.major}.{gpu_stats.minor}."
            + f" CUDA Toolkit = {torch.version.cuda}.\n"
        )
        print(statistics)
    except Exception:
        pass

    net.to(device)
    if use_ddp:
        net = DDP(net, device_ids=[rank], find_unused_parameters=True)

    params = group_decay(net)
    optimizer = setup_optimizer(params, config)

    if config.scheduler == "none":
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1.0)
    elif config.scheduler == "onecycle":
        steps_per_epoch = len(train_loader)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=config.learning_rate,
            epochs=config.epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.3,
        )
    elif config.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer)

    if "alignn_" in config.model.name:
        best_loss = np.inf
        criterion = nn.L1Loss()
        if classification:
            criterion = nn.NLLLoss()

        params = group_decay(net)
        optimizer = setup_optimizer(params, config)

        history_train = []
        history_val = []
        history_train_mae = []
        history_val_mae = []

        # Use progress bar for epochs if rank 0
        epoch_iter = range(config.epochs)
        if rank == 0:
            epoch_iter = tqdm(epoch_iter, desc="Training Progress")

        for e in epoch_iter:
            train_init_time = time.time()
            running_loss = 0
            running_loss1 = 0
            running_loss2 = 0
            running_loss3 = 0
            running_loss4 = 0
            running_loss5 = 0
            train_result = []

            # Training loop with optional progress bar
            train_iter = train_loader
            if rank == 0 and report_mae:
                train_iter = tqdm(
                    train_loader, desc=f"Epoch {e + 1} Train", leave=False
                )

            for dats, jid in zip(train_iter, train_loader.dataset.ids):
                info = {}
                optimizer.zero_grad()

                if config.compute_line_graph > 0:
                    result = net(
                        [
                            dats[0].to(device),
                            dats[1].to(device),
                            dats[2].to(device),
                        ]
                    )
                else:
                    result = net([dats[0].to(device), dats[1].to(device)])

                info["target_out"] = []
                info["pred_out"] = []
                info["target_atomwise_pred"] = []
                info["pred_atomwise_pred"] = []
                info["target_grad"] = []
                info["pred_grad"] = []
                info["target_stress"] = []
                info["pred_stress"] = []
                info["target_additional"] = []
                info["pred_additional"] = []

                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                loss5 = 0

                batch_size = dats[0].batch_size

                if config.model.output_features is not None:
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"],
                        dats[-1].to(device),
                    )
                    if normalize_loss:
                        loss1 = loss1 * batch_size / train_size

                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = result["out"].cpu().detach().numpy().tolist()
                    running_loss1 += loss1.item()

                if (
                    config.model.atomwise_output_features > 0
                    and config.model.atomwise_weight != 0
                ):
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    if normalize_loss:
                        loss2 = loss2 * batch_size / train_size

                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )
                    running_loss2 += loss2.item()

                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    if normalize_loss:
                        loss3 = loss3 * batch_size / train_size

                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = result["grad"].cpu().detach().numpy().tolist()
                    running_loss3 += loss3.item()

                if config.model.stresswise_weight != 0:
                    targ_stress = torch.stack(
                        [gg.ndata["stresses"][0] for gg in dgl.unbatch(dats[0])]
                    ).to(device)
                    pred_stress = result["stresses"]

                    loss4 = config.model.stresswise_weight * criterion(
                        pred_stress.to(device),
                        targ_stress.to(device),
                    )
                    if normalize_loss:
                        loss4 = loss4 * batch_size / train_size

                    info["target_stress"] = targ_stress.cpu().numpy().tolist()
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )
                    running_loss4 += loss4.item()

                if config.model.additional_output_weight != 0:
                    additional_dat = [
                        gg.ndata["additional"][0] for gg in dgl.unbatch(dats[0])
                    ]
                    targ = torch.stack(additional_dat).to(device)

                    loss5 = config.model.additional_output_weight * criterion(
                        result["additional"].to(device),
                        targ,
                    )
                    if normalize_loss:
                        loss5 = loss5 * batch_size / train_size

                    info["target_additional"] = targ.cpu().numpy().tolist()
                    info["pred_additional"] = (
                        result["additional"].cpu().detach().numpy().tolist()
                    )
                    running_loss5 += loss5.item()

                train_result.append(info)
                loss = loss1 + loss2 + loss3 + loss4 + loss5
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            scheduler.step()
            train_final_time = time.time()
            train_ep_time = train_final_time - train_init_time

            # Calculate train MAE if requested
            train_mae_dict = {}
            if report_mae:
                train_mae_dict["graph"] = calculate_mae_from_results(
                    train_result, "out"
                )
                train_mae_dict["atomwise"] = calculate_mae_from_results(
                    train_result, "atomwise_pred"
                )
                train_mae_dict["gradient"] = calculate_mae_from_results(
                    train_result, "grad"
                )
                train_mae_dict["stress"] = calculate_mae_from_results(
                    train_result, "stress"
                )
                train_mae_dict["additional"] = calculate_mae_from_results(
                    train_result, "additional"
                )
                history_train_mae.append(train_mae_dict)

            history_train.append(
                [
                    running_loss,
                    running_loss1,
                    running_loss2,
                    running_loss3,
                    running_loss4,
                    running_loss5,
                ]
            )

            dumpjson(
                filename=os.path.join(config.output_dir, "history_train.json"),
                data=history_train,
            )
            if report_mae:
                dumpjson(
                    filename=os.path.join(config.output_dir, "history_train_mae.json"),
                    data=history_train_mae,
                )

            # Validation loop
            val_loss = 0
            val_loss1 = 0
            val_loss2 = 0
            val_loss3 = 0
            val_loss4 = 0
            val_loss5 = 0
            val_result = []
            val_init_time = time.time()

            val_iter = val_loader
            if rank == 0 and report_mae:
                val_iter = tqdm(val_loader, desc=f"Epoch {e + 1} Val", leave=False)

            for dats, jid in zip(val_iter, val_loader.dataset.ids):
                info = {}
                info["id"] = jid
                optimizer.zero_grad()

                if config.compute_line_graph > 0:
                    result = net(
                        [
                            dats[0].to(device),
                            dats[1].to(device),
                            dats[2].to(device),
                        ]
                    )
                else:
                    result = net([dats[0].to(device), dats[1].to(device)])

                info["target_out"] = []
                info["pred_out"] = []
                info["target_atomwise_pred"] = []
                info["pred_atomwise_pred"] = []
                info["target_grad"] = []
                info["pred_grad"] = []
                info["target_stress"] = []
                info["pred_stress"] = []

                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0
                loss5 = 0

                batch_size = dats[0].batch_size

                if config.model.output_features is not None:
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"], dats[-1].to(device)
                    )
                    if normalize_loss:
                        loss1 = loss1 * batch_size / val_size

                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = result["out"].cpu().detach().numpy().tolist()
                    val_loss1 += loss1.item()

                if (
                    config.model.atomwise_output_features > 0
                    and config.model.atomwise_weight != 0
                ):
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    if normalize_loss:
                        loss2 = loss2 * batch_size / val_size

                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )
                    val_loss2 += loss2.item()

                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    if normalize_loss:
                        loss3 = loss3 * batch_size / val_size

                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = result["grad"].cpu().detach().numpy().tolist()
                    val_loss3 += loss3.item()

                if config.model.stresswise_weight != 0:
                    targ_stress = torch.stack(
                        [gg.ndata["stresses"][0] for gg in dgl.unbatch(dats[0])]
                    ).to(device)
                    pred_stress = result["stresses"]

                    loss4 = config.model.stresswise_weight * criterion(
                        pred_stress.to(device),
                        targ_stress.to(device),
                    )
                    if normalize_loss:
                        loss4 = loss4 * batch_size / val_size

                    info["target_stress"] = targ_stress.cpu().numpy().tolist()
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )
                    val_loss4 += loss4.item()

                if config.model.additional_output_weight != 0:
                    additional_dat = [
                        gg.ndata["additional"][0] for gg in dgl.unbatch(dats[0])
                    ]
                    targ = torch.stack(additional_dat).to(device)

                    loss5 = config.model.additional_output_weight * criterion(
                        result["additional"].to(device),
                        targ,
                    )
                    if normalize_loss:
                        loss5 = loss5 * batch_size / val_size

                    info["target_additional"] = targ.cpu().numpy().tolist()
                    info["pred_additional"] = (
                        result["additional"].cpu().detach().numpy().tolist()
                    )
                    val_loss5 += loss5.item()

                loss = loss1 + loss2 + loss3 + loss4 + loss5
                val_result.append(info)
                val_loss += loss.item()

            val_fin_time = time.time()
            val_ep_time = val_fin_time - val_init_time

            # Calculate val MAE if requested
            val_mae_dict = {}
            if report_mae:
                val_mae_dict["graph"] = calculate_mae_from_results(val_result, "out")
                val_mae_dict["atomwise"] = calculate_mae_from_results(
                    val_result, "atomwise_pred"
                )
                val_mae_dict["gradient"] = calculate_mae_from_results(
                    val_result, "grad"
                )
                val_mae_dict["stress"] = calculate_mae_from_results(
                    val_result, "stress"
                )
                val_mae_dict["additional"] = calculate_mae_from_results(
                    val_result, "additional"
                )
                history_val_mae.append(val_mae_dict)

            current_model_name = "current_model.pt"
            torch.save(
                net.state_dict(),
                os.path.join(config.output_dir, current_model_name),
            )

            saving_msg = ""
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_name = "best_model.pt"
                torch.save(
                    net.state_dict(),
                    os.path.join(config.output_dir, best_model_name),
                )
                saving_msg = "Saving best model"
                dumpjson(
                    filename=os.path.join(config.output_dir, "Train_results.json"),
                    data=train_result,
                )
                dumpjson(
                    filename=os.path.join(config.output_dir, "Val_results.json"),
                    data=val_result,
                )
                best_model = net

            history_val.append(
                [
                    val_loss,
                    val_loss1,
                    val_loss2,
                    val_loss3,
                    val_loss4,
                    val_loss5,
                ]
            )

            dumpjson(
                filename=os.path.join(config.output_dir, "history_val.json"),
                data=history_val,
            )
            if report_mae:
                dumpjson(
                    filename=os.path.join(config.output_dir, "history_val_mae.json"),
                    data=history_val_mae,
                )

            if rank == 0:
                if report_mae:
                    print_train_val_loss_enhanced(
                        e,
                        running_loss,
                        running_loss1,
                        running_loss2,
                        running_loss3,
                        running_loss4,
                        running_loss5,
                        val_loss,
                        val_loss1,
                        val_loss2,
                        val_loss3,
                        val_loss4,
                        val_loss5,
                        train_ep_time,
                        val_ep_time,
                        train_mae_dict=train_mae_dict,
                        val_mae_dict=val_mae_dict,
                        saving_msg=saving_msg,
                        normalized=normalize_loss,
                    )
                else:
                    print_train_val_loss(
                        e,
                        running_loss,
                        running_loss1,
                        running_loss2,
                        running_loss3,
                        running_loss4,
                        running_loss5,
                        val_loss,
                        val_loss1,
                        val_loss2,
                        val_loss3,
                        val_loss4,
                        val_loss5,
                        train_ep_time,
                        val_ep_time,
                        saving_msg=saving_msg,
                    )

        # Test evaluation
        if rank == 0 or world_size == 1:
            test_loss = 0
            test_result = []

            test_iter = test_loader
            if report_mae:
                test_iter = tqdm(test_loader, desc="Test Evaluation")

            for dats, jid in zip(test_iter, test_loader.dataset.ids):
                info = {}
                info["id"] = jid
                optimizer.zero_grad()

                if config.compute_line_graph > 0:
                    result = net(
                        [
                            dats[0].to(device),
                            dats[1].to(device),
                            dats[2].to(device),
                        ]
                    )
                else:
                    result = net([dats[0].to(device), dats[1].to(device)])

                loss1 = 0
                loss2 = 0
                loss3 = 0
                loss4 = 0

                if config.model.output_features is not None and not classification:
                    loss1 = config.model.graphwise_weight * criterion(
                        result["out"], dats[-1].to(device)
                    )
                    info["target_out"] = dats[-1].cpu().numpy().tolist()
                    info["pred_out"] = result["out"].cpu().detach().numpy().tolist()

                if config.model.atomwise_output_features > 0:
                    loss2 = config.model.atomwise_weight * criterion(
                        result["atomwise_pred"].to(device),
                        dats[0].ndata["atomwise_target"].to(device),
                    )
                    info["target_atomwise_pred"] = (
                        dats[0].ndata["atomwise_target"].cpu().numpy().tolist()
                    )
                    info["pred_atomwise_pred"] = (
                        result["atomwise_pred"].cpu().detach().numpy().tolist()
                    )

                if config.model.calculate_gradient:
                    loss3 = config.model.gradwise_weight * criterion(
                        result["grad"].to(device),
                        dats[0].ndata["atomwise_grad"].to(device),
                    )
                    info["target_grad"] = (
                        dats[0].ndata["atomwise_grad"].cpu().numpy().tolist()
                    )
                    info["pred_grad"] = result["grad"].cpu().detach().numpy().tolist()

                if config.model.stresswise_weight != 0:
                    targ_stress = torch.stack(
                        [gg.ndata["stresses"][0] for gg in dgl.unbatch(dats[0])]
                    ).to(device)
                    pred_stress = result["stresses"]

                    loss4 = config.model.stresswise_weight * criterion(
                        pred_stress.to(device),
                        targ_stress.to(device),
                    )
                    info["target_stress"] = targ_stress.cpu().numpy().tolist()
                    info["pred_stress"] = (
                        result["stresses"].cpu().detach().numpy().tolist()
                    )

                test_result.append(info)
                loss = loss1 + loss2 + loss3 + loss4
                if not classification:
                    test_loss += loss.item()

            print(f"Test Loss: {test_loss:.4f}")

            # Calculate test MAE if requested
            if report_mae:
                test_mae_dict = {}
                test_mae_dict["graph"] = calculate_mae_from_results(test_result, "out")
                test_mae_dict["atomwise"] = calculate_mae_from_results(
                    test_result, "atomwise_pred"
                )
                test_mae_dict["gradient"] = calculate_mae_from_results(
                    test_result, "grad"
                )
                test_mae_dict["stress"] = calculate_mae_from_results(
                    test_result, "stress"
                )

                print("\nTest MAE:")
                for key, value in test_mae_dict.items():
                    if value is not None:
                        print(f"  {key}: {value:.4f}")

                dumpjson(
                    filename=os.path.join(config.output_dir, "test_mae.json"),
                    data=test_mae_dict,
                )

            dumpjson(
                filename=os.path.join(config.output_dir, "Test_results.json"),
                data=test_result,
            )

            last_model_name = "last_model.pt"
            torch.save(
                net.state_dict(),
                os.path.join(config.output_dir, last_model_name),
            )

        # Handle predictions and other post-training tasks (same as original)
        if rank == 0 or world_size == 1:
            if config.write_predictions and classification:
                best_model.eval()
                f = open(
                    os.path.join(config.output_dir, "prediction_results_test_set.csv"),
                    "w",
                )
                f.write("id,target,prediction\n")
                targets = []
                predictions = []
                with torch.no_grad():
                    ids = test_loader.dataset.ids
                    for dat, id in zip(test_loader, ids):
                        g, lg, lat, target = dat
                        out_data = best_model(
                            [g.to(device), lg.to(device), lat.to(device)]
                        )["out"]
                        top_p, top_class = torch.topk(torch.exp(out_data), k=1)
                        target = int(target.cpu().numpy().flatten().tolist()[0])
                        f.write("%s, %d, %d\n" % (id, (target), (top_class)))
                        targets.append(target)
                        predictions.append(
                            top_class.cpu().numpy().flatten().tolist()[0]
                        )
                f.close()

                print(
                    "Test ROCAUC:",
                    roc_auc_score(np.array(targets), np.array(predictions)),
                )

            if (
                config.write_predictions
                and not classification
                and config.model.output_features > 1
            ):
                best_model.eval()
                mem = []
                with torch.no_grad():
                    ids = test_loader.dataset.ids
                    for dat, id in zip(test_loader, ids):
                        g, lg, lat, target = dat
                        out_data = best_model(
                            [g.to(device), lg.to(device), lat.to(device)]
                        )["out"]
                        out_data = out_data.detach().cpu().numpy().tolist()
                        if config.standard_scalar_and_pca:
                            sc = pk.load(open("sc.pkl", "rb"))
                            out_data = list(
                                sc.transform(np.array(out_data).reshape(1, -1))[0]
                            )
                        target = target.cpu().numpy().flatten().tolist()
                        info = {}
                        info["id"] = id
                        info["target"] = target
                        info["predictions"] = out_data
                        mem.append(info)
                dumpjson(
                    filename=os.path.join(
                        config.output_dir, "multi_out_predictions.json"
                    ),
                    data=mem,
                )

            if (
                config.write_predictions
                and not classification
                and config.model.output_features == 1
                and config.model.gradwise_weight == 0
            ):
                best_model.eval()
                f = open(
                    os.path.join(config.output_dir, "prediction_results_test_set.csv"),
                    "w",
                )
                f.write("id,target,prediction\n")
                targets = []
                predictions = []
                with torch.no_grad():
                    ids = test_loader.dataset.ids
                    for dat, id in zip(test_loader, ids):
                        g, lg, lat, target = dat
                        out_data = best_model(
                            [g.to(device), lg.to(device), lat.to(device)]
                        )["out"]
                        out_data = out_data.cpu().numpy().tolist()
                        if config.standard_scalar_and_pca:
                            sc = pk.load(
                                open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                            )
                            out_data = sc.transform(np.array(out_data).reshape(-1, 1))[
                                0
                            ][0]
                        target = target.cpu().numpy().flatten().tolist()
                        if len(target) == 1:
                            target = target[0]
                        f.write("%s, %6f, %6f\n" % (id, target, out_data))
                        targets.append(target)
                        predictions.append(out_data)
                f.close()

                print(
                    "Test MAE:",
                    mean_absolute_error(np.array(targets), np.array(predictions)),
                )

                # Train set predictions
                best_model.eval()
                f = open(
                    os.path.join(config.output_dir, "prediction_results_train_set.csv"),
                    "w",
                )
                f.write("target,prediction\n")
                targets = []
                predictions = []
                with torch.no_grad():
                    ids = train_loader.dataset.ids
                    for dat, id in zip(train_loader, ids):
                        g, lg, lat, target = dat
                        out_data = best_model(
                            [g.to(device), lg.to(device), lat.to(device)]
                        )["out"]
                        out_data = out_data.cpu().numpy().tolist()
                        if config.standard_scalar_and_pca:
                            sc = pk.load(
                                open(os.path.join(tmp_output_dir, "sc.pkl"), "rb")
                            )
                            out_data = sc.transform(np.array(out_data).reshape(-1, 1))[
                                0
                            ][0]
                        target = target.cpu().numpy().flatten().tolist()
                        for ii, jj in zip(target, out_data):
                            f.write("%6f, %6f\n" % (ii, jj))
                            targets.append(ii)
                            predictions.append(jj)
                f.close()

            if config.use_lmdb:
                print("Closing LMDB.")
                train_loader.dataset.close()
                val_loader.dataset.close()
                test_loader.dataset.close()


# Alias for backward compatibility
train_dgl = train_dgl_custom

if __name__ == "__main__":
    config = TrainingConfig(
        random_seed=123, epochs=10, n_train=32, n_val=32, batch_size=16
    )
    history = train_dgl(config)
