#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""

import os
import torch.distributed as dist
import csv
import sys
import json
import zipfile
from alignn.data import get_train_val_loaders

# Use enhanced training function
from alignn.trainV2 import train_dgl  # Enhanced version with MAE and normalized loss
from alignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse
from alignn.models.alignn_atomwise import ALIGNNAtomWise, ALIGNNAtomWiseConfig
import torch
import time
from jarvis.core.atoms import Atoms
import random
from ase.stress import voigt_6_to_full_3x3_stress
import pickle
from pathlib import Path

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Limit GPU memory usage to 65% (configurable)
    torch.cuda.set_per_process_memory_fraction(0.65, 0)
    # Print available GPU memory
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )
    print(f"Limiting to 65% of available memory")


def setup(rank=0, world_size=0, port="12356"):
    """Set up multi GPU rank."""
    # "12356"
    if port == "":
        port = str(random.randint(10000, 99999))
    if world_size > 1:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = port
        # os.environ["MASTER_PORT"] = "12355"
        # Initialize the distributed environment.
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)


def cleanup(world_size):
    """Clean up distributed process."""
    if world_size > 1:
        dist.destroy_process_group()


parser = argparse.ArgumentParser(description="Atomistic Line Graph Neural Network")
parser.add_argument(
    "--root_dir",
    default="./",
    help="Folder with id_props.csv, structure files",
)
parser.add_argument(
    "--config_name",
    default="alignn/examples/sample_data/config_example.json",
    help="Name of the config file",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)

parser.add_argument("--batch_size", default=None, help="Batch size, generally 64")

parser.add_argument("--epochs", default=None, help="Number of epochs, generally 300")

parser.add_argument(
    "--target_key",
    default="total_energy",
    help="Name of the key for graph level data such as total_energy",
)

parser.add_argument(
    "--id_key",
    default="jid",
    help="Name of the key for graph level id such as id",
)

parser.add_argument(
    "--force_key",
    default="forces",
    help="Name of key for gradient level data such as forces, (Natoms x p)",
)

parser.add_argument(
    "--atomwise_key",
    default="forces",
    help="Name of key for atomwise level data: forces, charges (Natoms x p)",
)

parser.add_argument(
    "--stresswise_key",
    default="stresses",
    help="Name of the key for stress (3x3) level data such as forces",
)

parser.add_argument(
    "--additional_output_key",
    default="additional_output",
    help="Name of the key for extra global output eg DOS",
)

parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)

parser.add_argument(
    "--restart_model_path",
    default=None,
    help="Checkpoint file path for model",
)

parser.add_argument(
    "--device",
    default=None,
    help="set device for training the model [e.g. cpu, cuda, cuda:2]",
)

# New arguments for improved functionality
parser.add_argument(
    "--dry_run",
    action="store_true",
    help="Only create LMDB cache and exit without training",
)

parser.add_argument(
    "--lmdb_cache_path",
    default=None,
    help="Path to existing LMDB cache to reuse (skips cache creation)",
)

parser.add_argument(
    "--report_mae",
    action="store_true",
    help="Report MAE at each epoch in addition to loss",
)

parser.add_argument(
    "--normalize_loss",
    action="store_true",
    help="Normalize loss by number of samples in each split",
)

parser.add_argument(
    "--cache_metadata_path",
    default=None,
    help="Path to save/load dataset metadata for cache verification",
)

parser.add_argument(
    "--skip_cache_verification",
    action="store_true",
    help="Skip LMDB cache verification (use if you're sure cache is compatible)",
)


def verify_cache_compatibility(dataset, cache_metadata_path):
    """Verify that the cached LMDB is compatible with current dataset."""
    if not os.path.exists(cache_metadata_path):
        return False

    with open(cache_metadata_path, "rb") as f:
        cached_metadata = pickle.load(f)

    # Create current metadata
    current_metadata = {
        "num_samples": len(dataset),
        "sample_ids": [d["jid"] for d in dataset],
        "has_forces": any("atomwise_grad" in d for d in dataset),
        "has_stress": any("stresses" in d for d in dataset),
        "has_atomwise": any("atomwise_target" in d for d in dataset),
    }

    # Check compatibility
    if cached_metadata["num_samples"] != current_metadata["num_samples"]:
        print(
            f"Cache mismatch: different number of samples "
            f"({cached_metadata['num_samples']} vs {current_metadata['num_samples']})"
        )
        return False

    if set(cached_metadata["sample_ids"]) != set(current_metadata["sample_ids"]):
        print("Cache mismatch: different sample IDs")
        return False

    for key in ["has_forces", "has_stress", "has_atomwise"]:
        if cached_metadata[key] != current_metadata[key]:
            print(f"Cache mismatch: different {key} status")
            return False

    print("Cache verified: compatible with current dataset")
    return True


def save_cache_metadata(dataset, cache_metadata_path):
    """Save dataset metadata for cache verification."""
    metadata = {
        "num_samples": len(dataset),
        "sample_ids": [d["jid"] for d in dataset],
        "has_forces": any("atomwise_grad" in d for d in dataset),
        "has_stress": any("stresses" in d for d in dataset),
        "has_atomwise": any("atomwise_target" in d for d in dataset),
        "creation_time": time.time(),
    }

    os.makedirs(os.path.dirname(cache_metadata_path), exist_ok=True)
    with open(cache_metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"Saved cache metadata to {cache_metadata_path}")


def train_for_folder(
    rank=0,
    world_size=0,
    root_dir="examples/sample_data",
    config_name="config.json",
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    id_key="jid",
    target_key="total_energy",
    atomwise_key="forces",
    gradwise_key="forces",
    stresswise_key="stresses",
    additional_output_key="additional_output",
    file_format="poscar",
    restart_model_path=None,
    output_dir=None,
    dry_run=False,
    lmdb_cache_path=None,
    report_mae=False,
    normalize_loss=False,
    cache_metadata_path=None,
    skip_cache_verification=False,
):
    """Train for a folder."""
    setup(rank=rank, world_size=world_size)
    print("root_dir", root_dir)

    # Load configuration
    config_dict = loadjson(config_name)
    config = TrainingConfig(**config_dict)

    # Update config with command line arguments
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)

    # Set up LMDB paths
    if lmdb_cache_path:
        config.use_lmdb = True
        # Override the default LMDB path with the provided one
        config.lmdb_path = lmdb_cache_path
        print(f"Using existing LMDB cache at: {lmdb_cache_path}")

    if cache_metadata_path is None:
        cache_metadata_path = os.path.join(config.output_dir, "cache_metadata.pkl")

    # Load dataset
    id_prop_json = os.path.join(root_dir, "id_prop.json")
    id_prop_json_zip = os.path.join(root_dir, "id_prop.json.zip")
    id_prop_csv = os.path.join(root_dir, "id_prop.csv")
    id_prop_csv_file = False
    multioutput = False

    if os.path.exists(id_prop_json_zip):
        dat = json.loads(zipfile.ZipFile(id_prop_json_zip).read("id_prop.json"))
    elif os.path.exists(id_prop_json):
        dat = loadjson(os.path.join(root_dir, "id_prop.json"))
    elif os.path.exists(id_prop_csv):
        id_prop_csv_file = True
        with open(id_prop_csv, "r") as f:
            reader = csv.reader(f)
            dat = [row for row in reader]
        print("id_prop_csv_file exists", id_prop_csv_file)
    else:
        print("Check dataset file.")
        sys.exit(1)

    # Process training flags
    train_grad = False
    train_stress = False
    train_additional_output = False
    train_atom = False

    try:
        if config.model.calculate_gradient and config.model.gradwise_weight != 0:
            train_grad = True
        if config.model.calculate_gradient and config.model.stresswise_weight != 0:
            train_stress = True
        if config.model.atomwise_weight != 0:
            train_atom = True
        if (
            config.model.additional_output_features > 0
            and config.model.additional_output_weight != 0
        ):
            train_additional_output = True
    except Exception as exp:
        print("exp", exp)
        pass

    target_atomwise = None
    target_grad = None
    target_stress = None
    target_additional_output = None

    # Process dataset
    n_outputs = []
    dataset = []
    for i in dat:
        info = {}
        if id_prop_csv_file:
            file_name = i[0]
            tmp = [float(j) for j in i[1:]]
            info["jid"] = file_name

            if len(tmp) == 1:
                tmp = tmp[0]
            else:
                multioutput = True
                n_outputs.append(tmp)
            info["target"] = tmp
            file_path = os.path.join(root_dir, file_name)
            if file_format == "poscar":
                atoms = Atoms.from_poscar(file_path)
            elif file_format == "cif":
                atoms = Atoms.from_cif(file_path)
            elif file_format == "xyz":
                atoms = Atoms.from_xyz(file_path, box_size=500)
            elif file_format == "pdb":
                atoms = Atoms.from_pdb(file_path, max_lat=500)
            else:
                raise NotImplementedError("File format not implemented", file_format)
            info["atoms"] = atoms.to_dict()
        else:
            info["target"] = i[target_key]
            info["atoms"] = i["atoms"]
            info["jid"] = i[id_key]

        if train_atom:
            target_atomwise = "atomwise_target"
            info["atomwise_target"] = i[atomwise_key]
        if train_grad:
            target_grad = "atomwise_grad"
            info["atomwise_grad"] = i[gradwise_key]
        if train_stress:
            if len(i[stresswise_key]) == 6:
                stress = voigt_6_to_full_3x3_stress(i[stresswise_key])
            else:
                stress = i[stresswise_key]
            info["stresses"] = stress
            target_stress = "stresses"
        if train_additional_output:
            target_additional_output = "additional"
            info["additional"] = i[additional_output_key]
        if "extra_features" in i:
            info["extra_features"] = i["extra_features"]
        dataset.append(info)

    print(f"len dataset: {len(dataset)}")
    print(f"train_stress: {train_stress}")
    print(f"train_grad: {train_grad}")
    print(f"train_atom: {train_atom}")

    del dat

    # Check if we should use existing cache
    skip_cache_creation = False
    if lmdb_cache_path and os.path.exists(lmdb_cache_path):
        if skip_cache_verification:
            skip_cache_creation = True
            print("Skipping cache verification as requested - using existing cache")
        elif verify_cache_compatibility(dataset, cache_metadata_path):
            skip_cache_creation = True
            print("Skipping cache creation - using existing compatible cache")
        else:
            print("Existing cache is incompatible - will create new cache")
            config.use_lmdb = True  # Force LMDB creation

    lists_length_equal = True
    line_graph = False

    if config.compute_line_graph > 0:
        line_graph = True

    if multioutput:
        print("multioutput", multioutput)
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]
        print("lists_length_equal", lists_length_equal, len(n_outputs[0]))
        if lists_length_equal:
            config.model.output_features = len(n_outputs[0])
        else:
            raise ValueError("Make sure the outputs are of same size.")

    # Create data loaders
    print("Creating data loaders...")
    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        target="target",
        target_atomwise=target_atomwise,
        target_grad=target_grad,
        target_stress=target_stress,
        target_additional_output=target_additional_output,
        n_train=config.n_train,
        n_val=config.n_val,
        n_test=config.n_test,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        line_graph=line_graph,
        batch_size=config.batch_size,
        atom_features=config.atom_features,
        neighbor_strategy=config.neighbor_strategy,
        standardize=config.atom_features != "cgcnn",
        id_tag=config.id_tag,
        pin_memory=config.pin_memory,
        workers=config.num_workers,
        save_dataloader=config.save_dataloader,
        use_canonize=config.use_canonize,
        filename=config.filename,
        cutoff=config.cutoff,
        cutoff_extra=config.cutoff_extra,
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

    # Save cache metadata if we created a new cache
    if config.use_lmdb and not skip_cache_creation:
        save_cache_metadata(dataset, cache_metadata_path)

    # If dry run, exit after creating cache
    if dry_run:
        print("\nDry run complete! LMDB cache created at:")
        print(f"  {config.output_dir}")
        print(f"\nDataset statistics:")
        print(f"  Total samples: {len(dataset)}")
        print(f"  Train samples: {len(train_loader.dataset)}")
        print(f"  Val samples: {len(val_loader.dataset)}")
        print(f"  Test samples: {len(test_loader.dataset)}")
        print("\nTo use this cache in future runs, use:")
        print(f"  --lmdb_cache_path {config.output_dir}")
        return

    # Load or create model
    model = None
    if restart_model_path is not None:
        print("Restarting the model training:", restart_model_path)
        if config.model.name == "alignn_atomwise":
            rest_config = loadjson(
                restart_model_path.replace("current_model.pt", "config.json")
            )
            tmp = ALIGNNAtomWiseConfig(**rest_config["model"])
            print("Rest config", tmp)
            model = ALIGNNAtomWise(tmp)
            model.load_state_dict(torch.load(restart_model_path, map_location=device))
            model = model.to(device)

    # Add custom training parameters to config
    if hasattr(config, "__dict__"):
        config.report_mae = report_mae
        config.normalize_loss = normalize_loss
        config.skip_cache_verification = skip_cache_verification
    else:
        # If config doesn't support direct attribute assignment
        config_dict = config.dict() if hasattr(config, "dict") else vars(config)
        config_dict["report_mae"] = report_mae
        config_dict["normalize_loss"] = normalize_loss
        config_dict["skip_cache_verification"] = skip_cache_verification
        config = TrainingConfig(**config_dict)

    print(f"\nTraining configuration:")
    print(f"  Report MAE: {report_mae}")
    print(f"  Normalize loss: {normalize_loss}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")

    t1 = time.time()
    train_dgl(
        config,
        model=model,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
        rank=rank,
        world_size=world_size,
    )
    t2 = time.time()
    print(f"\nTotal training time: {t2 - t1:.2f} seconds")


if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    world_size = int(torch.cuda.device_count())
    print("world_size", world_size)

    if world_size > 1:
        torch.multiprocessing.spawn(
            train_for_folder,
            args=(
                world_size,
                args.root_dir,
                args.config_name,
                args.classification_threshold,
                args.batch_size,
                args.epochs,
                args.id_key,
                args.target_key,
                args.atomwise_key,
                args.force_key,
                args.stresswise_key,
                args.additional_output_key,
                args.file_format,
                args.restart_model_path,
                args.output_dir,
                args.dry_run,
                args.lmdb_cache_path,
                args.report_mae,
                args.normalize_loss,
                args.cache_metadata_path,
                args.skip_cache_verification,
            ),
            nprocs=world_size,
        )
    else:
        train_for_folder(
            0,
            world_size,
            args.root_dir,
            args.config_name,
            args.classification_threshold,
            args.batch_size,
            args.epochs,
            args.id_key,
            args.target_key,
            args.atomwise_key,
            args.force_key,
            args.stresswise_key,
            args.additional_output_key,
            args.file_format,
            args.restart_model_path,
            args.output_dir,
            args.dry_run,
            args.lmdb_cache_path,
            args.report_mae,
            args.normalize_loss,
            args.cache_metadata_path,
            args.skip_cache_verification,
        )

    try:
        cleanup(world_size)
    except Exception:
        pass
