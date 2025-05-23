#!/usr/bin/env python

"""Module to train for a folder with formatted dataset."""

import csv
import os
import sys
import time
from jarvis.core.atoms import Atoms
from dealignn.data import get_train_val_loaders
from dealignn.train import train_dgl
from dealignn.config import TrainingConfig
from jarvis.db.jsonutils import loadjson
import argparse

import requests
import os
import zipfile
from tqdm import tqdm
from dealignn.models.alignn import ALIGNN, ALIGNNConfig
from dealignn.data import get_torch_dataset
from torch.utils.data import DataLoader
import tempfile
import torch
import sys

# from jarvis.db.jsonutils import loadjson
import argparse
from jarvis.core.atoms import Atoms
from jarvis.core.graphs import Graph
from jarvis.db.jsonutils import dumpjson
import pandas as pd
import json

device = "cpu"
if torch.cuda.is_available():
    device = torch.device("cuda")
    # Limit GPU memory usage to 50%
    torch.cuda.set_per_process_memory_fraction(0.65, 0)
    # Print available GPU memory
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"Total GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
    )
    print(f"Limiting to 65% of available memory")


parser = argparse.ArgumentParser(description="Atomistic Line Graph Neural Network")
parser.add_argument(
    "--root_dir",
    default="./sample",
    help="Folder with id_props.csv, structure files",
)
parser.add_argument(
    "--config_name",
    default="./config.json",
    help="Name of the config file",
)

parser.add_argument(
    "--file_format", default="poscar", help="poscar/cif/xyz/pdb file format."
)

parser.add_argument(
    "--keep_data_order",
    default=False,
    help="Whether to randomly shuffle samples, True/False",
)

parser.add_argument(
    "--classification_threshold",
    default=None,
    help="Floating point threshold for converting into 0/1 class"
    + ", use only for classification tasks",
)

parser.add_argument("--batch_size", default=64, help="Batch size, generally 64")

parser.add_argument("--epochs", default=300, help="Number of epochs, generally 1000")

parser.add_argument(
    "--output_dir",
    default="./",
    help="Folder to save outputs",
)

parser.add_argument(
    "--test_dir",
    default="",
    help="Folder to test and make predictions (optional)",
)


def get_test_loader(test_dir="./test", config=None, features=None):
    """Get test data loader from separate test directory."""
    id_prop_dat = os.path.join(test_dir, "id_prop.csv")
    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]
    dataset = []
    n_outputs = []
    for i in data:
        info = {}
        file_name = i[0]
        file_path = os.path.join(test_dir, file_name)
        atoms = Atoms.from_poscar(file_path)
        info["features"] = features[file_name]
        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name

        tmp = [float(j) for j in i[1:]]  # float(i[1])
        if len(tmp) == 1:
            tmp = tmp[0]
        else:
            multioutput = True
        info["target"] = tmp  # float(i[1])
        n_outputs.append(info["target"])
        dataset.append(info)

    test_data = get_torch_dataset(
        dataset=dataset,
        target=config.target,
        neighbor_strategy="k-nearest",
        atom_features="cgcnn",
        use_canonize=True,
        line_graph=True,
    )
    collate_fn = test_data.collate_line_graph
    test_loader = DataLoader(
        test_data,
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        drop_last=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )
    return test_loader


def train_for_folder(
    root_dir="examples/sample_data",
    config=None,
    keep_data_order=False,
    classification_threshold=None,
    batch_size=None,
    epochs=None,
    file_format="poscar",
    output_dir=None,
    test_dir=None,
    features=None,
    resume=None,
):
    """Train for a folder."""
    # config_dat=os.path.join(root_dir,config_name)
    id_prop_dat = os.path.join(root_dir, "id_prop.csv")
    #    config = loadjson(config_name)
    #    if type(config) is dict:
    #        try:
    #            config = TrainingConfig(**config)
    #        except Exception as exp:
    #            print("Check", exp)

    config.keep_data_order = keep_data_order
    if classification_threshold is not None:
        config.classification_threshold = float(classification_threshold)
    if output_dir is not None:
        config.output_dir = output_dir
    if batch_size is not None:
        config.batch_size = int(batch_size)
    if epochs is not None:
        config.epochs = int(epochs)
    with open(id_prop_dat, "r") as f:
        reader = csv.reader(f)
        data = [row for row in reader]

    dataset = []
    n_outputs = []
    multioutput = False
    lists_length_equal = True
    for i in data:
        info = {}
        file_name = i[0]
        file_path = os.path.join(root_dir, file_name)
        if file_format == "poscar":
            atoms = Atoms.from_poscar(file_path)
        elif file_format == "cif":
            atoms = Atoms.from_cif(file_path)  # +'.cif')
        elif file_format == "xyz":
            # Note using 500 angstrom as box size
            atoms = Atoms.from_xyz(file_path, box_size=500)
        elif file_format == "pdb":
            # Note using 500 angstrom as box size
            # Recommended install pytraj
            # conda install -c ambermd pytraj
            atoms = Atoms.from_pdb(file_path, max_lat=500)
        else:
            raise NotImplementedError("File format not implemented", file_format)

        info["features"] = features[file_name]
        info["atoms"] = atoms.to_dict()
        info["jid"] = file_name

        tmp = [float(j) for j in i[1:]]  # float(i[1])
        if len(tmp) == 1:
            tmp = tmp[0]
        else:
            multioutput = True
        info["target"] = tmp  # float(i[1])
        n_outputs.append(info["target"])
        dataset.append(info)
    if multioutput:
        lists_length_equal = False not in [
            len(i) == len(n_outputs[0]) for i in n_outputs
        ]

    # print ('n_outputs',n_outputs[0])
    if multioutput and classification_threshold is not None:
        raise ValueError("Classification for multi-output not implemented.")
    if multioutput and lists_length_equal:
        config.model.output_features = len(n_outputs[0])
    else:
        # TODO: Pad with NaN
        if not lists_length_equal:
            raise ValueError("Make sure the outputs are of same size.")
        else:
            config.model.output_features = 1
    (
        train_loader,
        val_loader,
        test_loader,
        prepare_batch,
    ) = get_train_val_loaders(
        dataset_array=dataset,
        target=config.target,
        #        features=features,
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
    )

    # Only use separate test data if explicitly requested and it exists
    if (
        test_dir
        and test_dir != ""
        and os.path.exists(os.path.join(test_dir, "id_prop.csv"))
    ):
        print(f"Using separate test data from {test_dir} instead of test split")
        test_loader = get_test_loader(
            test_dir=test_dir, config=config, features=features
        )
    else:
        print(
            f"Using test split from main dataset (no separate test directory specified or found)"
        )

    t1 = time.time()
    train_dgl(
        config,
        resume=resume,
        train_val_test_loaders=[
            train_loader,
            val_loader,
            test_loader,
            prepare_batch,
        ],
    )
    t2 = time.time()
    print("Time taken (s):", t2 - t1)


# Define hyperparameter sets
edge_input_features = [80]
hidden_features = [256]
triplet_input_features = [40]
lrs = [1e-3]

para_sets = []

for lr in lrs:
    for eif in edge_input_features:
        for hf in hidden_features:
            for tif in triplet_input_features:
                for i in range(1):
                    para_set = {}
                    para_set["learning_rate"] = lr
                    para_set["edge_input_features"] = eif
                    para_set["hidden_features"] = hf
                    para_set["triplet_input_features"] = tif
                    para_set["n_repeat"] = i
                    para_sets.append(para_set)


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


if __name__ == "__main__":
    args = parser.parse_args()

    # Load descriptor CSV
    descriptor_path = os.path.join(
        args.root_dir, "structure_descriptors_normalized.csv"
    )
    print(f"Looking for descriptors at: {descriptor_path}")

    if os.path.exists(descriptor_path):
        df = pd.read_csv(descriptor_path)
        features_dict = {}

        for _, row in df.iterrows():
            mid = str(row["material_id"]).strip()
            if mid.endswith(".cif"):
                mid = mid[:-4]
            features_dict[mid + ".cif"] = [row[feat] for feat in feature_list]
    else:
        # If descriptor file not found at the expected location, try alternative approach
        print(f"Descriptor file not found at {descriptor_path}")
        print("Using fallback approach to process descriptors...")

        # Check if the provided path has id_prop.csv
        id_prop_path = os.path.join(args.root_dir, "id_prop.csv")
        if not os.path.exists(id_prop_path):
            print(f"Error: Cannot find id_prop.csv in {args.root_dir}")
            sys.exit(1)

        # Read material IDs from id_prop.csv to know what structures to process
        with open(id_prop_path, "r") as f:
            reader = csv.reader(f)
            material_ids = [row[0] for row in reader]

        # Create a dict with empty features (or default features)
        features_dict = {}
        for mid in material_ids:
            # Default to 28 zero features (matching the length of feature_list)
            features_dict[mid] = [0.0] * len(feature_list)

        print(f"Created default features for {len(features_dict)} materials")

    # Step 1: Load the config dictionary
    with open(args.config_name) as f:
        config_dict = json.load(f)

    # Step 2: Create the Pydantic config object
    config = TrainingConfig(**config_dict)

    # Display config info
    print(f"Config loaded from {args.config_name}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Line graph: {config.compute_line_graph}")

    # Apply each parameter set
    for para_set in para_sets:
        print("=" * 50)
        print(f"Training with parameters: {para_set}")

        config.learning_rate = para_set["learning_rate"]
        config.model.edge_input_features = para_set["edge_input_features"]
        config.model.hidden_features = para_set["hidden_features"]
        config.model.triplet_input_features = para_set["triplet_input_features"]
        config.random_seed = para_set["n_repeat"]

        train_for_folder(
            root_dir=args.root_dir,
            config=config,
            keep_data_order=args.keep_data_order,
            classification_threshold=args.classification_threshold,
            output_dir=args.output_dir,
            batch_size=config.batch_size,
            epochs=config.epochs,
            file_format=args.file_format,
            test_dir=args.test_dir,
            features=features_dict,
            resume=None,  # Or your checkpoint like 'checkpoint_266.pt'
        )

        # Save results with parameter set info
        output_suffix = f"_lr{para_set['learning_rate']}_eif{para_set['edge_input_features']}_hf{para_set['hidden_features']}"

        # Find and rename output files if they exist
        for src_filename in [
            "prediction_results_test_set.csv",
            f"checkpoint_{config.epochs}.pt",
        ]:
            if os.path.exists(src_filename):
                base, ext = os.path.splitext(src_filename)
                dst_filename = f"{base}{output_suffix}{ext}"
                print(f"Renaming {src_filename} to {dst_filename}")
                os.rename(src_filename, dst_filename)

                # Move to output directory if specified
                if args.output_dir and args.output_dir != "./":
                    os.makedirs(args.output_dir, exist_ok=True)
                    dst_path = os.path.join(args.output_dir, dst_filename)
                    print(f"Moving {dst_filename} to {dst_path}")
                    os.rename(dst_filename, dst_path)
