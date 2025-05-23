#!/usr/bin/env python3
"""
Simple script to check your conda environment against ALIGNN requirements
"""

import sys
import subprocess
import importlib
import yaml
import os

# Define key packages to check
KEY_PACKAGES = [
    "numpy", 
    "torch", 
    "dgl", 
    "spglib", 
    "scipy",
    "matplotlib",
    "python-lmdb",
    "jarvis-tools"
]

# Function to get installed version
def get_installed_version(package_name):
    package_mapping = {
        "python-lmdb": "lmdb"  # Handle special naming cases
    }
    
    module_name = package_mapping.get(package_name, package_name)
    
    try:
        module = importlib.import_module(module_name)
        version = getattr(module, "__version__", "unknown")
        return version
    except ImportError:
        return "Not installed"

# Print environment info
print("=== Current Python Environment ===")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Get conda environment name
try:
    env_name = os.environ.get('CONDA_DEFAULT_ENV', 'Not in conda environment')
    print(f"Conda environment: {env_name}")
except Exception as e:
    print(f"Error getting conda environment: {e}")

print("\n=== Checking Key Packages ===")
for package in KEY_PACKAGES:
    version = get_installed_version(package)
    print(f"{package}: {version}")

print("\n=== Complete Package List ===")
try:
    # Run pip list
    result = subprocess.run(
        [sys.executable, "-m", "pip", "list"], 
        capture_output=True, 
        text=True
    )
    print(result.stdout)
except Exception as e:
    print(f"Error running pip list: {e}")

print("\n=== Environment File Information ===")
try:
    # Path to environment.yml (assuming it's in the current directory)
    env_file = "/mnt/iscsi/MLModels/learning_hybridizing_descriptors_upload/alignn/environment.yml"
    
    # Read the environment.yml file
    with open(env_file, 'r') as f:
        env_data = yaml.safe_load(f)
    
    # Extract information from environment.yml
    env_name = env_data.get('name', 'Unknown')
    print(f"Environment name in file: {env_name}")
    
    # Extract package versions from conda dependencies
    conda_deps = env_data.get('dependencies', [])
    conda_packages = {}
    
    for dep in conda_deps:
        if isinstance(dep, str) and '=' in dep:
            parts = dep.split('=')
            if len(parts) >= 2:
                package = parts[0]
                version = '='.join(parts[1:])
                conda_packages[package] = version
    
    # Extract pip packages if present
    pip_packages = {}
    for dep in conda_deps:
        if isinstance(dep, dict) and 'pip' in dep:
            for pip_dep in dep['pip']:
                if '==' in pip_dep:
                    package, version = pip_dep.split('==', 1)
                    pip_packages[package] = version
    
    # Compare KEY_PACKAGES with environment.yml
    print("\n=== Key Package Comparison ===")
    print("Package | Current Version | Required Version")
    print("--------|-----------------|------------------")
    
    for package in KEY_PACKAGES:
        current = get_installed_version(package)
        
        # Check in conda packages first, then pip packages
        if package in conda_packages:
            required = conda_packages[package]
        elif package in pip_packages:
            required = pip_packages[package]
        else:
            required = "Not specified"
            
        # Highlight mismatches
        if current != required and required != "Not specified":
            print(f"{package} | {current} | {required} ⚠️")
        else:
            print(f"{package} | {current} | {required}")
    
except Exception as e:
    print(f"Error processing environment.yml: {e}")

print("\nThis information should help you compare your current environment with the required one.")
