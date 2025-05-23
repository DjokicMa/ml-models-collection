#!/usr/bin/env python
"""Script to fix test_loader issue in fine-tuning.py properly"""

import os

# Path to the fine-tuning.py file
script_path = "/mnt/iscsi/MLModels/learning_hybridizing_descriptors_upload/dealignn/fine-tuning.py"

# Read the current content
with open(script_path, "r") as f:
    content = f.read()

# Find the train_for_folder function
function_start = content.find("def train_for_folder")
if function_start == -1:
    print("Could not find train_for_folder function")
    exit(1)

# Find the problematic line within the function
old_line = "    test_loader = get_test_loader(test_dir=test_dir, config=config, features=features)"
new_line = """    # Only use separate test data if explicitly requested
    if test_dir and test_dir != "" and os.path.exists(os.path.join(test_dir, "id_prop.csv")):
        print(f"Using separate test data from {test_dir} instead of test split")
        test_loader = get_test_loader(test_dir=test_dir, config=config, features=features)
    else:
        print(f"Using test split from main dataset (no separate test data)")"""

# Make the replacement
patched_content = content.replace(old_line, new_line)

# Write the patched file
with open(script_path, "w") as f:
    f.write(patched_content)

print(f"Patched {script_path} to use the test split from main dataset by default")
