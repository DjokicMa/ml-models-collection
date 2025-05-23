import pandas as pd
import os
import sys

def merge_csv_by_material_id(id_prop_file, structure_descriptors_file, output_file):
    """
    Merges two CSV files based on the 'material_id' column in id_prop.csv
    and a 'filename' column in structure_descriptors.csv (used as material_id)

    Args:
        id_prop_file: Path to the id_prop.csv file.
        structure_descriptors_file: Path to the structure_descriptors.csv file.
        output_file: Path to the output merged file.

    Returns:
        None. Saves the merged DataFrame to the output_file.
        Prints an informative message if successful or an error message if merging fails.
    """

    try:
        # Read the CSV files into pandas DataFrames
        id_prop_df = pd.read_csv(id_prop_file)
        structure_descriptors_df = pd.read_csv(structure_descriptors_file)

        # Extract material_id from structure_descriptors_df using filename
        # Ensure filenames are in the correct format (e.g., 'material_name.cif')
        structure_descriptors_df['material_id'] = structure_descriptors_df['filename'].str.replace('.cif', '')

        # Use a more robust merge method, handling potential missing values
        merged_df = pd.merge(id_prop_df, structure_descriptors_df, on='material_id', how='left')

        # Check for any remaining NaN values and handle them appropriately.
        #  This is crucial for robustness.  Print a warning if found.
        if merged_df.isnull().values.any():
            print("Warning: Missing values after merge. Check data integrity.")
            # Optionally, you can fill NaN values with a specific value (e.g., 0).
            # merged_df.fillna(0, inplace=True)
            # Or drop rows with NaN values.
            # merged_df.dropna(inplace=True)


        #Save the merged DataFrame to a new CSV file.
        merged_df.to_csv(output_file, index=False)
        print(f"CSV files merged successfully and saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: One or both input CSV files not found.")
        print(f"id_prop_file: {id_prop_file}")
        print(f"structure_descriptors_file: {structure_descriptors_file}")
        sys.exit(1) # Exit with error code
    except pd.errors.EmptyDataError:
        print(f"Error: One or both input CSV files are empty.")
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Column '{e.args[0]}' not found in one or both CSV files.")
        sys.exit(1)
    except Exception as e:  # Catch any other unexpected errors
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)

# Example usage (replace with your file paths)
id_prop_file = 'id_prop_w_header.csv'
structure_descriptors_file = 'structure_descriptors.csv'
output_file = 'merged_data.csv'

merge_csv_by_material_id(id_prop_file, structure_descriptors_file, output_file)
