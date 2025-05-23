"""
id_propFeature.py

This script calculates various structural descriptors for CIF files within a specified directory.
It utilizes pymatgen for structure analysis, numpy and pandas for data manipulation,
and multiprocessing for parallel processing to improve performance.

The script calculates descriptors such as:
- Spacegroup number and crystal system
- Structural complexity (based on number of distinct sites)
- Chemical bond length metrics (mean, std dev, relative deviations)
- Cell size variation
- Bond angles (mean, std dev)
- Coordination numbers (using VoronoiNN)
- Packing metrics (density, volume per atom, packing fraction)
- Lattice parameters (a, b, c, alpha, beta, gamma)
- Number of atoms

The results are saved to a CSV file, with support for resuming processing if the script is interrupted.

Usage:
1. Place the script in the directory containing your CIF files or specify the `cif_folder` variable.
2. Run the script from your terminal: `python structural_descriptor_calculator.py`
3. The calculated descriptors will be saved to 'structure_descriptors.csv' in the same directory.

Dependencies:
- pymatgen
- numpy
- pandas
- tqdm (for progress bar)

Make sure these libraries are installed in your Python environment (`pip install pymatgen numpy pandas tqdm`).

Author: Marcus Djokic
Date: May 6 2025
Version: 1.0
"""


import os
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.local_env import VoronoiNN, CrystalNN
import warnings
import multiprocessing as mp
from tqdm import tqdm  # For progress tracking
warnings.filterwarnings('ignore')

def get_spacegroup_info(structure):
    """
    Calculates the spacegroup number and crystal system for a given structure.

    This function uses the SpacegroupAnalyzer from pymatgen to determine
    the spacegroup number and the crystal system. The crystal system is
    returned as an integer code for easier numerical processing.

    Args:
        structure (pymatgen.core.Structure): The crystal structure object.

    Returns:
        tuple: A tuple containing:
            - spacegroup_num (int): The international spacegroup number.
            - crystal_system_int (int): An integer representing the crystal system:
                1: Triclinic
                2: Monoclinic
                3: Orthorhombic
                4: Tetragonal
                5: Trigonal
                6: Hexagonal
                7: Cubic
                0: Unknown or error
    """
    sga = SpacegroupAnalyzer(structure)
    spacegroup_num = sga.get_space_group_number()
    
    # Map crystal system to integer
    crystal_systems = {
        'triclinic': 1,
        'monoclinic': 2,
        'orthorhombic': 3,
        'tetragonal': 4,
        'trigonal': 5,
        'hexagonal': 6,
        'cubic': 7
    }
    
    crystal_system = sga.get_crystal_system()
    crystal_system_int = crystal_systems.get(crystal_system.lower(), 0)
    
    return spacegroup_num, crystal_system_int

def get_structural_complexity(structure):
    """
    Calculates a simplified measure of structural complexity based on the number of
    symmetrically distinct sites.

    This function uses the SpacegroupAnalyzer to obtain the symmetrized structure
    and counts the number of unique crystallographic sites. This count serves
    as a proxy for the complexity of the unit cell. It also calculates the
    complexity per atom by dividing the cell complexity by the total number of atoms.

    Note: This is a simplified approach. More rigorous measures of structural
    complexity exist (e.g., complexity based on group-subgroup relationships
    or information theory).

    Args:
        structure (pymatgen.core.Structure): The crystal structure object.

    Returns:
        tuple: A tuple containing:
            - complexity_per_atom (float): The number of symmetrically distinct sites
              divided by the total number of atoms in the unit cell.
            - complexity_per_cell (int): The total number of symmetrically distinct sites
              in the unit cell.
    """
    # This is a simplified approach, actual calculation can be more complex
    # Here using number of symmetrically distinct sites as proxy
    sga = SpacegroupAnalyzer(structure)
    symm_structure = sga.get_symmetrized_structure()
    n_distinct_sites = len(symm_structure.equivalent_sites)
    
    complexity_per_cell = n_distinct_sites
    complexity_per_atom = complexity_per_cell / len(structure)
    
    return complexity_per_atom, complexity_per_cell

def get_bond_metrics(structure: Structure):
    """
    Calculate chemical bond length metrics using CrystalNN for neighbor finding.

    Args:
        structure: A pymatgen Structure object.

    Returns:
        A tuple containing:
            - mad_rel_bond (float): Mean absolute deviation of relative bond lengths.
            - max_rel_bond (float): Maximum relative bond length.
            - min_rel_bond (float): Minimum relative bond length.
            - max_neighbor_dist_var (float): Maximum variation in average bond length per site.
            - range_neighbor_dist_var (float): Range of average bond length variation per site.
            - mean_neighbor_dist_var (float): Mean variation in average bond length per site.
            - avg_dev_neighbor_dist_var (float): Average deviation of neighbor distance variation.
            - mean_bond_length (float): Average chemical bond length across the structure.
            - std_dev_bond_length (float): Standard deviation of chemical bond lengths.
    Raises:
        ImportError: If pymatgen is not installed.
        Exception: For other errors during neighbor analysis.
    """
    nn = CrystalNN()

    all_bond_lengths = []
    site_avg_bonds = []

    try:
        for i, site in enumerate(structure):
            neighbors = nn.get_nn_info(structure, i)

            if neighbors:
                bonds = [structure.get_distance(i, n['site_index']) for n in neighbors]
                all_bond_lengths.extend(bonds)
                site_avg_bonds.append(np.mean(bonds))

    except ImportError:
        print("Error: pymatgen is not installed.")
        raise
    except Exception as e:
        print(f"An error occurred during CrystalNN analysis: {e}")
        return 0, 0, 0, 0, 0, 0, 0, 0, 0


    if not all_bond_lengths:
        return 0, 0, 0, 0, 0, 0, 0, 0, 0

    mean_bond_length = np.mean(all_bond_lengths)
    std_dev_bond_length = np.std(all_bond_lengths)

    rel_bonds = [b / mean_bond_length for b in all_bond_lengths]

    mad_rel_bond = np.mean(np.abs(np.array(rel_bonds) - 1))
    max_rel_bond = np.max(rel_bonds)
    min_rel_bond = np.min(rel_bonds)

    if len(site_avg_bonds) > 1:
        neighbor_dist_variations = np.abs(np.subtract.outer(site_avg_bonds, site_avg_bonds)).flatten()
        neighbor_dist_variations = neighbor_dist_variations[neighbor_dist_variations > 0]

        max_neighbor_dist_var = np.max(neighbor_dist_variations) if len(neighbor_dist_variations) > 0 else 0
        range_neighbor_dist_var = np.max(neighbor_dist_variations) - np.min(neighbor_dist_variations) if len(neighbor_dist_variations) > 0 else 0
        mean_neighbor_dist_var = np.mean(neighbor_dist_variations) if len(neighbor_dist_variations) > 0 else 0
        avg_dev_neighbor_dist_var = np.mean(np.abs(neighbor_dist_variations - mean_neighbor_dist_var)) if len(neighbor_dist_variations) > 0 else 0
    else:
        max_neighbor_dist_var = 0
        range_neighbor_dist_var = 0
        mean_neighbor_dist_var = 0
        avg_dev_neighbor_dist_var = 0

    return mad_rel_bond, max_rel_bond, min_rel_bond, max_neighbor_dist_var, range_neighbor_dist_var, mean_neighbor_dist_var, avg_dev_neighbor_dist_var, mean_bond_length, std_dev_bond_length

def get_cell_size_metrics(structure):
    """
    Calculates metrics related to the variation in the lattice parameters.

    This function calculates the Mean Absolute Deviation (MAD) of the relative
    cell parameters (a, b, c, alpha, beta, gamma) with respect to their average value.
    This provides a simple measure of how 'distorted' the unit cell shape is
    from a more symmetric (e.g., cubic or hexagonal) form.

    Args:
        structure (pymatgen.core.Structure): The crystal structure object.

    Returns:
        float: The Mean Absolute Deviation of the relative cell parameters.
               Returns 0 if the structure has no lattice (should not happen).
    """
    # Calculate the average cell parameter and MAD
    cell_params = structure.lattice.abc + structure.lattice.angles
    avg_cell_param = np.mean(cell_params)
    rel_cell_params = [p / avg_cell_param for p in cell_params]
    mad_rel_cell = np.mean(np.abs(np.array(rel_cell_params) - 1))
    
    return mad_rel_cell

def get_bond_angles(structure):
    """
    Calculates the average and standard deviation of bond angles in the structure
    using the CrystalNN neighbor finder.

    For each atom in the structure, this function identifies its neighbors using
    CrystalNN. It then calculates the angle formed by the central atom and
    each pair of its neighbors. The mean and standard deviation of all such
    calculated angles across the structure are returned.

    Args:
        structure (pymatgen.core.Structure): The crystal structure object.

    Returns:
        tuple: A tuple containing:
            - mean_angle (float): The average bond angle in degrees. Returns 0 if no
              angles could be calculated (e.g., structure with only one atom or
              no neighbors found).
            - std_dev_angle (float): The standard deviation of bond angles in degrees.
              Returns 0 if no angles could be calculated or only one angle was found.
    """
    nn = CrystalNN()
    angles = []
    
    try:
        for i, site in enumerate(structure):
            # Get the nearest neighbors
            neighbors = nn.get_nn_info(structure, i)
            if len(neighbors) < 2:
                continue
                
            # Calculate angles between all pairs of neighbors
            center = site.coords
            for j in range(len(neighbors)):
                site1 = neighbors[j]['site'].coords
                vec1 = site1 - center
                for k in range(j+1, len(neighbors)):
                    site2 = neighbors[k]['site'].coords
                    vec2 = site2 - center
                    
                    # Calculate angle using dot product
                    norm1 = np.linalg.norm(vec1)
                    norm2 = np.linalg.norm(vec2)
                    if norm1 > 0 and norm2 > 0:
                        dot = np.dot(vec1, vec2) / (norm1 * norm2)
                        # Handle numerical issues
                        dot = max(-1.0, min(1.0, dot))
                        angle = np.arccos(dot) * 180 / np.pi
                        angles.append(angle)
    except:
        return 0, 0
    
    if angles:
        mean_angle = np.mean(angles)
        std_dev_angle = np.std(angles)
    else:
        mean_angle = 0
        std_dev_angle = 0
        
    return mean_angle, std_dev_angle

def get_coordination_numbers(structure):
    """
    Calculates the average and standard deviation of coordination numbers
    for all sites in the structure using the VoronoiNN neighbor finder.

    VoronoiNN determines neighbors based on the faces of the Voronoi polyhedron
    around each atom. This function iterates through each site, finds its
    neighbors using VoronoiNN, counts the number of neighbors (the coordination
    number), and collects these numbers. Finally, it calculates the mean and
    standard deviation of the collected coordination numbers.

    Args:
        structure (pymatgen.core.Structure): The crystal structure object.

    Returns:
        tuple: A tuple containing:
            - mean_cn (float): The average coordination number across all sites.
              Returns 0 if no sites could be processed or no neighbors were found.
            - std_dev_cn (float): The standard deviation of coordination numbers.
              Returns 0 if no sites could be processed or only one site was processed.
    """
    voronoi = VoronoiNN()
    cns = []
    
    for i in range(len(structure)):
        try:
            cn = len(voronoi.get_nn_info(structure, i))
            cns.append(cn)
        except:
            # Skip problematic sites
            pass
    
    if cns:
        mean_cn = np.mean(cns)
        std_dev_cn = np.std(cns)
    else:
        mean_cn = 0
        std_dev_cn = 0
        
    return mean_cn, std_dev_cn

def get_packing_metrics(structure: Structure, grid_resolution: float = 0.5):
    """
    Calculates various packing metrics for a crystal structure, including
    density, volume per atom, and packing fraction.

    The packing fraction is estimated by overlaying a grid on the unit cell
    and determining the fraction of grid points that fall within the estimated
    van der Waals spheres of the atoms. A fixed dictionary of estimated atomic
    radii is used.

    Args:
        structure (pymatgen.core.Structure): The crystal structure object.
        grid_resolution (float): The spacing between grid points in Angstroms.
                                 A smaller value gives a more accurate but
                                 computationally more expensive calculation.

    Returns:
        tuple: A tuple containing:
            - density (float): The mass density of the structure in g/cm³.
            - volume_per_atom (float): The volume of the unit cell per atom in Å³.
            - packing_fraction (float): The estimated fraction of the unit cell
              volume occupied by atoms based on estimated radii. Returns 0 if
              an error occurs during calculation.
    Raises:
        ImportError: If pymatgen or numpy are not installed.
        MemoryError: If the grid resolution is too fine, leading to excessive memory usage.
        ValueError: If there's a mismatch in the number of atoms and available radii.
        Exception: For other unexpected errors during calculation.
    """
    density = structure.density
    volume_per_atom = structure.volume / len(structure)

    estimated_atomic_radii = {
        "H": 1.20, "He": 1.40, "Li": 1.82, "Be": 1.53, "B": 1.92, "C": 1.70,
        "N": 1.55, "O": 1.52, "F": 1.47, "Ne": 1.54, "Na": 2.27, "Mg": 1.73,
        "Al": 1.84, "Si": 2.10, "P": 1.80, "S": 1.80, "Cl": 1.75, "Ar": 1.88,
        "K": 2.75, "Ca": 2.31, "Sc": 2.11, "Ti": 1.87, "V": 1.79, "Cr": 1.89,
        "Mn": 1.97, "Fe": 1.94, "Co": 1.92, "Ni": 1.84, "Cu": 1.86, "Zn": 1.39, # Note: Zn example ionic
        "Ga": 1.87, "Ge": 2.11, "As": 1.85, "Se": 1.90, "Br": 1.85, "Kr": 2.02,
        "Rb": 3.03, "Sr": 2.49, "Y": 2.12, "Zr": 2.23, "Nb": 2.16, "Mo": 2.17,
        "Tc": 2.09, "Ru": 2.07, "Rh": 2.03, "Pd": 2.05, "Ag": 2.03, "Cd": 1.58,
        "In": 1.93, "Sn": 2.17, "Sb": 2.06, "Te": 2.06, "I": 1.98, "Xe": 2.16,
        "Cs": 3.43, "Ba": 2.68, "La": 2.40, "Ce": 2.35, "Pr": 2.39, "Nd": 2.29,
        "Pm": 2.36, "Sm": 2.36, "Eu": 2.32, "Gd": 2.37, "Tb": 2.36, "Dy": 2.35,
        "Ho": 2.33, "Er": 2.34, "Tm": 2.42, "Yb": 2.42, "Lu": 2.39, "Hf": 2.25,
        "Ta": 2.20, "W": 2.18, "Re": 2.17, "Os": 2.16, "Ir": 2.13, "Pt": 2.10,
        "Au": 2.10, "Hg": 1.55, "Tl": 1.96, "Pb": 2.02, "Bi": 2.07, "Po": 1.97,
        "At": 2.02, "Rn": 2.20, "Fr": 3.48, "Ra": 2.83, "Ac": 2.60, "Th": 2.37,
        "Pa": 2.30, "U": 2.41, "Np": 2.37, "Pu": 2.43, "Am": 2.44, "Cm": 2.45,
        "Bk": 2.45, "Cf": 2.45, "Es": 2.45, "Fm": 2.45, "Md": 2.45, "No": 2.45,
        "Lr": 2.45, "Rf": 2.16, "Db": 2.00, "Sg": 1.90, "Bh": 1.80, "Hs": 1.70,
        "Mt": 1.60, "Ds": 1.50, "Rg": 1.50, "Cn": 1.50, "Nh": 1.50, "Fl": 1.50,
        "Mc": 1.50, "Lv": 1.50, "Ts": 1.50, "Og": 1.50
    } # Using van der Waals radii where available, some estimates for others

    try:
        lattice = structure.lattice
        coords_frac = np.array([site.frac_coords for site in structure])
        elements = [site.specie.symbol for site in structure]

        atomic_radii_list = []
        for element_symbol in elements:
            radius = estimated_atomic_radii.get(element_symbol)
            if radius is not None:
                atomic_radii_list.append(radius)
            else:
                 print(f"Warning: No estimated atomic radius found for element {element_symbol}. Cannot include this atom in grid calculation.")
                 atomic_radii_list.append(0.0)

        if len(atomic_radii_list) != len(structure):
             raise ValueError("Mismatch between number of atoms and available radii.")

        atomic_radii_list = np.array(atomic_radii_list)

        a, b, c = lattice.lengths
        n_a = int(np.ceil(a / grid_resolution))
        n_b = int(np.ceil(b / grid_resolution))
        n_c = int(np.ceil(c / grid_resolution))

        frac_grid_a = np.linspace(0, 1.0 - (1.0/n_a), n_a)
        frac_grid_b = np.linspace(0, 1.0 - (1.0/n_b), n_b)
        frac_grid_c = np.linspace(0, 1.0 - (1.0/n_c), n_c)

        frac_grid = np.meshgrid(frac_grid_a, frac_grid_b, frac_grid_c, indexing='ij')
        grid_points_frac = np.vstack([g.ravel() for g in frac_grid]).T
        grid_points_cart = lattice.get_cartesian_coords(grid_points_frac)

        occupied_grid = np.zeros(grid_points_cart.shape[0], dtype=bool)

        for i, atom_coord_frac in enumerate(coords_frac):
            radius = atomic_radii_list[i]

            if radius <= 0:
                continue

            diff_frac = grid_points_frac - atom_coord_frac
            diff_frac_pbc = diff_frac - np.round(diff_frac)
            diff_cart_pbc = lattice.get_cartesian_coords(diff_frac_pbc)
            squared_periodic_distances = np.sum(diff_cart_pbc**2, axis=1)

            points_within_sphere = squared_periodic_distances <= (radius**2)
            occupied_grid = occupied_grid | points_within_sphere

        num_occupied_points = np.sum(occupied_grid)
        voxel_volume = structure.volume / (n_a * n_b * n_c)
        occupied_volume_grid = num_occupied_points * voxel_volume
        packing_fraction = occupied_volume_grid / structure.volume

        return density, volume_per_atom, packing_fraction

    except ImportError:
        print("Error: pymatgen or numpy is not installed. Please install them.")
        raise
    except MemoryError:
        print(f"Memory Error: The grid resolution ({grid_resolution} Å) is too fine. Try increasing grid_resolution.")
        return 0, 0, 0
    except Exception as e:
        print(f"An error occurred: {e}")
        return 0, 0, 0

def get_lattice_parameters(structure):
    """
    Extracts the lattice parameters (lengths and angles) from a crystal structure.

    Args:
        structure (pymatgen.core.Structure): The crystal structure object.

    Returns:
        tuple: A tuple containing the six lattice parameters:
            - a (float): Lattice vector length 'a' in Angstroms.
            - b (float): Lattice vector length 'b' in Angstroms.
            - c (float): Lattice vector length 'c' in Angstroms.
            - alpha (float): Lattice angle 'alpha' in degrees.
            - beta (float): Lattice angle 'beta' in degrees.
            - gamma (float): Lattice angle 'gamma' in degrees.
        Returns (0, 0, 0, 0, 0, 0) if the structure has no lattice.
    """
    a, b, c = structure.lattice.abc
    alpha, beta, gamma = structure.lattice.angles
    return a, b, c, alpha, beta, gamma

def calculate_structure_descriptors(cif_path):
    """
    Calculates a comprehensive set of structural descriptors for a crystal
    structure loaded from a CIF file.

    This function serves as the main processing unit for a single CIF file.
    It loads the structure, calls various helper functions to compute different
    types of descriptors, and compiles them into a dictionary. If any error
    occurs during the process (e.g., failed to load CIF, error during descriptor
    calculation), it catches the exception and returns a dictionary containing
    the material ID and an error message, preventing the entire script from crashing.

    Args:
        cif_path (str): The full path to the CIF file.

    Returns:
        dict: A dictionary containing the calculated structural descriptors
              with descriptive keys, or a dictionary with 'material_id' and
              an 'error' key if processing fails.
    """
    try:
        structure = Structure.from_file(cif_path)
        
        # Basic structure info
        natoms = len(structure)
        
        # Calculate all descriptors
        spacegroup_num, crystal_system_int = get_spacegroup_info(structure)
        complexity_per_atom, complexity_per_cell = get_structural_complexity(structure)
        mad_rel_bond, max_rel_bond, min_rel_bond, max_neighbor_dist_var, range_neighbor_dist_var, mean_neighbor_dist_var, avg_dev_neighbor_dist_var, mean_bond_length, std_dev_bond_length = get_bond_metrics(structure)
        mad_rel_cell = get_cell_size_metrics(structure)
        mean_angle, std_dev_angle = get_bond_angles(structure)
        mean_cn, std_dev_cn = get_coordination_numbers(structure)
        density, vpa, packing_fraction = get_packing_metrics(structure)
        a, b, c, alpha, beta, gamma = get_lattice_parameters(structure)
        
        # Return all descriptors as a dictionary
        return {
            'material_id': os.path.basename(cif_path).replace('.cif', ''),
            'spacegroup_num': spacegroup_num,
            'crystal_system_int': crystal_system_int,
            'structural complexity per atom': complexity_per_atom,
            'structural complexity per cell': complexity_per_cell,
            'mean absolute deviation in relative bond length': mad_rel_bond,
            'max relative bond length': max_rel_bond,
            'min relative bond length': min_rel_bond,
            'maximum neighbor distance variation': max_neighbor_dist_var,
            'range neighbor distance variation': range_neighbor_dist_var,
            'mean neighbor distance variation': mean_neighbor_dist_var,
            'avg_dev neighbor distance variation': avg_dev_neighbor_dist_var,
            'mean absolute deviation in relative cell size': mad_rel_cell,
            'mean Average bond length': mean_bond_length,
            'std_dev Average bond length': std_dev_bond_length,
            'mean Average bond angle': mean_angle,
            'std_dev Average bond angle': std_dev_angle,
            'mean CN_VoronoiNN': mean_cn,
            'std_dev CN_VoronoiNN': std_dev_cn,
            'density': density,
            'vpa': vpa,
            'packing fraction': packing_fraction,
            'a': a,
            'b': b,
            'c': c,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'natoms': natoms
        }
    except Exception as e:
        # Return default values on error
        return {'material_id': os.path.basename(cif_path), 'error': str(e)}
        #return {feature: 0 for feature in [
        #    'spacegroup_num', 'crystal_system_int', 'structural complexity per atom', 'structural complexity per cell',
        #    'mean absolute deviation in relative bond length', 'max relative bond length', 'min relative bond length',
        #    'maximum neighbor distance variation', 'range neighbor distance variation', 'mean neighbor distance variation',
        #    'avg_dev neighbor distance variation', 'mean absolute deviation in relative cell size', 'mean Average bond length',
        #    'std_dev Average bond length', 'mean Average bond angle', 'std_dev Average bond angle',
        #    'mean CN_VoronoiNN', 'std_dev CN_VoronoiNN', 'density', 'vpa', 'packing fraction',
        #    'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'natoms'
        #]}

def process_batch(cif_batch):
    """
    Processes a batch of CIF files using multiprocessing.

    This function takes a list of CIF file paths and processes them in parallel
    using a multiprocessing Pool. It uses `imap_unordered` to process files
    as they complete, which can be slightly more efficient than `map`.
    A progress bar from `tqdm` is included to visualize the processing of the batch.
    The number of worker processes is set to half the available CPU cores,
    with a minimum of 1 process.

    Args:
        cif_batch (list): A list of strings, where each string is the path
                          to a CIF file.

    Returns:
        list: A list of dictionaries, where each dictionary contains the
              calculated descriptors for a single CIF file (or an error message).
    """
    with mp.Pool(processes=max(1, mp.cpu_count() // 2)) as pool:
        results = list(tqdm(pool.imap_unordered(calculate_structure_descriptors, cif_batch), total=len(cif_batch)))
    return results

def chunk_list(lst, chunk_size):
    """
    Generates successive chunk_size-sized chunks from a list.

    This is a generator function that yields sub-lists (chunks) of the input
    list. Useful for processing large lists in smaller, manageable batches.

    Args:
        lst (list): The input list to be chunked.
        chunk_size (int): The maximum size of each chunk.

    Yields:
        list: A chunk (sub-list) of the input list. The last chunk may be
              smaller than chunk_size.
    """
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i + chunk_size]

def main():
    # --- Configuration ---
    cif_folder = "."  # Directory containing your CIF files. Set to "." for the current directory.
    output_file = "structure_descriptors.csv" # Name of the output CSV file.
    batch_size = 100  # Number of CIF files to process in each batch. Adjust based on system memory.
    # ---------------------

    # Get list of all CIF files
    cif_files = [os.path.join(cif_folder, f) for f in os.listdir(cif_folder) if f.endswith('.cif')]

    # Check if output already exists (to append in resume scenarios)
    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        done_files = set(existing_df['material_id'])
    else:
        existing_df = pd.DataFrame()
        done_files = set()

    # Filter out files already processed
    cif_files_to_process = [f for f in cif_files if os.path.basename(f) not in done_files]
    
    for batch_num, cif_batch in enumerate(chunk_list(cif_files_to_process, batch_size), 1):
        print(f"Processing batch {batch_num} with {len(cif_batch)} files...")
        
        batch_results = process_batch(cif_batch)
        
        # Convert to dataframe
        batch_df = pd.DataFrame(batch_results)
        
        # Append to CSV
        batch_df.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
        
        print(f"Batch {batch_num} done. Saved to {output_file}.")

if __name__ == "__main__":
    main()