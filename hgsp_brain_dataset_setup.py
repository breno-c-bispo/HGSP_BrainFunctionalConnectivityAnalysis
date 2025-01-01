import pathlib
import pandas as pd
import numpy as np
from scipy.stats import zscore
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

hoi_zscored_folder_path = pathlib.Path("./Schaefer_100Parcels_Atlas/hoi_files_zscored")
hoi_zscored_folder_path.mkdir(parents=True, exist_ok=True)

hoi_raw_folder_path = pathlib.Path("./Schaefer_100Parcels_Atlas/hoi_files_raw")
hoi_raw_folder_path.mkdir(parents=True, exist_ok=True)


def process_files_hoi(f):
    """
    Process a single HCP subject's Higher-Order Interaction (HOI) file.

    Parameters
    ----------
    f : str
        Path to the HOI file.

    Notes
    -----
    The HOI file is read into a pandas DataFrame, two HOI metrics are selected ('Int Info' and 'Total Corr'),
    that is the interaction information (II) and the total correlation (TC) metrics, and then converted to a numpy array.
    The numpy array is then z-scored along rows to obtain the z-scored HOI array. Subsequently, the z-scored HOI array
    is saved as a .npy file in the 'hoi_files_zscored' folder. The raw HOI array is also saved as a .npy file in the
    'hoi_files_raw' folder.
    """
    hoi = pd.read_csv(f, usecols=["Int Info", "Total Corr"]).to_numpy()
    z_scored_hoi = zscore(hoi, axis=0)
    idx = f.index("_REST")
    file_id = f[idx - 11 : idx - 5]
    file_rest = f[idx + 5]
    np.save(
        str(hoi_zscored_folder_path)
        + "/"
        + file_id
        + "_fMRI_REST"
        + file_rest
        + ".npy",
        z_scored_hoi,
    )
    np.save(
        str(hoi_raw_folder_path) + "/" + file_id + "_fMRI_REST" + file_rest + ".npy",
        hoi,
    )
    return


def main():
    """
    Main function to process correlation matrices and higher-order interaction (HOI) data.

    This function performs the following tasks:
    1. Computes the averaged correlation matrix from a set of input correlation files.
       The diagonal of each matrix is set to zero, the matrices are summed, and then averaged.
       The result is saved to 'corr_avr_matrix.txt'.

    2. Converts raw HOI data into z-scored numpy arrays and saves them as .npy files for further usage.
       The processing is done in parallel using a process pool executor for efficiency.

    3. Computes the averaged z-scored HOI hyperedges from the generated .npy files.
       The result is saved as 'hoi_avr_hyperedges.npy'.
    """

    # Read correlation matrices
    print("Computing averaged correlation matrix...")
    corr_avr = np.zeros((n_rois, n_rois))
    for f in tqdm(files_corr):
        temp = np.abs(np.loadtxt(f))
        np.fill_diagonal(temp, 0)
        corr_avr += temp / len(files_corr)
        pass
    corr_avr_dir = "./Schaefer_100Parcels_Atlas/corr_avr_matrix.txt"
    np.savetxt(corr_avr_dir, corr_avr)
    print(f"Done! Averaged correlation matrix saved to {corr_avr_dir}")

    # Read hyperedges and convert to z-scored hyperedges
    print("Converting raw hyperedges to z-scored .npy files for further usage...")
    with ProcessPoolExecutor() as executor:
        # Parallel processing of files
        list(tqdm(executor.map(process_files_hoi, files_hoi), total=len(files_hoi)))
    print("Done!")

    # Compute the mean individual z-scored hyperedges
    print("Computing mean individual z-scored hyperedges...")
    z_scored_hoi_avr = np.zeros((len(triangles), 2))
    files = glob(str(hoi_zscored_folder_path) + "/*.npy")
    for f in tqdm(files):
        z_scored_hoi = np.load(f)
        z_scored_hoi_avr += z_scored_hoi / len(files)

    hoi_avr_dir = "./Schaefer_100Parcels_Atlas/hoi_zscored_avr_hyperedges.npy"
    z_scored_hoi_avr = np.save(hoi_avr_dir, z_scored_hoi_avr)
    print(f"Done! Averaged z-scored hyperedges saved to {hoi_avr_dir}")


if __name__ == "__main__":
    main()
