import pandas as pd
from itertools import combinations
from glob import glob
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import sys
sys.path.insert(0, "./Background_Scripts")
from Background_Scripts.utils import get_symmetrized_ttensors, mean_tubal_scalars

# File paths
files_hoi_dir = './Schaefer_100Parcels_Atlas/hoi_files/*.npy'
file_clinical_dir = './unregistered_clinical_data.csv'

# Import files hoi
files_hoi = glob(files_hoi_dir)

# Extract Subject IDs and REST numbers
index_subject_id = files_hoi[0].index("\\")
subject_ids = [int(file[index_subject_id + 1:index_subject_id + 7]) for file in files_hoi]
rest_numbers = [int(file[index_subject_id + 12]) for file in files_hoi]
df_peaks = pd.DataFrame({'Subject': subject_ids, 'REST': rest_numbers})

# Generate triplet labels
triplets_labels = combinations(range(116), 3)
k = [0, 4]

# Function to process a single file
def process_file(file):
    As_ii, As_tc = get_symmetrized_ttensors(file, triplets_labels)
    peak_values_tubal_scalar_ii = mean_tubal_scalars(As_ii.fft, k)
    peak_values_tubal_scalar_tc = mean_tubal_scalars(As_tc.fft, k)
    return (
        peak_values_tubal_scalar_ii.loc[peak_values_tubal_scalar_ii['k'] == 0, 'mean'].values[0],
        peak_values_tubal_scalar_ii.loc[peak_values_tubal_scalar_ii['k'] == 4, 'mean'].values[0],
        peak_values_tubal_scalar_tc.loc[peak_values_tubal_scalar_tc['k'] == 0, 'mean'].values[0],
        peak_values_tubal_scalar_tc.loc[peak_values_tubal_scalar_tc['k'] == 4, 'mean'].values[0]
    )

def main():  
    # Parallel processing
    with ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_file, files_hoi), total=len(files_hoi)))

    # Unpack results into separate lists
    peak_ii_0, peak_ii_4, peak_tc_0, peak_tc_4 = zip(*results)

    # Add results to DataFrame
    df_peaks['peak_ii_0'] = peak_ii_0
    df_peaks['peak_ii_4'] = peak_ii_4
    df_peaks['peak_tc_0'] = peak_tc_0
    df_peaks['peak_tc_4'] = peak_tc_4

    # Merge with clinical data
    df_clinical = pd.read_excel(file_clinical, engine='openpyxl')
    df_clinical_peaks = pd.merge(df_clinical[['Subject', 'REST', 'Gender']], df_peaks, how='inner', on=['Subject'])

    # Save DataFrame to Excel file
    file_out = 'hgsp_peak_values.xlsx'
    df_clinical_peaks.to_excel(file_out, index=False)

if __name__ == '__main__':
    main()
