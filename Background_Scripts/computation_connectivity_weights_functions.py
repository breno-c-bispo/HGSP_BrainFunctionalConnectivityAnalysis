import os
import re
from glob import glob

import numpy as np
import pandas as pd
from hypergraph_connectivity_functions import get_symmetrized_t_fft
from joblib import Parallel, delayed
from scipy.stats import zscore
from thoi.measures.gaussian_copula import nplets_measures
from tqdm import tqdm


def _default_nbins(n_samples):
    return max(2, int((n_samples / 5) ** 0.5))


def compute_bins(data, nbins=None):
    """
    Compute per-variable bin edges for a 2D dataset.
    Parameters
    ----------
    data : array-like, shape (n_samples, n_variables)
        Input dataset. Each column is treated as a separate variable. The input
        will be interpreted as a 2-D array; n_samples is the number of rows and
        n_variables the number of columns. Values must be finite (no NaNs or
        infinities).
    nbins : int, optional
        Number of bins to produce for each variable. If None, a default number of
        bins is chosen by calling _default_nbins(n_samples), where n_samples is
        the number of rows in `data`.
    Returns
    -------
    edges : list of ndarray
        A list of length `n_variables` where each element is a 1-D NumPy array of
        length `nbins + 1` containing the bin edge locations for the
        corresponding variable (including both left and right edges).
    nbins : int
        The number of bins actually used (the provided value or the value
        returned by _default_nbins).
    Raises
    ------
    ValueError
        If any column in `data` contains non-finite values (NaN or infinite).
    Notes
    -----
    - For each variable (column) the minimum and maximum values are computed and
      numpy.linspace is used to generate evenly spaced bin edges between them.
    - If a column is (nearly) constant so that max - min < 1e-12, the maximum is
      increased by 1e-12 to avoid creating zero-width bins.
    Examples
    --------
    >>> # X is a 2-D NumPy array with shape (n_samples, n_variables)
    >>> edges, nbins = compute_bins(X, nbins=10)
    >>> len(edges) == X.shape[1]
    True
    >>> edges[0].shape[0] == nbins + 1
    True
    """
    n_samples, n_variables = data.shape
    if nbins is None:
        nbins = _default_nbins(n_samples)

    edges = []
    for i in range(n_variables):
        x_min = data[:, i].min()
        x_max = data[:, i].max()
        # avoid zero-width range
        if not np.isfinite(x_min) or not np.isfinite(x_max):
            raise ValueError("Non-finite values in data.")
        if x_max - x_min < 1e-12:
            x_max = x_min + 1e-12
        edges.append(np.linspace(x_min, x_max, nbins + 1))
    return edges, nbins


def joint_entropies_fast(data, nbins=None):
    """
    Compute marginal and pairwise joint entropies for all variables in a dataset using
    pre-binning and efficient histogramming.
    Parameters
    ----------
    data : ndarray, shape (n_samples, n_variables)
        2-D array where each column corresponds to a variable and each row to a sample/observation.
        Values may be continuous or discrete. Floating or integer types are accepted.
    nbins : int or None, optional
        Number of bins to use for discretizing each variable. If None, the helper function
        `compute_bins` (expected to be available in the same module) is called to determine
        appropriate bin edges and a common nbins value. If an integer is provided, the same
        number of bins is used for every variable (but `compute_bins` is still used to
        compute the bin edges when nbins is None).
    Returns
    -------
    H : ndarray, shape (n_variables, n_variables)
        Symmetric matrix of entropies in bits (base-2). The diagonal entries H[i, i] contain
        the marginal entropy of variable i, and the off-diagonal entries H[i, j] = H[j, i]
        contain the joint entropy of variables (i, j).
    Notes
    -----
    - Discretization / binning:
      The function relies on `compute_bins(data, nbins)` to obtain bin edges (one per variable)
      and a resolved integer nbins. Observations are assigned to bins using
      `np.searchsorted` and clipped to the range [0, nbins-1]. Binning is performed once per
      variable and reused for all pairwise joint-histogram computations.
    - Histogram and entropy calculation:
      Marginal histograms are computed with `np.bincount`. Joint histograms for a pair (i, j)
      are built efficiently by encoding 2-D bin indices into a single index via
      `bin_i * nbins + bin_j`, followed by `np.bincount` and reshaping back to (nbins, nbins).
      Probabilities are estimated by dividing histogram counts by `n_samples`. Entropy is
      computed as -sum(p * log2(p)) over non-zero probability bins (zero-probability bins are
      ignored to avoid log(0)).
    - Numerical considerations:
      The function uses base-2 logarithm so returned entropies are in bits. Small sample sizes
      and coarse/fine binning choices will affect the bias/variance of the estimated entropies.
    Raises
    ------
    ValueError
        If `data` is not a 2-D array or if `n_samples == 0` or `n_variables == 0`.
    TypeError
        If `nbins` is provided but cannot be interpreted as an integer.
    Examples
    --------
    >>> # Given a 2-D numpy array `X` with shape (n_samples, n_variables):
    >>> H = joint_entropies_fast(X, nbins=16)
    >>> # H[i, i] -> marginal entropy of variable i (in bits)
    >>> # H[i, j] -> joint entropy of variables i and j (in bits)
    """
    n_samples, n_variables = data.shape
    edges, nbins = compute_bins(data, nbins)

    # Precompute bin indices once
    bin_idx = np.zeros((n_variables, n_samples), dtype=np.int32)
    for i in range(n_variables):
        idx = np.searchsorted(edges[i], data[:, i], side="right") - 1
        bin_idx[i] = np.clip(idx, 0, nbins - 1)

    H = np.zeros((n_variables, n_variables))

    # Marginal entropies
    for i in range(n_variables):
        hist = np.bincount(bin_idx[i], minlength=nbins)
        p = hist / n_samples
        m = p > 0
        H[i, i] = -(p[m] * np.log2(p[m])).sum()

    # Joint entropies
    for i in range(n_variables):
        for j in range(i + 1, n_variables):
            # Build joint histogram from pre-binned indices
            hist2d = np.bincount(
                bin_idx[i] * nbins + bin_idx[j], minlength=nbins * nbins
            ).reshape(nbins, nbins)

            pxy = hist2d / n_samples
            m = pxy > 0
            Hxy = -(pxy[m] * np.log2(pxy[m])).sum()
            H[i, j] = H[j, i] = Hxy

    return H


def mi_matrix(signals, nbins=None, normalized=False):
    """
    Compute the pairwise mutual information (MI) matrix for a set of signals.
    Parameters
    ----------
    signals : array-like, shape (n_variables, n_samples)
        2D array of input signals where each row corresponds to a variable (signal)
        and each column to a sample/timepoint. The function transposes the input
        (signals.T) internally to obtain shape (n_samples, n_variables) expected by
        the underlying entropy routine.
    nbins : int or sequence or None, optional
        Number of bins (or bin specification) used for discretization when
        estimating entropies. The exact interpretation is forwarded to
        `joint_entropies_fast`. If None, the default behavior of
        `joint_entropies_fast` is used.
    normalized : bool, optional
        If True, return the normalized mutual information (NMI) defined here as
        2 * I(X;Y) / (H(X) + H(Y)). If False (default), return raw mutual
        information values in bits/nats consistent with the entropy estimator.
    Returns
    -------
    MI : ndarray, shape (n_variables, n_variables)
        Symmetric matrix of pairwise mutual information values. Diagonal entries
        are set to 0.0. If `normalized=True`, values are in [0, 1] where possible;
        any non-finite results (e.g. divide-by-zero) are replaced with 0.0.
    Notes
    -----
    - This function relies on `joint_entropies_fast(data, nbins)` to compute the
      matrix of joint entropies H for all variable pairs. `H[i, i]` is treated as
      the marginal entropy H(X_i).
    - Mutual information is computed as MI(i, j) = H(X_i) + H(X_j) - H(X_i, X_j).
    - When normalized, the function uses: NMI(i, j) = 2 * MI(i, j) / (H(X_i) + H(X_j)).
      Any resulting non-finite values (NaN or Inf) are set to 0.0.
    Raises
    ------
    TypeError
        If `signals` is not array-like or cannot be transposed to a 2D array.
    ValueError
        If the output of `joint_entropies_fast` does not have a compatible shape.
    Example
    -------
    # given signals as a NumPy array of shape (n_variables, n_samples):
    # MI = mi_matrix(signals, nbins=16, normalized=True)
    """

    data = signals.T  # (n_samples, n_variables)
    H = joint_entropies_fast(data, nbins)
    h = H.diagonal()
    sum_h = h[:, None] + h[None, :]
    MI = sum_h - H
    if normalized:
        with np.errstate(divide="ignore", invalid="ignore"):
            MI = 2.0 * MI / sum_h
            MI[~np.isfinite(MI)] = 0.0
    np.fill_diagonal(MI, 0.0)
    return MI


def pack_upper(A: np.ndarray) -> np.ndarray:
    """
    Pack the strictly upper-triangular entries of a square 2-D array into a 1-D array.
    Parameters
    ----------
    A : numpy.ndarray
        Square 2-D input array of shape (n, n). The function requires a square matrix;
        a ValueError is raised otherwise.
    Returns
    -------
    numpy.ndarray
        1-D array of length n*(n-1)/2 containing the elements A[i, j] for 0 <= i < j < n.
        Elements are ordered according to np.triu_indices_from(A, k=1) (row-major order:
        row 0 then row 1, etc.). The returned array has the same dtype as A.
    Raises
    ------
    ValueError
        If A is not square.
    Notes
    -----
    - Diagonal elements are excluded (k=1).
    - For n <= 1 the result is an empty 1-D array.
    - The returned array is a copy (fancy indexing is used).
    Examples
    --------
    For A = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
    the result is array([2, 3, 6]).
    """

    if A.shape[0] != A.shape[1]:
        raise ValueError("Matrix must be square.")
    return A[np.triu_indices_from(A, k=1)]


def process_mi_weights(
    signal_folder,
    subject_id,
    rest_id,
    nbins_mi,
    normalized,
    base_folder,
):
    """
    Process and save mutual information (MI) connectivity weights for a subject/rest session.

    Parameters:
        signal_folder (str): Path to the .npy file containing the time-series signals to be processed.
            Note: despite the parameter name, this value is passed directly to numpy.load, so it must be
            a path acceptable to np.load (typically a .npy file).
        subject_id (str | int): Identifier for the subject; used to construct the output filename.
        rest_id (str | int): Identifier for the rest/session; used to construct the output filename.
        nbins_mi (int): Number of bins to use when computing mutual information via mi_matrix.
        normalized (bool): Whether to compute a normalized mutual information matrix.
        base_folder (str): Directory where the resulting MI file will be saved.

    Behavior:
        - Builds the output path as os.path.join(base_folder, f"{subject_id}_fMRI_REST{rest_id}.npy").
        - If that output file already exists, the function returns without further action.
        - Otherwise:
            1. Loads the input signals with numpy.load(signal_folder).
            2. Computes an MI matrix via mi_matrix(loaded_signals, nbins=nbins_mi, normalized=normalized).
            3. Packs the upper triangle of the MI matrix with pack_upper(...) and saves it to the output path
               using numpy.save.

    Returns:
        None

    Side effects:
        Writes a .npy file to disk at the constructed output path when the file does not already exist.

    Exceptions:
        Exceptions raised by os.path, numpy.load, mi_matrix, pack_upper, or numpy.save (e.g., FileNotFoundError,
        ValueError, IOError) are not caught and will propagate to the caller.

    Notes:
        - The functions mi_matrix and pack_upper must be defined or imported in the module where this function is used.
        - If signal_folder is actually a directory of files, the caller must load/aggregate the signals before calling
          this function or modify the argument to point to a single .npy file.
    """

    mi_file = os.path.join(
        base_folder,
        f"{subject_id}_fMRI_REST{rest_id}.npy",
    )
    if not (os.path.exists(mi_file)):
        matrix = mi_matrix(
            np.load(signal_folder), nbins=nbins_mi, normalized=normalized
        )
        np.save(mi_file, pack_upper(matrix))


def compute_mi_weights_from_time_series(
    input_dir, output_dir, nbins_mi=10, normalized=True
):
    """
    Compute mutual-information (MI) connectivity weights for a collection of time-series files.
    This function scans an input directory for NumPy (.npy) files whose filenames encode
    a subject identifier and REST run index using the pattern "<subject>_fMRI_REST<rest>.npy".
    For each matched file it schedules a parallel call to `process_mi_weights` (via joblib.Parallel)
    to compute and save MI-based connectivity weights into the specified output directory.
    Parameters
    ----------
    input_dir : str or os.PathLike
        Path to the directory containing time-series .npy files. Files must match the
        filename pattern "<subject>_fMRI_REST<rest>.npy" where <subject> and <rest> are integers.
    output_dir : str or os.PathLike
        Directory where output weight files (produced by `process_mi_weights`) will be written.
        The directory will be created if it does not already exist.
    nbins_mi : int, optional (default=10)
        Number of discrete bins to use when estimating mutual information from continuous
        time-series values. Passed through to `process_mi_weights`.
    normalized : bool, optional (default=True)
        Whether to compute a normalized variant of mutual information. Passed through to
        `process_mi_weights`.
    Returns
    -------
    None
        Results are written to disk by `process_mi_weights`. This function returns nothing.
    Raises
    ------
    ValueError
        If any .npy filename in `input_dir` does not match the required "<subject>_fMRI_REST<rest>.npy"
        naming pattern, a ValueError is raised identifying the offending file.
    Notes
    -----
    - The function uses glob("*.npy") to discover files and enforces the filename pattern via a regex.
    - Processing is parallelized with joblib.Parallel using all available CPUs (n_jobs=-1).
    - `process_mi_weights` is expected to accept the arguments
      (file_path, subject_id, rest_id, nbins_mi, normalized, output_dir) and handle saving outputs.
    - Input .npy files are expected to contain the time-series data in a format `process_mi_weights`
      can consume (e.g., a 2D array of shape [n_regions, n_timepoints] or similar).
    Examples
    --------
    Simple usage:
        compute_mi_weights_from_time_series("/path/to/time_series", "/path/to/output", nbins_mi=12, normalized=False)
    """

    files_time_series = glob(os.path.join(input_dir, "*.npy"))
    df_time_series = []
    pattern_time_series = re.compile(r"(\d+)_fMRI_REST(\d+)\.npy$")
    for f in files_time_series:
        m = pattern_time_series.search(os.path.basename(f))
        if not m:
            raise ValueError(f"Could not parse Subject/REST from {f}")
        subj = int(m.group(1))
        rest = int(m.group(2))
        df_time_series.append({"Subject": subj, "REST": rest, "File": f})
    df_time_series = (
        pd.DataFrame(df_time_series)
        .sort_values(by=["Subject", "REST"])
        .reset_index(drop=True)
    )

    os.makedirs(output_dir, exist_ok=True)

    signal_folders = df_time_series["File"].tolist()
    subject_ids = df_time_series["Subject"].tolist()
    rest_ids = df_time_series["REST"].tolist()
    Parallel(n_jobs=-1, verbose=10)(
        delayed(process_mi_weights)(
            signal_folder,
            subject_id,
            rest_id,
            nbins_mi,
            normalized,
            output_dir,
        )
        for signal_folder, subject_id, rest_id in zip(
            signal_folders, subject_ids, rest_ids
        )
    )


def process_oi_tc(
    signal_folder,
    triplets,
    out_file,
    zscored,
):
    """
    Process and save oscillatory interaction (OI) and transfer complexity (TC) measures
    extracted from multivariate time series data.

    Parameters
    ----------
    signal_folder : str
        Path to a NumPy .npy file (or any file readable by numpy.load) containing the
        signal array. The array is expected to be shaped (n_variables, n_samples) or
        equivalent such that it becomes (n_samples, n_variables) after transposition.
    triplets : array-like
        Iterable of index triplets (e.g., shape (n_triplets, 3)) that specify the
        variable combinations passed to `nplets_measures`. The exact format is
        determined by the implementation of `nplets_measures`.
    out_file : str
        Path where the resulting 2-column array (OI, TC) will be saved using
        numpy.save. If the filename does not include the ".npy" extension, NumPy
        will append it.
    zscored : bool
        If True, each column of the resulting (n_items, 2) array is z-scored
        (zero mean, unit variance) across rows before saving. If False, the raw
        OI and TC values are saved.

    Returns
    -------
    None
        The function writes the resulting array to disk and does not return a value.

    Side effects
    ------------
    - Loads data from disk using numpy.load.
    - Calls `nplets_measures(X, triplets)` to compute measures for the specified
      triplets.
    - Saves a 2-column NumPy array to `out_file` with columns [OI, TC] (optionally
      z-scored).

    Notes
    -----
    - The function expects `nplets_measures` to return an array-like object from
      which OI is extracted as measures[:, 0, 2] and TC as measures[:, 0, 0].
    - Input transposition (X = np.load(signal_folder).T) is applied so that the
      loaded array becomes (n_samples, n_variables). Ensure your input file shape
      matches this convention.
    - Any exceptions raised by numpy.load, nplets_measures, zscore, or numpy.save
      (e.g., file-not-found, shape errors, or invalid triplets) will propagate to
      the caller.

    Example
    -------
    >>> # Given "signals.npy" with shape (n_variables, n_samples) and a triplets array:
    >>> process_oi_tc("signals.npy", triplets, "oi_tc_results.npy", zscored=True)
    """

    X = np.load(signal_folder).T  # (n_samples, n_variables)
    measures = nplets_measures(X, triplets)
    oi = np.array(measures[:, 0, 2])
    tc = np.array(measures[:, 0, 0])
    oi_tc = np.vstack([oi, tc]).T
    if zscored:
        oi_tc = zscore(oi_tc, axis=0)
    np.save(out_file, oi_tc)


def compute_oi_tc_weights_from_time_series(
    input_dir, output_dir, hoi_labels, zscored=False, n_jobs=-1
):
    """
    Compute the OI and TC weights of HOIs from
    per-subject fMRI time-series files and save the results to an output
    directory. The function searches for .npy time-series files in input_dir
    whose basenames match the pattern "<Subject>_fMRI_REST<REST>.npy", parses
    Subject and REST identifiers, and processes only those inputs whose
    corresponding output files do not already exist in output_dir.
    Parameters
    ----------
    input_dir : str or os.PathLike
        Directory containing input NumPy (.npy) time-series files. Files must
        be named like "<subject>_fMRI_REST<rest>.npy" (e.g., "101_fMRI_REST1.npy").
    output_dir : str or os.PathLike
        Directory where the computed ROI TC weight files will be written. The
        function will create this directory if it does not exist. Output files
        are named "<subject>_fMRI_REST<rest>.npy".
    hoi_labels : array-like
        Labels (or indices) of the HOIs/ROIs to be used by the underlying
        processing function (process_oi_tc). The exact expected format is
        determined by process_oi_tc (commonly a list/array of region labels).
    zscored : bool, optional (default=False)
        If True, instruct the underlying processing to z-score signals before
        computing ROI TC weights. The flag is forwarded to process_oi_tc.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs to use when processing multiple files. Values
        follow joblib convention (e.g., -1 uses all available CPUs).
    Returns
    -------
    None
        This function writes output files to disk and returns nothing.
    Behavior and side effects
    -------------------------
    - Scans input_dir for files matching "*.npy".
    - Expects basenames to match the regex r"(\d+)_fMRI_REST(\d+)\.npy$" to
      extract integer Subject and REST identifiers. If any file does not
      match, a ValueError is raised.
    - Builds a sorted pandas.DataFrame of detected files and the parsed IDs.
    - Constructs corresponding output file paths inside output_dir using the
      same Subject and REST identifiers.
    - Skips processing for any input whose target output file already exists.
      If all outputs already exist, the function prints a message and returns.
    - Creates output_dir if it does not exist.
    - Uses joblib.Parallel + delayed to call process_oi_tc in parallel for
      the files that need processing:
        process_oi_tc(signal_file, hoi_labels, output_file, zscored=zscored)
    - Prints progress information about how many files will be processed and
      the Subject/REST of the first file in the worklist.
    Exceptions
    ----------
    ValueError
        Raised when a discovered input filename does not conform to the
        expected naming pattern and thus Subject/REST cannot be parsed.
    Notes
    -----
    - This function depends on the following imports/objects being available:
      os, re, glob (or glob.glob), pandas as pd, joblib.Parallel and
      joblib.delayed, and a callable process_oi_tc(signal_file, hoi_labels,
      output_file, zscored=...).
    - The precise format and contents of the saved output .npy files are
      determined by process_oi_tc.
    - The function avoids overwriting existing outputs; to force re-processing,
      remove or rename the existing output files first.
    """

    files_time_series = glob(os.path.join(input_dir, "*.npy"))
    df_time_series = []
    pattern_time_series = re.compile(r"(\d+)_fMRI_REST(\d+)\.npy$")
    for f in files_time_series:
        m = pattern_time_series.search(os.path.basename(f))
        if not m:
            raise ValueError(f"Could not parse Subject/REST from {f}")
        subj = int(m.group(1))
        rest = int(m.group(2))
        df_time_series.append({"Subject": subj, "REST": rest, "File": f})
    df_time_series = (
        pd.DataFrame(df_time_series)
        .sort_values(by=["Subject", "REST"])
        .reset_index(drop=True)
    )

    # prepare list of inputs whose outputs do NOT yet exist
    df_temp = df_time_series.copy()
    os.makedirs(output_dir, exist_ok=True)

    df_temp["Output"] = df_temp.apply(
        lambda r: os.path.join(
            output_dir, f"{int(r['Subject'])}_fMRI_REST{int(r['REST'])}.npy"
        ),
        axis=1,
    )

    # keep only rows where the target output file is missing
    df_temp = df_temp[~df_temp["Output"].apply(os.path.exists)].reset_index(drop=True)

    if df_temp.empty:
        print("No files to process: all outputs already exist in", output_dir)
        return
    else:
        print(
            f"Will process {len(df_temp)} files. Starting from Subject {df_temp.at[0, 'Subject']} REST {df_temp.at[0, 'REST']}."
        )

    signal_folders = df_temp["File"].tolist()
    output_files = df_temp["Output"].tolist()
    Parallel(n_jobs=n_jobs, verbose=10)(
        delayed(process_oi_tc)(
            signal_folder,
            hoi_labels,
            output_file,
            zscored=zscored,
        )
        for signal_folder, output_file in zip(signal_folders, output_files)
    )


def load_connectivity_files(directory: str, filename_regex: str = None) -> pd.DataFrame:
    """
    Recursively collect .npy files inside 'directory' and return those that match the given regex.

    - Subject: first integer capture group
    - REST: second integer capture group
    - Optional: third group (e.g., 'oi'/'tc') -> 'Metric'
    - Optional: fourth group (e.g., mode) -> 'Mode'
    - File: full path

    If filename_regex is None, tries common patterns and ignores non-matching files.
    """
    files = sorted(glob(os.path.join(directory, "**", "*.npy"), recursive=True))
    rows = []

    if filename_regex:
        pat = re.compile(filename_regex)
        for f in files:
            base = os.path.basename(f)
            m = pat.search(base)
            if not m:
                continue  # ignore non-matching files
            row = {
                "Subject": int(m.group(1)),
                "REST": int(m.group(2)),
                "File": f,
            }
            # Optional groups
            if m.lastindex and m.lastindex >= 3:
                row["Metric"] = m.group(3)  # e.g., 'ii' or 'tc'
            if m.lastindex and m.lastindex >= 4:
                try:
                    row["Mode"] = int(m.group(4))
                except (TypeError, ValueError):
                    row["Mode"] = m.group(4)
            rows.append(row)

        if not rows:
            raise ValueError(
                f"No files in '{directory}' matched pattern: {filename_regex}"
            )
    else:
        patterns = [
            re.compile(r)
            for r in (
                r"(\d+)_fMRI_REST(\d+)_A_(ii|tc)_(\d+)",
                r"(\d+)_fMRI_REST(\d+)",
                r"(\d+)_REST(\d+)",
                r"^(\d+).*?(\d+)",
            )
        ]
        for f in files:
            base = os.path.basename(f)
            for pat in patterns:
                m = pat.search(base)
                if m:
                    row = {
                        "Subject": int(m.group(1)),
                        "REST": int(m.group(2)),
                        "File": f,
                    }
                    if m.lastindex and m.lastindex >= 3:
                        row["Metric"] = m.group(3)
                    if m.lastindex and m.lastindex >= 4:
                        try:
                            row["Mode"] = int(m.group(4))
                        except (TypeError, ValueError):
                            row["Mode"] = m.group(4)
                    rows.append(row)
                    break
        if not rows:
            raise ValueError(f"No parsable files found in '{directory}'")

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["Subject", "REST", "Metric", "Mode"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def compute_mean_weights(df, output_file, zscored=False):
    """
    Compute the element-wise mean of numpy arrays referenced in a DataFrame and save the result.
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing a column "File" with filesystem paths (strings or Path-like) to .npy files to be averaged.
    output_file : str or os.PathLike
        Destination path where the computed average array will be saved via numpy.save. If the file exists it will be overwritten.
    zscored : bool, optional
        If True, each loaded array is z-scored along axis=0 using scipy.stats.zscore before accumulation. Default is False.
    Returns
    -------
    None
        The averaged array is written to `output_file` and the function returns None.
    Raises
    ------
    ValueError
        If the DataFrame contains no files (i.e., the "File" list is empty).
    OSError, IOError
        If loading (np.load) or saving (np.save) files fails due to I/O issues.
    ValueError, numpy.AxisError, or broadcasting-related exceptions
        If arrays cannot be z-scored along axis=0 or if loaded arrays have incompatible shapes for element-wise summation.
    Side effects
    ------------
    - Iteratively loads each .npy file in df["File"] and accumulates a running sum (a tqdm progress bar is displayed).
    - Optionally applies z-scoring to each loaded array before accumulation.
    - Saves the computed average array to `output_file` using numpy.save.
    - Prints a confirmation message indicating where the average was saved.
    Notes
    -----
    - All arrays must have the same shape for element-wise averaging; otherwise the operation will raise a runtime error.
    - z-scoring requires scipy.stats.zscore to be available in the environment.
    - Memory usage includes at least one full array for the running sum (same shape as inputs) plus whichever array is currently loaded.
    - If any file fails to load, an exception is raised and no average is saved.
    Examples
    --------
    >>> # Basic usage
    >>> compute_mean_weights(df, "mean_weights.npy")
    >>> # With z-scoring applied before averaging
    >>> compute_mean_weights(df, Path("mean_weights_zscored.npy"), zscored=True)
    """

    files = df["File"].tolist()
    if not files:
        raise ValueError("No .npy files found")

    sum_weights = None
    count = 0

    for file in tqdm(files, desc="Computing mean weights"):
        weights_data = np.load(file)
        if zscored:
            weights_data = zscore(weights_data, axis=0)
        if sum_weights is None:
            sum_weights = np.zeros_like(weights_data)
        sum_weights += weights_data
        count += 1

    avg_weights = sum_weights / count
    np.save(output_file, avg_weights)
    print(f"Average weights saved to {output_file}")
    return


def process_ttensors(
    hoi_folder, subject_id, rest_id, hoi_labels, base_folder, modes, absolute=True
):
    """
    Process tensor modes and save per-mode connectivity arrays to disk.
    This function loads symmetrized tensors for a subject/rest session (by
    calling get_symmetrized_t_fft), optionally takes absolute values of each mode slice,
    packs the upper-triangular part of each mode matrix using pack_upper, and saves the
    resulting arrays to .npy files under mode-specific subfolders in base_folder.
    Behavior summary:
    - If all expected output files for all requested modes already exist, the function
        prints a message and returns immediately (no processing or overwriting).
    - Otherwise, it obtains As_ii and As_tc from get_symmetrized_t_fft(hoi_folder, hoi_labels),
        extracts the requested mode slices, optionally applies absolute value, packs the
        upper triangle with pack_upper, and writes two .npy files per mode:
            {base_folder}/mode_{mode}/{subject_id}_fMRI_REST{rest_id}_A_oi_{mode}.npy
            {base_folder}/mode_{mode}/{subject_id}_fMRI_REST{rest_id}_A_tc_{mode}.npy
    Parameters
    ----------
    hoi_folder : str
            Path to the folder containing higher-order interaction (HOI) data required by
            get_symmetrized_t_fft.
    subject_id : str
            Subject identifier used to construct output filenames.
    rest_id : int or str
            Rest/session identifier appended to filenames (formatted as REST{rest_id}).
    hoi_labels : sequence
            Labels or identifiers passed to get_symmetrized_t_fft to retrieve the tensors.
    base_folder : str
            Base directory where per-mode subdirectories and output .npy files will be saved.
    modes : sequence of int
            Iterable of integer mode indices to extract from the third axis of the returned
            tensors (As_ii and As_tc). Each mode produces one pair of output files.
    absolute : bool, optional (default True)
            If True, take the elementwise absolute value of each mode matrix before packing
            and saving. If False, the raw values are saved.
    Returns
    -------
    None
            The function has no return value. Its primary effect is writing .npy files to disk
            and printing a status message when skipping processing because files already exist.
    Side effects
    ------------
    - Calls get_symmetrized_t_fft(hoi_folder, hoi_labels) and relies on its output shapes:
        As_ii and As_tc are expected to be 3D arrays with shape (n_nodes, n_nodes, n_modes).
    - Calls pack_upper(matrix) on each 2D mode-slice prior to saving; the saved arrays
        therefore represent packed upper-triangular data (not full square matrices).
    - Writes files using numpy.save; existing files are not overwritten if all expected
        files are present at function start.
    Exceptions
    ----------
    - Any exceptions raised by get_symmetrized_t_fft, pack_upper, numpy.save, or os.path
        operations will propagate. Typical errors include file I/O errors or indexing errors
        if the provided mode indices are out of range for the returned tensors.
    """

    A_oi_modes_dir = [
        os.path.join(
            base_folder,
            f"mode_{mode}",
            f"{subject_id}_fMRI_REST{rest_id}_A_oi_{mode}.npy",
        )
        for mode in modes
    ]

    A_tc_modes_dir = [
        os.path.join(
            base_folder,
            f"mode_{mode}",
            f"{subject_id}_fMRI_REST{rest_id}_A_tc_{mode}.npy",
        )
        for mode in modes
    ]

    if all(os.path.exists(f) for f in A_oi_modes_dir + A_tc_modes_dir):
        print(
            f"All mode files already exist for Subject {subject_id} REST {rest_id}. Skipping."
        )
        return

    As_ii, As_tc = get_symmetrized_t_fft(
        hoi_folder,
        hoi_labels,
    )

    A_oi_modes = []
    A_tc_modes = []
    for mode in modes:
        if absolute:
            A_oi_modes.append(np.abs(As_ii[:, :, mode]))
            A_tc_modes.append(np.abs(As_tc[:, :, mode]))
        else:
            A_oi_modes.append(As_ii[:, :, mode])
            A_tc_modes.append(As_tc[:, :, mode])

    for i, mode in enumerate(modes):
        np.save(A_oi_modes_dir[i], pack_upper(A_oi_modes[i]))
        np.save(A_tc_modes_dir[i], pack_upper(A_tc_modes[i]))


def compute_hypergraph_modes(
    hoi_dir, output_dir, hoi_labels, modes, absolute=True, n_jobs=-1
):
    """
    Compute and save hypergraph modes for a collection of HOI (.npy) files.
    This function scans a directory containing HOI files named with the pattern
    "<subject>_fMRI_REST<rest>.npy", sorts them by subject and rest session, creates
    per-mode output subdirectories under output_dir, and processes each HOI file by
    calling process_ttensors. Processing is performed either sequentially (with a
    tqdm progress bar) when n_jobs == 1 or in parallel using joblib.Parallel for
    other n_jobs values.
    Parameters
    ----------
    hoi_dir : str or os.PathLike
        Path to a directory containing HOI files. Files are expected to be NumPy
        .npy files named using the regex r"(\d+)_fMRI_REST(\d+)\.npy$" where the first
        captured group is the Subject ID (integer) and the second is the REST/session
        ID (integer).
    output_dir : str or os.PathLike
        Directory where results will be written. The function will create this
        directory if it does not exist, and will create a subdirectory named
        "mode_{mode}" for each entry in modes.
    hoi_labels : sequence
        Labels or metadata associated with HOI entries that will be forwarded to
        process_ttensors. Expected shape/type depends on process_ttensors.
    modes : sequence of int
        Mode indices to process. For each mode a corresponding subdirectory
        (output_dir/mode_{mode}) will be created and results for that mode will be
        saved there by process_ttensors.
    absolute : bool, optional (default=True)
        If True, indicates that absolute values should be used when computing modes.
        The exact behavior is passed through to process_ttensors.
    n_jobs : int, optional (default=-1)
        Number of parallel workers to use. If n_jobs == 1 the function runs
        sequentially and shows a tqdm progress bar. For other values it uses
        joblib.Parallel with verbose=10 to run process_ttensors concurrently.
    Returns
    -------
    None
        Results are written to disk under output_dir; no value is returned.
    Raises
    ------
    ValueError
        If any file in hoi_dir does not match the expected filename pattern and
        therefore the Subject/REST IDs cannot be parsed.
    OSError
        If output directory creation fails due to filesystem permissions or similar
        errors.
    Exception
        Any exception raised by process_ttensors (e.g., I/O or processing errors)
        will propagate to the caller.
    Notes
    -----
    - The function relies on process_ttensors to perform the actual computation and
      saving of mode-specific outputs. Ensure that process_ttensors is importable
      and accepts the parameters: (hoi_file, subject_id, rest_id, hoi_labels,
      output_dir, modes=modes, absolute=absolute).
    - Input HOI files must be valid NumPy .npy files containing the expected HOI
      data structure.
    - When running in parallel, file ordering is determined by the sorted list of
      discovered files but processing may complete out-of-order.
    """

    files_hoi = glob(os.path.join(hoi_dir, "*.npy"))
    df_hois = []
    pattern_hoi = re.compile(r"(\d+)_fMRI_REST(\d+)\.npy$")
    for f in files_hoi:
        m = pattern_hoi.search(os.path.basename(f))
        if not m:
            raise ValueError(f"Could not parse Subject/REST from {f}")
        subj = int(m.group(1))
        rest = int(m.group(2))
        df_hois.append({"Subject": subj, "REST": rest, "File": f})
    df_hois = (
        pd.DataFrame(df_hois).sort_values(by=["Subject", "REST"]).reset_index(drop=True)
    )

    os.makedirs(output_dir, exist_ok=True)
    for mode in modes:
        os.makedirs(os.path.join(output_dir, f"mode_{mode}"), exist_ok=True)

    hoi_folders = df_hois["File"].tolist()
    subject_ids = df_hois["Subject"].tolist()
    rest_ids = df_hois["REST"].tolist()
    if n_jobs == 1:
        for hoi_folder, subject_id, rest_id in tqdm(
            zip(hoi_folders, subject_ids, rest_ids), total=len(hoi_folders)
        ):
            process_ttensors(
                hoi_folder,
                subject_id,
                rest_id,
                hoi_labels,
                output_dir,
                modes=modes,
                absolute=absolute,
            )
    else:
        Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(process_ttensors)(
                hoi_folder,
                subject_id,
                rest_id,
                hoi_labels,
                output_dir,
                modes=modes,
                absolute=absolute,
            )
            for hoi_folder, subject_id, rest_id in zip(
                hoi_folders, subject_ids, rest_ids
            )
        )
