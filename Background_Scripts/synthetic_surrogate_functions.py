import os
import re
from glob import glob

import numpy as np
import pandas as pd
from hypergraph_connectivity_functions import all_triangles
from joblib import Parallel, delayed
from numpy.random import default_rng
from scipy.stats import skewnorm
from scipy.stats import t as student_t
from spatiotemporal import phase_randomize
from statsmodels.tsa.api import VAR
from tqdm import tqdm


def generate_phase_randomized_time_series(input_dir, output_dir, seed=42):
    """
    Generate phase-randomized surrogate time series for .npy files in a directory.
    This function locates NumPy (.npy) time-series files in input_dir whose names
    match the pattern "<subject>_fMRI_REST<rest>.npy" (both subject and rest are
    integers), applies phase_randomize(...) to each loaded array, and saves the
    resulting surrogate arrays to output_dir with the same filename pattern.
    If output_dir does not exist it will be created.
    Parameters
    ----------
    input_dir : str
        Path to the directory containing the original .npy time-series files.
    output_dir : str
        Path to the directory where phase-randomized .npy files will be written.
    seed : int, optional
        Random seed passed to phase_randomize for reproducibility (default: 42).
    Returns
    -------
    None
    Raises
    ------
    ValueError
        If any found file does not match the expected filename pattern and thus
        its Subject/REST cannot be parsed.
    OSError, IOError
        If filesystem operations (reading/writing files or creating directories)
        fail.
    Any
        Exceptions raised by numpy.load, numpy.save, or phase_randomize will be
        propagated to the caller.
    Notes
    -----
    - Files are discovered using a glob for "*.npy" and then filtered/parsed using
      the regex r"(\d+)_fMRI_REST(\d+)\.npy$".
    - The discovered files are sorted by Subject and REST before processing.
    - Progress is reported via tqdm.
    - The same seed is forwarded to each call to phase_randomize; depending on the
      implementation of phase_randomize this may produce identical or
      reproducible surrogates across files.
    Example
    -------
    >>> generate_phase_randomized_time_series("data/original", "data/surrogates", seed=123)
    """

    files_time_series = glob(os.path.join(input_dir, "*.npy"))
    df_time_series = []
    pattern_time_series = re.compile(r"(\d+)_fMRI_REST(\d+)\.npy$")
    for f in files_time_series:
        m = pattern_time_series.search(os.path.basename(f))
        if not m:
            raise ValueError(f"Could not parse Subject/REST/bin from {f}")
        subj = int(m.group(1))
        rest = int(m.group(2))
        df_time_series.append({"Subject": subj, "REST": rest, "File": f})
    df_time_series = (
        pd.DataFrame(df_time_series)
        .sort_values(by=["Subject", "REST"])
        .reset_index(drop=True)
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for idx, row in tqdm(
        df_time_series.iterrows(),
        total=len(df_time_series),
        desc="Generating phase randomized time series",
    ):
        time_series = np.load(row["File"])
        time_series_pr = phase_randomize(time_series, seed=seed)
        out_file = os.path.join(
            output_dir, f"{row['Subject']}_fMRI_REST{row['REST']}.npy"
        )
        np.save(out_file, time_series_pr)


# ---------- helpers ----------
def _triplets_from_dict(weights_dict):
    keys = sorted((i, j, k) for (i, j, k) in weights_dict.keys())
    I = np.array([i for i, _, _ in keys], dtype=np.int32)
    J = np.array([j for _, j, _ in keys], dtype=np.int32)
    K = np.array([k for _, _, k in keys], dtype=np.int32)
    w = np.array([float(weights_dict[(i, j, k)]) for (i, j, k) in keys], dtype=float)
    return I, J, K, w


def _rescale_weights(w, mode="l2", eps=1e-12):
    if (mode is None) or (not np.any(w)):
        return w
    if mode == "l2":
        return w / (np.linalg.norm(w) + eps)
    if mode == "max":
        return w / (np.max(np.abs(w)) + eps)
    if mode == "sum":
        return w / (np.sum(np.abs(w)) + eps)
    return w


def _sample_II_like(w, rng, skew_scale=1.0):
    if w.size == 0 or not np.any(w):
        return np.zeros_like(w)
    alphas = skew_scale * np.sign(w) * np.abs(w)
    s = skewnorm.rvs(a=alphas, loc=0.0, scale=1.0, size=w.size, random_state=rng)
    return s * np.abs(w)


def _sample_TC_like(w, rng, t_df=7, t_scale=1.0):
    if w.size == 0 or not np.any(w):
        return np.zeros_like(w)
    s = student_t.rvs(df=t_df, loc=0.0, scale=t_scale, size=w.size, random_state=rng)
    return s * np.abs(w)


def _accumulate_to_nodes(n, I, J, K, s, out=None, scale=1 / np.sqrt(3)):
    V = out if out is not None else np.zeros(n, dtype=float)
    V.fill(0.0)
    np.add.at(V, I, s)
    np.add.at(V, J, s)
    np.add.at(V, K, s)
    if scale != 1.0:
        V *= scale
    return V


# ---------- VAR(p) + diagonal white noise + OI + TC (both) ----------
def multivariate_VAR_with_both_triads_diag_noise(
    X,
    order=1,
    seed=None,
    # weights of triplets (each optional)
    weights_triplets_oi=None,  # dict {(i,j,k): OI}
    weights_triplets_tc=None,  # dict {(i,j,k): TC}
    # global gains
    psi_oi=0.0,
    psi_tc=0.0,
    # rescale mode for each family of weights
    rescale_mode_oi="l2",
    rescale_mode_tc="l2",
    # noise hyperparameters
    skew_scale=1.0,  # only for OI
    t_df=7,
    t_scale=1.0,  # only for TC
    # mini-batch per family
    batch_ratio_oi=1.0,
    batch_ratio_tc=1.0,
):
    """Generate a surrogate multivariate time series from a fitted VAR(p) model
    with additive diagonal Gaussian innovations and optional triadic contributions
    from two families of triadic interactions: OI (ordinal/influence-like) and
    TC (heavy-tailed/t-distributed contributions).
    Parameters
    ----------
    X : array_like, shape (n, T)
        Observed multivariate time series used to fit the VAR model. Rows correspond
        to variables (n), columns to time points (T). The function fits a VAR on
        this data and uses its coefficients and residual standard deviations to
        construct the surrogate.
    order : int, optional (default=1)
        Autoregressive order p used to fit the VAR(p) model.
    seed : int or numpy.random.Generator, optional
        Seed or RNG to initialize the internal random number generator. If None,
        a nondeterministic RNG is used.
    weights_triplets_oi : dict or None, optional
        Dictionary defining OI triadic interactions. Expected format is a mapping
        where keys identify triplets (i, j, k) and values are the associated
        raw weights. The helper _triplets_from_dict is used internally to convert
        this dict into arrays of indices and weights. If None, no OI contribution
        is added.
    weights_triplets_tc : dict or None, optional
        Dictionary defining TC triadic interactions, analogous to
        weights_triplets_oi. If None, no TC contribution is added.
    psi_oi : float, optional (default=0.0)
        Global gain (scalar multiplier) applied to the OI triadic contribution.
        If zero, OI terms are skipped.
    psi_tc : float, optional (default=0.0)
        Global gain (scalar multiplier) applied to the TC triadic contribution.
        If zero, TC terms are skipped.
    rescale_mode_oi : str, optional (default="l2")
        Mode passed to the internal _rescale_weights for OI weights. Controls how
        raw triplet weights are normalized before sampling. See _rescale_weights for
        supported modes (default behavior is L2 normalization).
    rescale_mode_tc : str, optional (default="l2")
        Same as rescale_mode_oi but for TC weights.
    skew_scale : float, optional (default=1.0)
        Scale parameter passed to the OI sampler (_sample_II_like). Controls the
        magnitude of skew/shape for OI-generated triadic contributions.
    t_df : float, optional (default=7)
        Degrees of freedom for the TC sampler (_sample_TC_like). Lower values
        produce heavier tails.
    t_scale : float, optional (default=1.0)
        Scale parameter for the TC sampler (_sample_TC_like).
    batch_ratio_oi : float in (0, 1] or 1.0, optional (default=1.0)
        If < 1.0 and there are m OI triplets, a random subset of size
        ceil(batch_ratio_oi * m) is sampled (without replacement) at each time
        step and rescaled by 1 / batch_ratio_oi to approximate the full-sum
        contribution. If 1.0, all triplets are used every step.
    batch_ratio_tc : float in (0, 1] or 1.0, optional (default=1.0)
        Same as batch_ratio_oi but for TC triplets.
    Returns
    -------
    surrogate : ndarray, shape (n, T)
        Simulated surrogate multivariate time series. The first `order` columns are
        initialized by copying a random contiguous block of `order` observations
        from the original data X. For t >= order, observations are generated as:
          surrogate[t] = sum_{lag=1..p} A_lag @ surrogate[t-lag]
                         + diagonal Gaussian innovation (drawn from N(0, resid_std^2))
                         + psi_oi * Vt_oi + psi_tc * Vt_tc
        where A_lag are VAR coefficients fitted to X, resid_std are the
        per-variable residual standard deviations, and Vt_oi / Vt_tc are node-wise
        accumulations of triadic contributions for OI and TC respectively.
    Notes
    -----
    - The function fits a VAR(order) model to X (using statsmodels VAR internally)
      and uses the estimated coefficients and residual standard deviations to
      construct the baseline linear and diagonal-noise components.
    - OI triadic samples are generated by _sample_II_like and can incorporate
      skew via `skew_scale`. TC triadic samples are generated by
      _sample_TC_like and have t-distributed heavy tails controlled by (t_df,
      t_scale).
    - Triplet dictionaries are converted to index arrays (I, J, K) and weight
      arrays by the internal helper _triplets_from_dict; weights are then
      rescaled with _rescale_weights according to the chosen rescale mode.
    - When batching is used (batch_ratio_* < 1.0) a random subset of triplets is
      used at each time step and the summed contribution is scaled by 1 / batch_ratio
      to keep the expected contribution approximately equal to the full-sum.
    - If a family of triads is absent (weights_* is None) or its global gain
      psi_* is zero, that term is omitted.
    - Randomness (start block selection, innovations, triadic sampling) is
      controlled by `seed` for reproducibility.
    Raises
    ------
    ValueError
        If X has incompatible shape or if internal helpers raise on invalid
        weight dictionaries or rescale modes.
    See also
    --------
    _rescale_weights, _triplets_from_dict, _sample_II_like, _sample_TC_like,
    _accumulate_to_nodes"""

    rng = default_rng(seed)
    data = X.T  # (T, n)
    T, n = data.shape

    # --- VAR(p) ---
    model = VAR(data)
    results = model.fit(order)
    coefs = results.coefs  # (order, n, n)
    residuals = results.resid  # (T-order, n)
    resid_std = residuals.std(axis=0, ddof=1)  # (n,)

    surrogate = np.zeros_like(data)
    start_idx = rng.integers(0, T - order + 1)
    surrogate[:order] = data[start_idx : start_idx + order]
    white_resid = rng.standard_normal(size=residuals.shape) * resid_std  # diag noise

    # --- Preparation OI ---
    if weights_triplets_oi is not None:
        Ioi, Joi, Koi, woi_raw = _triplets_from_dict(weights_triplets_oi)
        woi = _rescale_weights(woi_raw, rescale_mode_oi)
        moi = woi.size
        qoi = float(batch_ratio_oi)
        use_batch_oi = (qoi < 1.0) and (moi > 0)
        boi = max(1, int(np.ceil(qoi * moi))) if use_batch_oi else moi
        inv_qoi = (1.0 / qoi) if use_batch_oi else 1.0
        Vbuf_oi = np.zeros(n, dtype=float)
    else:
        moi = 0

    # --- Preparation TC ---
    if weights_triplets_tc is not None:
        Itc, Jtc, Ktc, wtc_raw = _triplets_from_dict(weights_triplets_tc)
        wtc = _rescale_weights(wtc_raw, rescale_mode_tc)
        mtc = wtc.size
        qtc = float(batch_ratio_tc)
        use_batch_tc = (qtc < 1.0) and (mtc > 0)
        btc = max(1, int(np.ceil(qtc * mtc))) if use_batch_tc else mtc
        inv_qtc = (1.0 / qtc) if use_batch_tc else 1.0
        Vbuf_tc = np.zeros(n, dtype=float)
    else:
        mtc = 0

    # --- Simulation ---
    for t in range(order, T):
        # VAR(p) part
        for lag in range(1, order + 1):
            surrogate[t] += coefs[lag - 1] @ surrogate[t - lag]
        # Diagonal Gaussian innovation
        surrogate[t] += white_resid[t - order]

        # Term OI (if any)
        if moi > 0 and psi_oi != 0.0:
            if use_batch_oi:
                sel = rng.choice(moi, size=boi, replace=False)
                s_sel = _sample_II_like(woi[sel], rng, skew_scale=skew_scale)
                _accumulate_to_nodes(
                    n, Ioi[sel], Joi[sel], Koi[sel], s_sel, out=Vbuf_oi
                )
                Vt_oi = inv_qoi * Vbuf_oi
            else:
                s_all = _sample_II_like(woi, rng, skew_scale=skew_scale)
                _accumulate_to_nodes(n, Ioi, Joi, Koi, s_all, out=Vbuf_oi)
                Vt_oi = Vbuf_oi
        else:
            Vt_oi = 0.0

        # Term TC (if any)
        if mtc > 0 and psi_tc != 0.0:
            if use_batch_tc:
                sel = rng.choice(mtc, size=btc, replace=False)
                s_sel = _sample_TC_like(wtc[sel], rng, t_df=t_df, t_scale=t_scale)
                _accumulate_to_nodes(
                    n, Itc[sel], Jtc[sel], Ktc[sel], s_sel, out=Vbuf_tc
                )
                Vt_tc = inv_qtc * Vbuf_tc
            else:
                s_all = _sample_TC_like(wtc, rng, t_df=t_df, t_scale=t_scale)
                _accumulate_to_nodes(n, Itc, Jtc, Ktc, s_all, out=Vbuf_tc)
                Vt_tc = Vbuf_tc
        else:
            Vt_tc = 0.0

        # Sum of both contributions
        surrogate[t] += (psi_oi * Vt_oi) + (psi_tc * Vt_tc)

    return surrogate.T  # (n, T)


def process_synthetic_time_series(
    df_time_series_hois, output_dir, psi_oi, psi_tc, order=1, seed=42
):
    """
    Process synthetic time series for a set of subjects and save the generated arrays to disk.
    For each row in df_time_series_hois this function:
    - Constructs an output filename of the form "{Subject}_fMRI_REST{REST}.npy" within output_dir.
    - Skips processing when the output file already exists (prints a message and continues).
    - Loads a time series array from the path in row["File_ts"] and a HOI (higher-order interactions)
        array from the path in row["File_hoi"] (both expected to be NumPy .npy files).
    - Builds two dictionaries mapping triplets (a, b, c) from the global all_triangles sequence to
        OI and TC weights taken from hoi[:, 0] and hoi[:, 1], respectively.
    - Calls multivariate_VAR_with_both_triads_diag_noise(...) with those weight dictionaries and a
        fixed set of additional parameters to synthesize a new multivariate time series.
    - Saves the generated time series as a NumPy .npy file at the constructed output path.
    Parameters
    ----------
    df_time_series_hois : pandas.DataFrame
            Table describing input time series and HOI files. Must contain at least the columns:
            - "Subject": identifier used in the output filename
            - "REST": string/number used in the output filename
            - "File_ts": filesystem path to a NumPy .npy file containing the observed time series
            - "File_hoi": filesystem path to a NumPy .npy file containing HOI weights
    output_dir : str or pathlib.Path
            Directory where synthetic .npy files will be written. The directory should exist and be
            writable. Files are named "{Subject}_fMRI_REST{REST}.npy".
    psi_oi : float
            Parameter controlling the contribution/strength of OI (odd-interaction) triadic effects
            passed into the generator function.
    psi_tc : float
            Parameter controlling the contribution/strength of TC (triadic-cokurtosis) triadic effects
            passed into the generator function.
    order : int, optional (default=1)
            VAR model order passed to the multivariate generator (number of lags).
    seed : int, optional (default=42)
            Random seed used to make generation reproducible.
    Returns
    -------
    None
            The function has side effects only: it writes one .npy file per processed row to output_dir.
            When an output file already exists it is left unchanged and processing for that row is skipped.
    Notes and expectations
    ----------------------
    - The function expects that the variables all_triangles and multivariate_VAR_with_both_triads_diag_noise
        are available in the calling module's scope.
    - The HOI NumPy array is expected to have shape (n_triangles, 2) (first column -> OI, second -> TC).
    - The time series NumPy array should be compatible with the generator function (typically a 2D
        array with shape corresponding to timepoints × channels or channels × timepoints depending on
        your generator's API).
    - If an input file path does not exist, a FileNotFoundError (or NumPy load error) will be raised.
    - If shapes or number of triangles do not match, the generator or subsequent code may raise
        a ValueError or IndexError.
    """

    for idx, row in df_time_series_hois.iterrows():
        out_file = os.path.join(
            output_dir, f"{row['Subject']}_fMRI_REST{row['REST']}.npy"
        )
        if os.path.exists(out_file):
            print(f"File {out_file} already exists. Skipping.")
            continue

        time_series = np.load(row["File_ts"])
        hoi = np.load(row["File_hoi"])
        dict_weights_oi = {}
        dict_weights_tc = {}
        for i, (a, b, c) in enumerate(all_triangles):
            dict_weights_oi[(a, b, c)] = hoi[i, 0]
            dict_weights_tc[(a, b, c)] = hoi[i, 1]

        time_series_synthetic = multivariate_VAR_with_both_triads_diag_noise(
            time_series,
            order=order,
            seed=seed,
            weights_triplets_oi=dict_weights_oi,  # {(i,j,k): OI}
            weights_triplets_tc=dict_weights_tc,  # {(i,j,k): TC}
            psi_oi=psi_oi,
            psi_tc=psi_tc,
            rescale_mode_oi="l2",
            rescale_mode_tc="l2",
            skew_scale=1.2,  # ↑ = more symmetry (coskewness) for OI
            t_df=7,
            t_scale=1.0,  # ↓df = heavier tails (↑ cokurtosis) for TC
            batch_ratio_oi=0.05,  # mini-batch 5% for OI
            batch_ratio_tc=0.05,  # mini-batch 5% for TC
        )
        np.save(out_file, time_series_synthetic)


def generate_synthetic_time_series(
    input_dir, triangle_weights_dir, output_dir, psi_oi, psi_tc, order=1, seed=42
):
    """
    Generate synthetic time series by pairing raw time-series files with corresponding
    triangle-weight (higher-order interaction) files, then delegating per-pair
    processing to process_synthetic_time_series in parallel.
    The function:
    - Scans input_dir for NumPy (.npy) files with names matching the pattern
        "<Subject>_fMRI_REST<REST>.npy".
    - Scans triangle_weights_dir for .npy files using the same filename pattern.
    - Builds two dataframes (one for time series, one for HOI/triangle weights),
        extracts Subject and REST identifiers from filenames, and merges them on
        (Subject, REST).
    - Ensures output_dir exists (creates it if necessary).
    - Launches parallel processing (joblib.Parallel, n_jobs=-1) calling
        process_synthetic_time_series for each matched pair (a single-row DataFrame
        is passed to the worker), forwarding psi_oi, psi_tc, order, and seed.
    Parameters
    ----------
    input_dir : str
            Path to the directory containing raw time-series .npy files. Files must be
            named like "123_fMRI_REST1.npy" where the leading integer is a subject id
            and the terminal integer is the REST/session id.
    triangle_weights_dir : str
            Path to the directory containing triangle/HOI .npy files. Filenames must
            follow the same "<Subject>_fMRI_REST<REST>.npy" convention so they can be
            matched to the time-series files.
    output_dir : str
            Directory where per-pair synthetic outputs will be written by the worker
            function. Will be created if it does not exist.
    psi_oi :
            Object providing the "outer information" parameter(s) required by
            process_synthetic_time_series. Typical uses pass an array-like (e.g.,
            numpy.ndarray) or a configuration object — this function forwards psi_oi
            unchanged to the worker, so it must be in the expected format.
    psi_tc :
            Object providing the "temporal coupling" parameter(s) required by
            process_synthetic_time_series. Like psi_oi, this is forwarded as-is and
            must conform to the worker's expectations.
    order : int, optional
            Integer order parameter forwarded to process_synthetic_time_series (default: 1).
    seed : int, optional
            Random seed forwarded to process_synthetic_time_series to control
            deterministic behavior (default: 42).
    Returns
    -------
    None
            This function performs side effects (creating output files in output_dir)
            and does not return a value.
    Raises
    ------
    ValueError
            If any .npy file in input_dir or triangle_weights_dir does not match the
            required filename pattern and therefore cannot be parsed into Subject and
            REST identifiers.
    OSError
            If output_dir cannot be created due to filesystem permissions or other OS
            errors (propagated from os.makedirs).
    Side effects
    ------------
    - Creates output_dir if it does not exist.
    - Launches joblib.Parallel with n_jobs=-1 (uses all available CPU cores) and
        verbose=10; this may spawn multiple worker processes/threads depending on the
        environment and joblib backend.
    - Calls process_synthetic_time_series for each matched (Subject, REST) pair;
        that function is responsible for loading the .npy files, generating the
        synthetic series, and writing results to output_dir.
    Notes
    -----
    - The filename regex used is r"(\d+)_fMRI_REST(\d+)\.npy$". Filenames that do
        not conform will trigger a ValueError.
    - Matching between time-series and triangle-weight files is strict: only pairs
        that share identical Subject and REST identifiers are processed.
    - process_synthetic_time_series is expected to accept a single-row DataFrame
        (with columns "Subject", "REST", "File_ts", "File_hoi"), plus the psi_oi,
        psi_tc, order, and seed parameters.
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

    files_hoi = glob(os.path.join(triangle_weights_dir, "*.npy"))
    df_hoi = []
    pattern_hoi = re.compile(r"(\d+)_fMRI_REST(\d+)\.npy$")
    for f in files_hoi:
        m = pattern_hoi.search(os.path.basename(f))
        if not m:
            raise ValueError(f"Could not parse Subject/REST from {f}")
        subj = int(m.group(1))
        rest = int(m.group(2))
        df_hoi.append({"Subject": subj, "REST": rest, "File": f})
    df_hoi = (
        pd.DataFrame(df_hoi).sort_values(by=["Subject", "REST"]).reset_index(drop=True)
    )

    df_time_series_hois = pd.merge(
        df_time_series,
        df_hoi,
        on=["Subject", "REST"],
        how="inner",
        suffixes=("_ts", "_hoi"),
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    Parallel(n_jobs=-1, verbose=10)(
        delayed(process_synthetic_time_series)(
            df_time_series_hois.iloc[[i]],
            output_dir,
            psi_oi,
            psi_tc,
            order=order,
            seed=seed,
        )
        for i in range(len(df_time_series_hois))
    )
