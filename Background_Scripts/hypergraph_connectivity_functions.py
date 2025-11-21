from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from karateclub import DANMF
from numpy import linalg as lg
from permetrics import ClusteringMetric
from scipy import sparse
from scipy.fft import rfftn
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import (
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler, normalize
from tqdm import tqdm

from Background_Scripts.HCP_Data_Vis_Schaefer_100Parcels import df_schaefer

# Triple-wise permutations for 3-node hyperedges
_PERM3 = np.array(
    [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]], dtype=np.int64
)

N_rois = len(df_schaefer)
all_edges = list(combinations(range(N_rois), 2))
all_triangles = list(combinations(range(N_rois), 3))


def build_adjacency_tensors(file, hoi_labels, dtype=np.float32):
    """
    Construct super-symmetric 3rd-order adjacency tensors for
    O-information (OI) and Total Correlation (TC).

    Parameters
    ----------
    file : str
        Path to a `.npy` file containing shape (E, 2), where each row stores
        [OI_weight, TC_weight].
    hoi_labels : array-like, shape (E, M)
        Canonical hyperedge indices. M must be 3 (triangles).
        Each row corresponds to the nodes forming one hyperedge.
    dtype : numpy.dtype, optional
        Numerical precision of the output tensors. Default is float32.

    Returns
    -------
    A_OI : np.ndarray
        Super-symmetric adjacency tensor of shape (N,)*3 for O-information.
    A_TC : np.ndarray
        Super-symmetric adjacency tensor of shape (N,)*3 for Total Correlation.

    Notes
    -----
    - Super-symmetry is enforced using the predefined permutation table `_PERM3`.
    - `N_rois` must exist in the outer scope and define the number of nodes.

    Raises
    ------
    ValueError
        If input shapes are invalid or indices fall outside the allowed range.
    """
    labels = np.asarray(hoi_labels, dtype=np.int64)
    if labels.ndim != 2 or labels.shape[1] not in (3, 4):
        raise ValueError("hoi_labels deve ter shape (E, 3) ou (E, 4).")
    E, M = labels.shape
    if (labels < 0).any() or (labels >= N_rois).any():
        raise ValueError("Índices fora do intervalo [0, N).")

    tri = np.load(file)
    if tri.ndim != 2 or tri.shape[1] != 2 or tri.shape[0] != E:
        raise ValueError(
            "Arquivo deve ter shape (E, 2) e mesmo número de linhas de hoi_labels."
        )

    w_OI = tri[:, 0].astype(dtype, copy=False)
    w_TC = tri[:, 1].astype(dtype, copy=False)

    A_OI = np.zeros((N_rois,) * M, dtype=dtype)
    A_TC = np.zeros((N_rois,) * M, dtype=dtype)

    i, j, k = labels.T
    for p in _PERM3:
        ax0 = [i, j, k][p[0]]
        ax1 = [i, j, k][p[1]]
        ax2 = [i, j, k][p[2]]
        A_OI[ax0, ax1, ax2] = w_OI
        A_TC[ax0, ax1, ax2] = w_TC

    return A_OI, A_TC


def sym_order(A, i):
    """
    Apply a symmetric extension to array A along the i-th axis (1-based).
    This constructs and returns the array
        0.5 * concatenate([zeros_like_slice, A, flip(A, axis=axis)], axis=axis)
    where zeros_like_slice is an array of zeros with the same shape as A except
    that the length along the chosen axis is 1.
    Parameters
    ----------
    A : array_like
        Input array. Will be converted to a NumPy ndarray of dtype float64.
    i : int
        Axis index in 1-based (MATLAB-style) numbering. The axis to perform the
        symmetric operation on. Valid values are 1..A.ndim. Internally converted
        to 0-based index.
    Returns
    -------
    ndarray
        A new NumPy ndarray (dtype float64) equal to 0.5 * [zeros_slice, A, reversed_A]
        concatenated along the specified axis. The returned array is dense (not a view).
    Raises
    ------
    IndexError, ValueError
        Raised by NumPy if the provided axis (after converting to 0-based) is
        out of range for the input array, or if input shapes are incompatible for
        concatenation.
    Notes
    -----
    - The input is cast to float64, so original dtype and precision are not preserved.
    - The function produces a copy of the data.
    - The "flip" operation reverses elements along the chosen axis.
    """
    axis = i - 1
    Ashape = A.shape
    first0_shape = list(Ashape)
    first0_shape[axis] = 1

    A_float = np.asarray(A, dtype=np.float64)
    first_0 = np.zeros(first0_shape, dtype=A_float.dtype)
    reverse_A = np.flip(A_float, axis=axis)

    A_cat = np.concatenate([first_0, A_float, reverse_A], axis=axis)
    return 0.5 * A_cat


def symmetrize_tensor(A):
    """
    Apply symmetric concatenation (via `sym_order`) to every mode
    from mode 3 to mode p (inclusive), where p is the tensor order.

    Parameters
    ----------
    A : np.ndarray
        Input tensor of order p >= 2, typically shaped (n, n, n3, ..., np).

    Returns
    -------
    np.ndarray
        Symmetrized tensor with expanded dimensions along modes 3..p.

    Notes
    -----
    - This operation is required before applying t-FFT or t-product operations.
    - Uses MATLAB-style 1-based indexing for consistency with tensor algebra.
    """
    p = np.ndim(A)
    out = np.asarray(A)
    for i in range(3, p + 1):
        out = sym_order(out, i)
    return out


def t_fft(A, workers=-1):
    """
    Compute an optimized real FFT (rfftn) along modes 3..p of tensor `A`,
    reducing computation and memory by ~50% compared to full FFT.

    Parameters
    ----------
    A : np.ndarray
        Input tensor.
    workers : int, optional
        Number of threads for rfftn. Default is -1 (all available cores).

    Returns
    -------
    flatten_D : np.ndarray
        A MATLAB-style flattened view of the FFT coefficients with shape
        (n1, n2, num_slices).
    D : np.ndarray
        Full rFFT tensor with the natural rfftn output dimensions.

    Notes
    -----
    - If `A` has only 1 or 2 dimensions, no FFT is applied.
    - If the imaginary component is below numerical tolerance, the result is
      automatically cast to real values.
    - Designed for tensors prepared with symmetrization (t-SVD pipeline).
    """
    tol = np.finfo(float).eps
    A = np.asarray(A, dtype=np.float64)
    p = A.ndim

    if p > 2:
        axes = tuple(range(2, p))
        D = rfftn(A, axes=axes, workers=workers)
    else:
        D = A

    shape_ten = list(A.shape)
    if p > 2:
        shape_ten[-1] = shape_ten[-1] // 2 + 1
    num_slices = int(np.prod(shape_ten[2:])) if p > 2 else 1

    flatten_D = np.reshape(D, (A.shape[0], A.shape[1], num_slices), order="F")

    imag_norm = np.linalg.norm(np.imag(flatten_D).ravel())
    ref_norm = max(np.linalg.norm(flatten_D.ravel()), 1.0)
    if imag_norm <= tol * ref_norm:
        flatten_D = np.real(flatten_D)
        D = np.real(D)

    return flatten_D, D


def get_symmetrized_t_fft(file, hoi_labels):
    """
    Build adjacency tensors, symmetrize them, and compute their t-FFT
    representation (flattened format).

    Parameters
    ----------
    file : str
        Path to the .npy file containing OI/TC weights.
    hoi_labels : array-like
        Hyperedge labels used to construct adjacency tensors.

    Returns
    -------
    flatten_As_ii_t_fft : np.ndarray
        Flattened t-FFT of the symmetrized OI adjacency tensor.
    flatten_As_tc_t_fft : np.ndarray
        Flattened t-FFT of the symmetrized TC adjacency tensor.

    Notes
    -----
    - Wrapper for: build_adjacency_tensors → symmetrize_tensor → t_fft.
    - Useful for tensor-SVD and higher-order spectral analysis.
    """

    A_tensor_ii, A_tensor_tc = build_adjacency_tensors(file, hoi_labels)

    As_ii = symmetrize_tensor(A_tensor_ii)
    flatten_As_ii_t_fft, _ = t_fft(As_ii)
    As_tc = symmetrize_tensor(A_tensor_tc)
    flatten_As_tc_t_fft, _ = t_fft(As_tc)
    return flatten_As_ii_t_fft, flatten_As_tc_t_fft


def unpack_upper(vec: np.ndarray, n: int) -> np.ndarray:
    """
    Reconstruct a symmetric (n x n) matrix from a vector containing the
    upper-triangular elements (k = 1).

    Parameters
    ----------
    vec : np.ndarray
        1D vector of length n(n-1)/2 with upper-triangular values.
    n : int
        Matrix dimension.

    Returns
    -------
    A : np.ndarray
        Symmetric matrix with A[i, j] = A[j, i] corresponding to the values
        from `vec`.
    """
    A = np.zeros((n, n), dtype=vec.dtype)
    i, j = np.triu_indices(n, k=1)
    A[i, j] = vec
    A[j, i] = vec
    return A


def normalize_matrix(A, sym=True):
    """Return the normalized matrix version of the adjacency matrix 'A' (if 'laplacian' is False),
    or the normalized matrix version of the laplacian matrix of 'A' (if 'laplacian' is True).

    Parameters
    ----------
    A: numpy array matrix

    sym: boolean
        If True, the function returns the symmetrically normalized version of 'A'.
        Otherwise, it returns the randow walk normalized version of 'A'.

    """

    A = np.squeeze(A)
    vector_degree = np.sum(A, axis=0)
    D_inv = np.diag(1 / vector_degree)

    if sym:
        D_inv_sqrt = np.sqrt(D_inv)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    else:
        A_norm = D_inv @ A

    return A_norm


def compute_graph_eigenpairs(
    A,
    shift_operator="laplacian",
    norm_type="sym",
    max_ev=None,
    eig_type=None,
    use_sparse_threshold=1000,
):
    """
    Compute eigenvalues/eigenvectors for adjacency or Laplacian.

    Parameters
    ----------
    A : ndarray or sparse matrix, shape (N,N)
    shift_operator : 'adjacency' or 'laplacian'
    norm_type : 'sym', 'rw' or None
    max_ev : int or None
        Number of eigenvectors to compute. If None, compute all.
    eig_type : None, 'U', or 'L'
        Passed to eigh: use upper ('U') or lower ('L') triangular part.
        If None, default is lower.
    use_sparse_threshold : int
        If N >= threshold and max_ev << N, prefer sparse solver.

    Returns
    -------
    vals : (m,) array
    vecs : (N,m) array
    """
    A = sparse.csr_matrix(A) if sparse.issparse(A) else np.asarray(A)
    N = A.shape[0]

    # degree vector
    if sparse.issparse(A):
        deg = np.array(A.sum(axis=0)).ravel()
    else:
        deg = A.sum(axis=0)
    deg_safe = deg.copy()
    deg_safe[deg_safe == 0] = 1.0

    # build operator
    if shift_operator == "adjacency":
        F = A
    elif shift_operator == "laplacian":
        if sparse.issparse(A):
            F = sparse.diags(deg) - A
        else:
            F = np.diag(deg) - A
    else:
        raise ValueError("shift_operator must be 'adjacency' or 'laplacian'")

    # normalization
    if norm_type == "sym":
        inv_sqrt = 1.0 / np.sqrt(deg_safe)
        if sparse.issparse(F):
            D_sqrt_inv = sparse.diags(inv_sqrt)
            F = D_sqrt_inv @ F @ D_sqrt_inv
        else:
            F = np.diag(inv_sqrt) @ F @ np.diag(inv_sqrt)
    elif norm_type == "rw":
        inv = 1.0 / deg_safe
        if sparse.issparse(F):
            F = sparse.diags(inv) @ F
        else:
            F = np.diag(inv) @ F
    elif norm_type is None:
        pass
    else:
        raise ValueError("norm_type must be 'sym', 'rw', or None")

    # eigen-decomposition
    if max_ev is None or max_ev >= N:
        # full decomposition
        if sparse.issparse(F):
            F = F.toarray()
        if eig_type in ("U", "L"):
            vals, vecs = eigh(F, lower=(eig_type == "L"))
        else:
            vals, vecs = lg.eigh(F)
    else:
        # partial eigenpairs
        if shift_operator == "adjacency":
            which = "LA"  # largest algebraic
        else:
            which = "SA"  # smallest algebraic
        if sparse.issparse(F) or (N >= use_sparse_threshold and max_ev * 5 < N):
            vals, vecs = eigsh(F, k=max_ev, which=which)
        else:
            if sparse.issparse(F):
                F = F.toarray()
            if eig_type in ("U", "L"):
                vals_full, vecs_full = eigh(F, lower=(eig_type == "L"))
            else:
                vals_full, vecs_full = lg.eigh(F)
            if shift_operator == "adjacency":
                idx = np.argsort(vals_full)[-max_ev:]
            else:
                idx = np.argsort(vals_full)[:max_ev]
            vals = vals_full[idx]
            vecs = vecs_full[:, idx]

    # order eigenpairs consistently
    if shift_operator == "adjacency":
        order = np.argsort(vals)[::-1]  # largest first
    else:
        order = np.argsort(vals)  # smallest first
    vals = vals[order]
    vecs = vecs[:, order]

    return vals, vecs


def preprocess_embeddings(U, scaling=True, normalizing=True):
    """
    Preprocess embedding vectors by optional feature standardization and row-wise normalization.
    Parameters
    ----------
    U : array-like of shape (n_samples, n_features)
        Input embedding matrix where each row corresponds to a sample/embedding and each
        column to a feature/dimension.
    scaling : bool, default=True
        If True, apply sklearn.preprocessing.StandardScaler to standardize features
        (zero mean and unit variance) across samples (i.e., columns are standardized).
    normalizing : bool, default=True
        If True, apply sklearn.preprocessing.normalize with axis=1 to scale each
        sample (row) to unit norm. By default this performs L2 normalization.
    Returns
    -------
    numpy.ndarray of shape (n_samples, n_features)
        The preprocessed embedding matrix. The function returns a new array and does
        not modify the input in-place.
    Notes
    -----
    The operations are applied in the following order: scaling (if enabled) followed
    by row-wise normalization (if enabled). This combination is useful when you want
    features to be standardized before converting each embedding vector to a unit
    vector for similarity computations or downstream models.
    """

    # Standardize across rows (samples)
    if scaling:
        scaler = StandardScaler()
        U_scaled = scaler.fit_transform(U)
    else:
        U_scaled = U.copy()

    if normalizing:
        U_scaled = normalize(U_scaled, axis=1)

    return U_scaled


def graph_spectral_clustering_optimized(
    A,
    k_clusters,
    shift_operator="laplacian",
    norm_type="sym",
    scaling=True,
    normalizing=True,
    metrics=None,
    y_true=None,
    use_mini_batch=False,
    parallel=False,
    n_jobs=-1,
):
    """
    Perform spectral clustering on a graph using eigendecomposition and K-means.
    This function implements spectral clustering by computing graph eigenpairs, preprocessing
    the eigenvector embeddings, and applying K-means clustering. It supports multiple cluster
    numbers, various shift operators, and optional parallel execution.
    Parameters
    ----------
    A : array-like or sparse matrix
        Adjacency matrix or graph representation of shape (N, N) where N is the number of nodes.
    k_clusters : int or iterable of int
        Number of clusters or list of cluster numbers to try.
    shift_operator : str, optional, default="laplacian"
        Type of shift operator to use for eigendecomposition (e.g., "laplacian", "adjacency").
    norm_type : str, optional, default="sym"
        Normalization type for the shift operator (e.g., "sym" for symmetric, "rw" for random walk).
    scaling : bool, optional, default=True
        Whether to scale the eigenvector embeddings before clustering.
    normalizing : bool, optional, default=True
        Whether to normalize the eigenvector embeddings before clustering.
    metrics : list of str, optional, default=None
        List of clustering metric names to compute (e.g., "SI" for Silhouette Index,
        "DBI" for Davies-Bouldin Index).
    y_true : array-like, optional, default=None
        True cluster labels for computing supervised metrics. Shape (N,).
    use_mini_batch : bool, optional, default=False
        Whether to use MiniBatchKMeans instead of standard KMeans for faster computation.
    parallel : bool, optional, default=False
        Whether to run clustering for different k values in parallel.
    n_jobs : int, optional, default=-1
        Number of parallel jobs to run. -1 means using all processors.
    Returns
    -------
    dict
        Dictionary containing:
        - "models" : dict
            Dictionary mapping k to cluster labels array for each k in k_clusters_list.
        - "embeddings" : dict
            Dictionary mapping k to preprocessed embedding matrix of shape (N, k) for each k.
        - "scores" : dict
            Dictionary mapping k to dict of metric scores for each k (if metrics is not None).
    Notes
    -----
    - The function computes eigenpairs up to the maximum k value in k_clusters.
    - Embeddings are preprocessed (scaled/normalized) before K-means clustering.
    - Special handling for "SI" (Silhouette Index) and "DBI" (Davies-Bouldin Index) metrics.
    - All K-means models use random_state=42 and n_init=50 for reproducibility.
    Examples
    --------
    >>> result = graph_spectral_clustering_optimized(A, k_clusters=3, metrics=["SI", "DBI"])
    >>> labels = result["models"][3]
    >>> scores = result["scores"][3]
    """

    if isinstance(k_clusters, int):
        k_clusters_list = [k_clusters]
    else:
        k_clusters_list = list(k_clusters)
    # compute eigenpairs up to max_k_feat
    # solver_max_ev overrides if provided
    max_ev = max(k_clusters_list)
    evals, evecs = compute_graph_eigenpairs(
        A, shift_operator=shift_operator, norm_type=norm_type, max_ev=max_ev
    )
    # evecs shape: (N, m)

    # KMeans runner for single k
    def run_k(k):
        # Preprocess embeddings once for the maximal needed features and reuse slices
        X = preprocess_embeddings(
            evecs[:, :k], scaling=scaling, normalizing=normalizing
        )
        if use_mini_batch:
            model = (
                MiniBatchKMeans(n_clusters=k, random_state=42, n_init=50).fit(X).labels_
            )
        else:
            model = KMeans(n_clusters=k, random_state=42, n_init=50).fit(X).labels_

        return k, model, X

    # parallelize runs
    if parallel:
        jobs = Parallel(n_jobs=n_jobs)(delayed(run_k)(k) for k in k_clusters_list)
    else:
        jobs = [run_k(k) for k in k_clusters_list]

    models = {}
    embeddings = {}
    scores = {}
    for k, model, X in jobs:
        models[k] = model
        embeddings[k] = X
        if metrics is not None:
            cm = ClusteringMetric(X=X, y_pred=model, y_true=y_true)
            scores[k] = cm.get_metrics_by_list_names(metrics)
            if "SI" in metrics:
                scores[k]["SI"] = silhouette_score(X, model)
            if "DBI" in metrics:
                scores[k]["DBI"] = davies_bouldin_score(X, model)

    return {"models": models, "embeddings": embeddings, "scores": scores}


def danmf_clustering_optimized(
    A,
    k_clusters,
    layers,
    pre_iterations=100,
    iterations=100,
    lamb=0.01,
    scaling=True,
    normalizing=True,
    metrics=None,
    y_true=None,
    use_mini_batch=False,
    parallel=False,
    n_jobs=-1,
):
    """
    Perform DANMF (Deep Autoencoder-like Nonnegative Matrix Factorization) clustering with optional optimization.
    This function applies DANMF to generate graph embeddings and then performs KMeans clustering
    on the resulting embeddings. It supports multiple cluster numbers, parallel processing, and
    various clustering metrics evaluation.
    Parameters
    ----------
    A : numpy.ndarray or scipy.sparse matrix
        Adjacency matrix representing the graph structure.
    k_clusters : int or array-like
        Number of clusters or list of cluster numbers to evaluate. If int, a single clustering
        is performed. If array-like, multiple clusterings are performed for each value.
    layers : list of int
        List of layer sizes for the DANMF model architecture (excluding the output layer).
    pre_iterations : int, optional (default=100)
        Number of pre-training iterations for DANMF.
    iterations : int, optional (default=100)
        Number of training iterations for DANMF.
    lamb : float, optional (default=0.01)
        Regularization parameter for DANMF.
    scaling : bool, optional (default=True)
        Whether to scale the embeddings before clustering.
    normalizing : bool, optional (default=True)
        Whether to normalize the embeddings before clustering.
    metrics : list of str, optional (default=None)
        List of metric names to compute for clustering evaluation.
        Supported metrics include 'SI' (Silhouette Index) and 'DBI' (Davies-Bouldin Index).
    y_true : array-like, optional (default=None)
        True labels for computing supervised clustering metrics.
    use_mini_batch : bool, optional (default=False)
        Whether to use MiniBatchKMeans instead of standard KMeans.
    parallel : bool, optional (default=False)
        Whether to run multiple k values in parallel.
    n_jobs : int, optional (default=-1)
        Number of parallel jobs. -1 means using all processors.
    Returns
    -------
    dict
        Dictionary containing:
        - 'models': dict mapping k values to predicted cluster labels
        - 'embeddings': dict mapping k values to preprocessed embeddings
        - 'scores': dict mapping k values to computed clustering metrics (if metrics is not None)
    Notes
    -----
    - The adjacency matrix A is first normalized using normalize_matrix().
    - A NetworkX graph is created from the normalized adjacency matrix.
    - For each k value, DANMF generates embeddings which are then clustered using KMeans.
    - If parallel=True, multiple k values are processed simultaneously using joblib.Parallel.
    """

    if isinstance(k_clusters, int):
        k_clusters_list = [k_clusters]
    else:
        k_clusters_list = list(k_clusters)
    A_norm = normalize_matrix(A)
    G = nx.from_numpy_matrix(A_norm)

    # KMeans runner for single k
    def run_k(k):
        layers_model = layers + [k]
        danmf_model = DANMF(
            layers=layers_model,
            pre_iterations=pre_iterations,
            iterations=iterations,
            lamb=lamb,
        )
        danmf_model.fit(G)
        embeddings_danmf = danmf_model.get_embedding()
        labels_danmf = list(danmf_model.get_memberships().values())
        X = preprocess_embeddings(
            embeddings_danmf, scaling=scaling, normalizing=normalizing
        )
        if use_mini_batch:
            labels_danmf = (
                MiniBatchKMeans(n_clusters=k, random_state=42, n_init=50).fit(X).labels_
            )
        else:
            labels_danmf = (
                KMeans(n_clusters=k, random_state=42, n_init=50).fit(X).labels_
            )

        return k, labels_danmf, X

    # parallelize runs
    if parallel:
        jobs = Parallel(n_jobs=n_jobs)(delayed(run_k)(k) for k in k_clusters_list)
    else:
        jobs = [run_k(k) for k in k_clusters_list]

    models = {}
    embeddings = {}
    scores = {}
    for k, model, X in jobs:
        models[k] = model
        embeddings[k] = X
        if metrics is not None:
            cm = ClusteringMetric(X=X, y_pred=model, y_true=y_true)
            scores[k] = cm.get_metrics_by_list_names(metrics)
            if "SI" in metrics:
                scores[k]["SI"] = silhouette_score(X, model)
            if "DBI" in metrics:
                scores[k]["DBI"] = davies_bouldin_score(X, model)

    return {"models": models, "embeddings": embeddings, "scores": scores}


def consistent_communities(
    df_rest,
    k_clusters,
    danmf=False,
    spectral_shift_operator="laplacian",
    spectral_norm_type="sym",
    danmf_layers=[58],
    danmf_pre_iterations=100,
    danmf_iterations=4000,
    danmf_lamb=0.1,
):
    """
    For each matrix in matrices_rest1 and matrices_rest2, perform spectral clustering
    with the specified k_clusters. Then, for each subject that appears in both lists,
    compute the normalized mutual information (NMI) between their cluster labels from
    rest1 and rest2. Return the average NMI across all such subjects.

    Parameters
    ----------
    matrices_rest1 : list of tuples (subject_id, matrix)
        List of adjacency matrices for rest1, each associated with a subject ID.
    matrices_rest2 : list of tuples (subject_id, matrix)
        List of adjacency matrices for rest2, each associated with a subject ID.
    k_clusters : int
        Number of clusters to use in spectral clustering.
    shift_operator : str
        Shift operator to use in spectral clustering ('adjacency' or 'laplacian').
    norm_type : str or None
        Normalization type to use in spectral clustering ('sym', 'rw', or None).

    Returns
    -------
    float
        Average NMI between cluster labels from rest1 and rest2 for subjects present in both.
    """
    data = []
    subjects_id = df_rest["Subject"].unique()
    for subject in tqdm(subjects_id):
        matrix_rest1 = unpack_upper(
            np.load(
                df_rest[(df_rest["Subject"] == subject) & (df_rest["REST"] == 1)][
                    "File"
                ].values[0]
            ),
            N_rois,
        )
        matrix_rest2 = unpack_upper(
            np.load(
                df_rest[(df_rest["Subject"] == subject) & (df_rest["REST"] == 2)][
                    "File"
                ].values[0]
            ),
            N_rois,
        )

        # If matrices are not symmetric and non-negative raise error
        if not (
            np.allclose(matrix_rest1, matrix_rest1.T) and np.all(matrix_rest1 >= 0)
        ):
            raise ValueError(
                f"Matrix for subject {subject} (REST 1) is not symmetric and non-negative."
            )
        if not (
            np.allclose(matrix_rest2, matrix_rest2.T) and np.all(matrix_rest2 >= 0)
        ):
            raise ValueError(
                f"Matrix for subject {subject} (REST 2) is not symmetric and non-negative."
            )

        if not danmf:
            results_rest1 = graph_spectral_clustering_optimized(
                A=matrix_rest1,
                k_clusters=k_clusters,
                metrics=None,
                shift_operator=spectral_shift_operator,
                norm_type=spectral_norm_type,
                parallel=True,
            )
            results_rest2 = graph_spectral_clustering_optimized(
                A=matrix_rest2,
                k_clusters=k_clusters,
                metrics=None,
                shift_operator=spectral_shift_operator,
                norm_type=spectral_norm_type,
                parallel=True,
            )
        else:
            results_rest1 = danmf_clustering_optimized(
                A=matrix_rest1,
                k_clusters=k_clusters,
                metrics=None,
                layers=danmf_layers,
                pre_iterations=danmf_pre_iterations,
                iterations=danmf_iterations,
                lamb=danmf_lamb,
                parallel=True,
            )
            results_rest2 = danmf_clustering_optimized(
                A=matrix_rest2,
                k_clusters=k_clusters,
                metrics=None,
                layers=danmf_layers,
                pre_iterations=danmf_pre_iterations,
                iterations=danmf_iterations,
                lamb=danmf_lamb,
                parallel=True,
            )

        for k in k_clusters:
            labels_rest1 = results_rest1["models"][k]
            labels_rest2 = results_rest2["models"][k]
            nmi = normalized_mutual_info_score(labels_rest1, labels_rest2)
            data.append({"Subject": subject, "k": k, "NMI": nmi})
    df_nmis_subject = pd.DataFrame(data)

    return df_nmis_subject


def simmilarity_emp_syn_sur(
    df_emp,
    df_syn_hoi,
    df_syn_pair,
    df_sur,
    k_clusters,
    danmf=False,
    spectral_shift_operator="laplacian",
    spectral_norm_type="sym",
    danmf_layers=[58],
    danmf_pre_iterations=100,
    danmf_iterations=4000,
    danmf_lamb=0.1,
):
    """
    Compute normalized mutual information (NMI) between community assignments obtained from empirical, synthetic (higher-order interactions and pairwise) and surrogate connectivity matrices, for each subject and REST session, across a set of cluster counts.
    The function:
    - Joins four input dataframes on the columns "Subject" and "REST". Each dataframe must contain a "File" column (renamed internally to identify source).
    - Loads connectivity data from the file paths stored in the "File" columns using numpy.load and converts them to full adjacency matrices via unpack_upper(..., N_rois). (Note: N_rois must be defined in the calling environment.)
    - Validates that each loaded matrix is symmetric and non-negative; raises ValueError otherwise.
    - Computes cluster labels for each matrix using either spectral clustering (graph_spectral_clustering_optimized) or DANMF-based clustering (danmf_clustering_optimized) depending on the `danmf` flag. Clustering is performed for each k in k_clusters.
    - Computes NMI between empirical labels and (i) synthetic HOI labels, (ii) synthetic pairwise labels, and (iii) surrogate labels for every k.
    - Returns a per-subject / per-REST DataFrame with the computed NMIs stored as dictionaries mapping k -> NMI value.
    Parameters
    ----------
    df_emp : pandas.DataFrame
        DataFrame containing empirical file paths with at least columns ["Subject", "REST", "File"].
    df_syn_hoi : pandas.DataFrame
        DataFrame containing synthetic HOI file paths with at least columns ["Subject", "REST", "File"].
    df_syn_pair : pandas.DataFrame
        DataFrame containing synthetic pairwise file paths with at least columns ["Subject", "REST", "File"].
    df_sur : pandas.DataFrame
        DataFrame containing surrogate file paths with at least columns ["Subject", "REST", "File"].
    k_clusters : iterable of int
        Iterable (e.g., list or array) of integers specifying the numbers of clusters (K) to evaluate.
    danmf : bool, optional
        If False (default), use graph spectral clustering. If True, use DANMF-based clustering.
    spectral_shift_operator : str, optional
        Shift operator name passed to spectral clustering (default "laplacian").
    spectral_norm_type : str, optional
        Normalization type used for spectral clustering (default "sym").
    danmf_layers : list of int, optional
        Layer sizes for DANMF when danmf=True (default [58]).
    danmf_pre_iterations : int, optional
        Number of pre-training iterations for DANMF (default 100).
    danmf_iterations : int, optional
        Number of DANMF training iterations (default 4000).
    danmf_lamb : float, optional
        Regularization parameter lambda for DANMF (default 0.1).
    Returns
    -------
    pandas.DataFrame
        A DataFrame with columns ["Subject", "REST", "nmis_emp_syn_hoi", "nmis_emp_syn_pair", "nmis_emp_sur"].
        For each row:
        - "nmis_emp_syn_hoi" is a dict {k: nmi_value} comparing empirical vs synthetic HOI clustering.
        - "nmis_emp_syn_pair" is a dict {k: nmi_value} comparing empirical vs synthetic pairwise clustering.
        - "nmis_emp_sur" is a dict {k: nmi_value} comparing empirical vs surrogate clustering.
    Raises
    ------
    ValueError
        If any loaded matrix is not symmetric or contains negative entries.
    Any exceptions raised by numpy.load, unpack_upper, graph_spectral_clustering_optimized, danmf_clustering_optimized,
    or normalized_mutual_info_score will propagate to the caller.
    Notes
    -----
    - The function expects that the file paths in the "File" columns can be loaded with numpy.load and are compatible with unpack_upper(..., N_rois).
    - N_rois is referenced when calling unpack_upper; it must be defined in the environment where this function is executed.
    - Clustering functions are invoked with parallel=True; ensure the environment supports parallel execution if needed.
    - tqdm is used to show progress for iterations over subjects.
    Example
    -------
    Assuming dataframes df_emp, df_syn_hoi, df_syn_pair, df_sur exist and N_rois is defined:
    k_values = [2, 3, 4, 5]
    df_nmis = simmilarity_emp_syn_sur(df_emp, df_syn_hoi, df_syn_pair, df_sur, k_values)
    """

    df_empirical = df_emp[["Subject", "REST", "File"]].rename(
        columns={"File": "File_emp"}
    )
    df_synthetic_hoi = df_syn_hoi[["Subject", "REST", "File"]].rename(
        columns={"File": "File_syn_hoi"}
    )
    df_synthetic_pair = df_syn_pair[["Subject", "REST", "File"]].rename(
        columns={"File": "File_syn_pair"}
    )
    df_surrogate = df_sur[["Subject", "REST", "File"]].rename(
        columns={"File": "File_sur"}
    )

    df = (
        df_empirical.merge(df_synthetic_hoi, on=["Subject", "REST"], how="inner")
        .merge(df_synthetic_pair, on=["Subject", "REST"], how="inner")
        .merge(df_surrogate, on=["Subject", "REST"], how="inner")
    )

    df["nmis_emp_syn_hoi"] = [None] * len(df)
    df["nmis_emp_syn_pair"] = [None] * len(df)
    df["nmis_emp_sur"] = [None] * len(df)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        matrix_emp = unpack_upper(np.load(row["File_emp"]), N_rois)
        matrix_syn_hoi = unpack_upper(np.load(row["File_syn_hoi"]), N_rois)
        matrix_syn_pair = unpack_upper(np.load(row["File_syn_pair"]), N_rois)
        matrix_sur = unpack_upper(np.load(row["File_sur"]), N_rois)

        if not (np.allclose(matrix_emp, matrix_emp.T) and np.all(matrix_emp >= 0)):
            raise ValueError(
                f"Empirical Matrix for subject {row['Subject']} is not symmetric and non-negative."
            )
        if not (np.allclose(matrix_sur, matrix_sur.T) and np.all(matrix_sur >= 0)):
            raise ValueError(
                f"Surrogate Matrix for subject {row['Subject']} is not symmetric and non-negative."
            )
        if not (
            np.allclose(matrix_syn_hoi, matrix_syn_hoi.T)
            and np.all(matrix_syn_hoi >= 0)
        ):
            raise ValueError(
                f"Synthetic HOI Matrix for subject {row['Subject']} is not symmetric and non-negative."
            )

        if not (
            np.allclose(matrix_syn_pair, matrix_syn_pair.T)
            and np.all(matrix_syn_pair >= 0)
        ):
            raise ValueError(
                f"Synthetic Pairwise Matrix for subject {row['Subject']} is not symmetric and non-negative."
            )

        if not danmf:
            results_emp = graph_spectral_clustering_optimized(
                A=matrix_emp,
                k_clusters=k_clusters,
                metrics=None,
                shift_operator=spectral_shift_operator,
                norm_type=spectral_norm_type,
                parallel=True,
            )
            results_syn_hoi = graph_spectral_clustering_optimized(
                A=matrix_syn_hoi,
                k_clusters=k_clusters,
                metrics=None,
                shift_operator=spectral_shift_operator,
                norm_type=spectral_norm_type,
                parallel=True,
            )
            results_syn_pair = graph_spectral_clustering_optimized(
                A=matrix_syn_pair,
                k_clusters=k_clusters,
                metrics=None,
                shift_operator=spectral_shift_operator,
                norm_type=spectral_norm_type,
                parallel=True,
            )
            results_sur = graph_spectral_clustering_optimized(
                A=matrix_sur,
                k_clusters=k_clusters,
                metrics=None,
                shift_operator=spectral_shift_operator,
                norm_type=spectral_norm_type,
                parallel=True,
            )
        else:
            results_emp = danmf_clustering_optimized(
                A=matrix_emp,
                k_clusters=k_clusters,
                metrics=None,
                layers=danmf_layers,
                pre_iterations=danmf_pre_iterations,
                iterations=danmf_iterations,
                lamb=danmf_lamb,
                parallel=True,
            )
            results_syn_hoi = danmf_clustering_optimized(
                A=matrix_syn_hoi,
                k_clusters=k_clusters,
                metrics=None,
                layers=danmf_layers,
                pre_iterations=danmf_pre_iterations,
                iterations=danmf_iterations,
                lamb=danmf_lamb,
                parallel=True,
            )
            results_syn_pair = danmf_clustering_optimized(
                A=matrix_syn_pair,
                k_clusters=k_clusters,
                metrics=None,
                layers=danmf_layers,
                pre_iterations=danmf_pre_iterations,
                iterations=danmf_iterations,
                lamb=danmf_lamb,
                parallel=True,
            )
            results_sur = danmf_clustering_optimized(
                A=matrix_sur,
                k_clusters=k_clusters,
                metrics=None,
                layers=danmf_layers,
                pre_iterations=danmf_pre_iterations,
                iterations=danmf_iterations,
                lamb=danmf_lamb,
                parallel=True,
            )

        nmi_emp_syn_hoi = {}
        nmi_emp_syn_pair = {}
        nmi_emp_sur = {}
        for k in k_clusters:
            labels_emp = results_emp["models"][k]
            labels_syn_hoi = results_syn_hoi["models"][k]
            labels_syn_pair = results_syn_pair["models"][k]
            labels_sur = results_sur["models"][k]

            nmi_emp_syn_hoi[k] = normalized_mutual_info_score(
                labels_emp, labels_syn_hoi
            )
            nmi_emp_syn_pair[k] = normalized_mutual_info_score(
                labels_emp, labels_syn_pair
            )
            nmi_emp_sur[k] = normalized_mutual_info_score(labels_emp, labels_sur)

        df.at[idx, "nmis_emp_syn_hoi"] = nmi_emp_syn_hoi
        df.at[idx, "nmis_emp_syn_pair"] = nmi_emp_syn_pair
        df.at[idx, "nmis_emp_sur"] = nmi_emp_sur

    df_nmis_subject = df[
        ["Subject", "REST", "nmis_emp_syn_hoi", "nmis_emp_syn_pair", "nmis_emp_sur"]
    ]

    return df_nmis_subject
