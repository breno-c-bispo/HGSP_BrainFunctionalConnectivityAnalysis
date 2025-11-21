import sys

sys.path.insert(0, "./Background_Scripts")

from math import ceil

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from computation_connectivity_weights_functions import pack_upper
from hypergraph_connectivity_functions import (
    all_edges,
    all_triangles,
    get_symmetrized_t_fft,
)
from scipy.stats import mannwhitneyu, pearsonr, ttest_ind, ttest_rel, zscore
from sklearn.metrics import silhouette_samples

mpl.rcParams["text.usetex"] = False
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "Times New Roman"


def plot_weights_histogram(mi_avr_file, hoi_avr_file, file_name=None):
    """
    Plot three histograms showing weight distributions for edges and hyperedges.
    Parameters
    ----------
    mi_avr_file : str or os.PathLike
        Path to a .npy file containing a 1D numpy array of edge weights for the graph
        (G_MI). The array length must match the number of elements in the global iterable
        `all_edges`. Each array element is associated with the corresponding entry in
        `all_edges` by index.
    hoi_avr_file : str or os.PathLike
        Path to a .npy file containing a 2D numpy array of hyperedge weights for the
        hypergraph (H_OI/H_TC). Expected shape is (len(all_triangles), 2), where
        column 0 corresponds to the H_OI weight (called hoi_ii in the function) and column 1
        corresponds to the H_TC weight (called hoi_tc). The row order must match the global
        iterable `all_triangles`.
    file_name : str or os.PathLike, optional
        If provided, the plotted figure is saved to this path (PNG) with dpi=300 and
        bbox_inches='tight'. If None (default), the figure is not saved but is shown
        interactively via plt.show().
    Returns
    -------
    None
        This function does not return a value. It displays the figure (and optionally saves it)
        and populates three subplots: edge weights (blue), H_OI hyperedge weights (red), and
        H_TC hyperedge weights (green). Each histogram includes a KDE overlay.
    Side effects and requirements
    -----------------------------
    - Uses numpy.load to read the .npy files.
    - Relies on the global variables `all_edges` and `all_triangles` being defined and ordered
      consistently with the saved arrays.
    - Requires matplotlib.pyplot (as plt) and seaborn (as sns) to be imported in the caller's
      namespace.
    - Creates a 1x3 subplot figure sized (16, 3) and calls plt.show() at the end.
    - If the input arrays' lengths/shapes do not match the expected sizes derived from the
      globals, indexing errors or unexpected behavior may occur.
    - No input validation is performed on array contents (NaNs, infinities, or negative values
      will be plotted as-is).
    Examples
    --------
    >>> plot_weights_histogram("edge_weights.npy", "hyperedge_weights.npy")
    >>> plot_weights_histogram("edge_weights.npy", "hyperedge_weights.npy", file_name="weights.png")
    """
    edge_avr = np.load(mi_avr_file)
    edges = {}
    for i, e in enumerate(all_edges):
        edges[e] = edge_avr[i]
    hoi_avr = np.load(hoi_avr_file)
    hoi_oi = {}
    hoi_tc = {}
    for i, triangle in enumerate(all_triangles):
        hoi_oi[triangle] = hoi_avr[i, 0]
        hoi_tc[triangle] = hoi_avr[i, 1]

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(16, 3))

    sns.histplot(
        list(edges.values()),
        bins=50,
        color="blue",
        ax=axes[0],
        kde=True,
    )
    axes[0].set_title(r"Edge weight distribution of $\mathcal{G}_{MI}$", fontsize=20)
    axes[0].set_xlabel("Edge weight", fontsize=16)
    axes[0].set_ylabel("Number of edges", fontsize=16)
    axes[0].tick_params(axis="both", which="major", labelsize=14)

    sns.histplot(
        list(hoi_oi.values()),
        bins=50,
        color="red",
        ax=axes[1],
        kde=True,
    )
    axes[1].set_title(
        r"Hyperedge weight distribution of $\mathcal{H}_{OI}$", fontsize=20
    )
    axes[1].set_xlabel("Hyperedge weight", fontsize=16)
    axes[1].set_ylabel("Number of hyperedges", fontsize=16)
    axes[1].tick_params(axis="both", which="major", labelsize=14)

    sns.histplot(
        list(hoi_tc.values()),
        bins=50,
        color="green",
        ax=axes[2],
        kde=True,
    )
    axes[2].set_title(
        r"Hyperedge weight distribution of $\mathcal{H}_{TC}$", fontsize=20
    )
    axes[2].set_xlabel("Hyperedge weight", fontsize=16)
    axes[2].set_ylabel("Number of hyperedges", fontsize=16)
    axes[2].tick_params(axis="both", which="major", labelsize=14)
    plt.tight_layout()
    if file_name is not None:
        plt.savefig(
            file_name,
            dpi=300,
            bbox_inches="tight",
        )
    plt.show()


def plot_clustering_performance_metrics(
    df_plot: pd.DataFrame,
    file_name: str,
    title: str = "Clustering performance metrics vs number of clusters (k)",
    show: bool = True,
    normalize: bool = False,
    objectives: dict = None,
    plot_global: bool = False,
):
    """
    Plot clustering performance metrics vs number of clusters and compute
    a composite ("Global") score. The user may choose whether the Global
    curve is plotted.

    Parameters
    ----------
    df_plot : pandas.DataFrame
        DataFrame with clustering metrics. Index must represent k.
    file_name : str or None
        Path to save the figure. If None, the plot is not saved.
    title : str
        Title of the figure.
    show : bool
        If True, displays the plot.
    normalize : bool
        If True, perform min–max normalization per metric.
    objectives : dict
        Mapping metric -> {"min" or "max"} defining optimization direction.
    plot_global : bool
        If True, the Global metric curve is plotted.
        If False, Global is computed but not displayed.

    Returns
    -------
    pandas.DataFrame
        Long DataFrame with columns:
            - k : number of clusters
            - metric : metric name (including Global)
            - score : original or aggregated value
            - score_norm : normalized score (if normalize=True)
    """

    # ---- Validations ----
    if not isinstance(df_plot, pd.DataFrame) or df_plot.empty:
        raise ValueError(
            "df_plot must be a non-empty DataFrame with metrics as columns."
        )

    df = df_plot.copy()
    metrics_to_plot = list(objectives.keys()) if objectives else None

    # Keep only metrics specified
    if metrics_to_plot is not None:
        keep = [m for m in metrics_to_plot if m in df.columns]
        if not keep:
            raise ValueError(
                "None of the requested metrics_to_plot are present in df_plot."
            )
        df = df[keep]

    # Reshape to long format
    df = df.sort_index()
    df = df.reset_index().rename(columns={"index": "k"})
    df["k"] = df["k"].astype(int)
    df_long = df.melt(id_vars="k", var_name="metric", value_name="score")

    # ---- Normalization ----
    y_col = "score"
    if normalize:

        def _minmax(s):
            smin, smax = s.min(), s.max()
            return (s - smin) / (smax - smin) if smax != smin else np.zeros_like(s)

        df_long["score_norm"] = df_long.groupby("metric")["score"].transform(_minmax)
        y_col = "score_norm"

    # ---- Compute Global score ----
    overall = {}
    for k in df_long["k"].unique():
        score_overall = 0
        for key, direction in objectives.items():
            score = df_long[(df_long["k"] == k) & (df_long["metric"] == key)][
                y_col
            ].values[0]

            if direction == "max":
                score_overall += score
            elif direction == "min":
                score_overall += df_long[df_long["metric"] == key][y_col].max() - score
            else:
                raise ValueError("Objectives must be 'max' or 'min'.")

        overall[k] = score_overall

    if normalize:
        vmin, vmax = min(overall.values()), max(overall.values())
        overall = {
            k: (v - vmin) / (vmax - vmin) if vmax != vmin else 0
            for k, v in overall.items()
        }

    temp = pd.DataFrame(
        {
            "k": list(overall.keys()),
            "metric": ["Global"] * len(overall),
            "score": list(overall.values()),
            "score_norm": list(overall.values()),
        }
    )

    df_long = pd.concat([df_long, temp], ignore_index=True)

    # ---- Plot ----
    fig, ax = plt.subplots(figsize=(8, 5))

    for metric, sub in df_long.groupby("metric"):
        if (metric == "Global") and (not plot_global):
            continue  # skip plotting Global curve

        sub = sub.sort_values("k")
        alpha = 1.0 if metric == "Global" or not plot_global else 0.3

        ax.plot(
            sub["k"],
            sub[y_col],
            marker="o",
            label=metric,
            linewidth=2,
            alpha=alpha,
        )

    ax.set_title(f"{title}{' (normalized)' if normalize else ''}")
    ax.set_xlabel("k (number of clusters)")
    ax.set_ylabel("Score" + (" (normalized [0, 1])" if normalize else ""))
    ax.grid(True, linestyle="--", alpha=0.4)

    # Legend only for visible curves
    ax.legend(title="Metric", loc="best")

    fig.tight_layout()

    # Save
    if file_name is not None:
        fig.savefig(file_name, dpi=300, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return df_long


def plot_nmis_emp_syn_sur(
    df_nmis,
    k_clusters,
    k_selected=None,
    fig_name=None,
    violin_fig_name=None,
    fig_size=(8, 5),
    violin_fig_size=(5, 5),
    fig_title=None,
    label_fontsize=18,
    tick_fontsize=18,
    title_fontsize=18,
    legend_fontsize=14,
    text_fontsize=19,
    markersize=20,
):
    """
    Plot NMIS comparisons between empirical, synthetic (hybrid and pairwise) and surrogate communities.
    This function summarizes and visualizes Normalized Mutual Information Scores (NMIS)
    stored in a DataFrame and produces:
    - a line plot of mean NMIS ± 1 standard deviation across samples for each K
        for three conditions: Empirical/(hybrid) Synthetic, Empirical/(pairwise) Synthetic, Empirical/Surrogate;
    - optional red markers on the line plot where the p-value for the difference between
        hybrid and pairwise synthetic results is < 0.05 for a given K;
    - an optional violin plot (split) for a selected K showing the distribution of NMIS
        for Empirical⇔(hybrid) Synthetic vs Empirical⇔(pairwise) Synthetic, with an associated
        paired t-test p-value shown in the title.
    Parameters
    ----------
    df_nmis : pandas.DataFrame
            DataFrame where each row corresponds to one sample/replicate and must contain
            the following columns (values in each cell are typically dict-like mapping K -> NMIS):
                - 'nmis_emp_syn_hoi'   : NMIS between empirical and (hybrid) synthetic (dict or NaN)
                - 'nmis_emp_syn_pair'  : NMIS between empirical and (pairwise) synthetic (dict or NaN)
                - 'nmis_emp_sur'       : NMIS between empirical and surrogate (dict or NaN)
            Any non-dict entries are treated as missing (NaN). The function copies df_nmis internally.
    k_clusters : sequence
            Sequence (list/array) of K values (cluster numbers) to consider and use as x-axis ticks.
            These keys are used to extract values from the per-row dicts.
    k_selected : scalar, optional
            If provided, must be one of the values in k_clusters. When set, the function:
                - creates a split violin plot comparing Empirical⇔(hybrid) vs Empirical⇔(pairwise)
                    distributions at that K.
            Default is None (no violin plot / per-K annotation).
    fig_name : str or path-like, optional
            If provided, the main plot is saved to this filename (PNG) using dpi=300 and bbox_inches='tight'.
            Default is None (no save).
    violin_fig_name : str or path-like, optional
            If provided and k_selected is not None, the violin plot is saved to this filename (PNG).
            Default is None (no save).
    fig_size : tuple, optional
            Matplotlib figure size for the main plot (width, height). Default (8, 5).
    violin_fig_size : tuple, optional
            Matplotlib figure size for the violin plot. Default (5, 5).
    fig_title : str or None
            Title fragment used in plot titles. If None, a short generic title will still be used.
    label_fontsize, tick_fontsize, title_fontsize, legend_fontsize, text_fontsize : int, optional
            Font sizes for labels, ticks, title, legend and annotation text respectively.
    markersize : int, optional
            Marker size used for the significance circle markers highlighting K values where
            hybrid vs pairwise p < 0.05. Default 20.
    Behavior and calculations
    -------------------------
    - Extracts vectors of NMIS per row for each K from the three DataFrame columns, producing
        arrays of shape (n_samples, n_K). Missing entries are represented as np.nan.
    - For each K:
            - mean and sample standard deviation (ddof=1) are computed across rows ignoring NaNs.
            - a p-value comparing hybrid vs pairwise is computed:
                    * if a paired sample exists for a row (both values non-NaN) for that K and there
                        are >1 such paired observations, a paired t-test (scipy.stats.ttest_rel) is used.
                    * otherwise, if there are >1 unpaired observations for each group, a Welch
                        t-test (scipy.stats.ttest_ind with equal_var=False) is used.
                    * if insufficient data, the p-value is set to NaN.
    - The main plot:
            - draws lines+markers for the means of the three conditions across K,
                and plots bars corresponding to ±1 standard deviation (as filled bars).
            - marks K positions with red hollow circles where p < 0.05 (hybrid vs pairwise).
            - if k_selected is provided, annotates the axis with means for that K and the
                paired p-value (if computable), and draws a vertical dashed line at the selected K.
    - If k_selected is provided, a split seaborn violin plot is produced comparing the
        distributions of NMIS for Empirical⇔(hybrid) vs Empirical⇔(pairwise) at that K.
        The violin plot uses only samples where both hybrid and pairwise values are available
        (paired comparisons). A paired t-test p-value is computed and shown (or "NA" if not available).
    Dependencies
    ------------
    This function expects the following imports in the calling scope:
            import matplotlib.pyplot as plt
    It also uses matplotlib/seaborn styling for plotting.
    Return
    ------
    None
            The function shows the main plot (and the violin plot when requested) and may save
            them to disk if fig_name and/or violin_fig_name are provided.
    Raises
    ------
    ValueError
            - If df_nmis is empty.
            - If k_selected is not None and is not contained in k_clusters.
    Notes
    -----
    - The function treats non-dict entries in expected dict columns as missing.
    - Means and SDs are computed along axis=0 ignoring NaNs.
    - p-values are computed per-K using either paired or unpaired t-tests depending on data availability.
    - Plotting is performed with plt.show(), so calling code in non-interactive environments may
        need to manage figure display or saving accordingly.
    Examples
    --------
    Basic usage:
            plot_nmis_emp_syn_sur(df_nmis, k_clusters=[2,3,4,5], fig_title="My dataset")
    With violin at a selected K and saving figures:
            plot_nmis_emp_syn_sur(
                    k_clusters=[2,3,4,5],
                    k_selected=3,
                    fig_name="nmis_summary.png",
                    violin_fig_name="nmis_violin_k3.png",
                    fig_title="Dataset X"
    """

    df = df_nmis.copy()
    if df.empty:
        raise ValueError("DataFrame 'df' está vazio.")

    if k_selected not in k_clusters and k_selected is not None:
        raise ValueError("k_selected must be one of k_clusters")

    arr_hoi, arr_pair, arr_sur = [], [], []
    if k_selected is not None:
        nmis_emp_syn_hoi_selected, nmis_emp_syn_pair_selected, nmis_emp_sur_selected = (
            [],
            [],
            [],
        )

    for _, row in df.iterrows():
        d_hoi = row["nmis_emp_syn_hoi"]
        d_pair = row["nmis_emp_syn_pair"]
        d_sur = row["nmis_emp_sur"]

        vals_hoi = [
            d_hoi.get(k, np.nan) if isinstance(d_hoi, dict) else np.nan
            for k in k_clusters
        ]
        vals_pair = [
            d_pair.get(k, np.nan) if isinstance(d_pair, dict) else np.nan
            for k in k_clusters
        ]
        vals_sur = [
            d_sur.get(k, np.nan) if isinstance(d_sur, dict) else np.nan
            for k in k_clusters
        ]

        arr_hoi.append(vals_hoi)
        arr_pair.append(vals_pair)
        arr_sur.append(vals_sur)

        if k_selected is not None:
            nmis_emp_syn_hoi_selected.append(
                d_hoi.get(k_selected, np.nan) if isinstance(d_hoi, dict) else np.nan
            )
            nmis_emp_syn_pair_selected.append(
                d_pair.get(k_selected, np.nan) if isinstance(d_pair, dict) else np.nan
            )
            nmis_emp_sur_selected.append(
                d_sur.get(k_selected, np.nan) if isinstance(d_sur, dict) else np.nan
            )

    arr_hoi = np.array(arr_hoi, dtype=float)
    arr_pair = np.array(arr_pair, dtype=float)
    arr_sur = np.array(arr_sur, dtype=float)

    mean_hoi = np.nanmean(arr_hoi, axis=0)
    std_hoi = np.nanstd(arr_hoi, axis=0, ddof=1)
    mean_pair = np.nanmean(arr_pair, axis=0)
    std_pair = np.nanstd(arr_pair, axis=0, ddof=1)
    mean_sur = np.nanmean(arr_sur, axis=0)
    std_sur = np.nanstd(arr_sur, axis=0, ddof=1)

    # p-value per K between the two "synthetic" curves (hybrid vs pairwise)
    pvals_hybrid_vs_pair = []
    for j, _ in enumerate(k_clusters):
        x = arr_hoi[:, j]
        y = arr_pair[:, j]
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() > 1:
            _, pval = ttest_rel(x[mask], y[mask])
        else:
            x2 = x[~np.isnan(x)]
            y2 = y[~np.isnan(y)]
            if len(x2) > 1 and len(y2) > 1:
                _, pval = ttest_ind(x2, y2, equal_var=False)
            else:
                pval = np.nan
        pvals_hybrid_vs_pair.append(pval)
    pvals_hybrid_vs_pair = np.array(pvals_hybrid_vs_pair)

    # K selected for the violin plot
    if k_selected is not None:
        a = np.asarray(nmis_emp_syn_hoi_selected, dtype=float)
        b = np.asarray(nmis_emp_syn_pair_selected, dtype=float)
        c = np.asarray(nmis_emp_sur_selected, dtype=float)
        mask_ab = ~np.isnan(a) & ~np.isnan(b)
        p_syn_hoi_syn_pair = np.nan
        if mask_ab.sum() > 1:
            _, p_syn_hoi_syn_pair = ttest_rel(a[mask_ab], b[mask_ab])

    # --- Main plot: lines with markers + bars (±1 SD) ---
    ks = np.array(k_clusters, dtype=float)
    fig, ax = plt.subplots(figsize=fig_size)

    if ks.size > 1:
        uniq = np.unique(np.sort(ks))
        step = float(np.min(np.diff(uniq))) if uniq.size > 1 else 1.0
    else:
        step = 1.0
    width = 0.8 * step

    # 1) hybrid
    (h1,) = ax.plot(
        ks,
        mean_hoi,
        linestyle="-",
        marker="o",
        markersize=6,
        markerfacecolor="auto",
        label="Empirical / (hybrid) Synthetic",
        color="C0",
    )
    color1 = h1.get_color()
    lower = mean_hoi - std_hoi
    upper = mean_hoi + std_hoi
    ax.bar(
        ks,
        height=(upper - lower),
        bottom=lower,
        width=width,
        color=color1,
        alpha=0.25,
        align="center",
        edgecolor="none",
        linewidth=0,
    )

    # 2) pairwise
    (h2,) = ax.plot(
        ks,
        mean_pair,
        linestyle="-",
        marker="o",
        markersize=6,
        markerfacecolor="auto",
        label="Empirical / (pairwise) Synthetic",
        color="C2",
    )
    color2 = h2.get_color()
    lower = mean_pair - std_pair
    upper = mean_pair + std_pair
    ax.bar(
        ks,
        height=(upper - lower),
        bottom=lower,
        width=width,
        color=color2,
        alpha=0.25,
        align="center",
        edgecolor="none",
        linewidth=0,
    )

    # 3) surrogate
    (h3,) = ax.plot(
        ks,
        mean_sur,
        linestyle="-",
        marker="o",
        markersize=6,
        markerfacecolor="auto",
        label="Empirical / Surrogate",
        color="C1",
    )
    color3 = h3.get_color()
    lower = mean_sur - std_sur
    upper = mean_sur + std_sur
    ax.bar(
        ks,
        height=(upper - lower),
        bottom=lower,
        width=width,
        color=color3,
        alpha=0.25,
        align="center",
        edgecolor="none",
        linewidth=0,
    )

    # --- Red circles where p < 0.05 (between hybrid and pairwise) ---
    added_legend = False
    for j, k in enumerate(ks):
        p = pvals_hybrid_vs_pair[j]
        if not np.isnan(p) and p < 0.05:
            y_mid = 0.5 * (mean_hoi[j] + mean_pair[j])
            ax.plot(
                k,
                y_mid,
                "o",
                markersize=markersize,
                markerfacecolor="none",
                markeredgecolor="red",
                markeredgewidth=2,
                label=(
                    "p ≤ 0.05 (hybrid vs pairwise)"
                    if not added_legend
                    else "_nolegend_"
                ),
            )
            added_legend = True

    ax.set_xlabel("$K$ (number of clusters)", fontsize=label_fontsize)
    ax.set_ylabel("NMIS (Mean ± 1 SD)", fontsize=label_fontsize)
    ax.set_title(
        f"Similarity between empirical, synthetic and surrogate communities: {fig_title}",
        fontsize=title_fontsize,
    )
    ax.legend(fontsize=legend_fontsize, loc="upper left")
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    ax.set_xticks(ks)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if fig_name is not None:
        plt.savefig(fig_name, dpi=300, bbox_inches="tight")
    plt.show()

    # --- VIOLIN PLOT SPLIT (seaborn) ---
    if k_selected is not None:
        valid_mask = (~np.isnan(a)) & (~np.isnan(b))

        df_vio = pd.DataFrame(
            {
                "NMIS": np.concatenate([a[valid_mask], b[valid_mask]]),
                "Type": (
                    ["Empirical $\Leftrightarrow$ (hybrid) Synthetic"]
                    * valid_mask.sum()
                    + ["Empirical $\Leftrightarrow$ (pairwise) Synthetic"]
                    * valid_mask.sum()
                ),
                "Group": ["NMIS comparison"] * (2 * valid_mask.sum()),
            }
        )

        if not np.isnan(p_syn_hoi_syn_pair):
            p_title = f"$p$-value = {p_syn_hoi_syn_pair:.1e}"
        else:
            p_title = "$p$ = NA"

        plt.figure(figsize=violin_fig_size)
        sns.violinplot(
            data=df_vio,
            x="Group",
            y="NMIS",
            hue="Type",
            split=True,
            inner="quart",
            palette={
                "Empirical $\Leftrightarrow$ (hybrid) Synthetic": "C0",
                "Empirical $\Leftrightarrow$ (pairwise) Synthetic": "C2",
            },
            linewidth=1.0,
        )

        plt.title(
            f"{fig_title}: NMIS distributions at $K$ = {k_selected}, {p_title}",
            fontsize=title_fontsize,
        )
        plt.ylabel("NMIS", fontsize=label_fontsize)
        plt.xlabel("")
        plt.legend(title="", fontsize=legend_fontsize, loc="upper right")
        plt.grid(True, linestyle="--", alpha=0.4)
        plt.tick_params(axis="both", which="major", labelsize=tick_fontsize)
        plt.tight_layout()
        if violin_fig_name is not None:
            plt.savefig(violin_fig_name, dpi=300, bbox_inches="tight")
        plt.show()


def plot_nmi_curves(
    df_list,
    labels=None,
    colors=None,
    title="NMI across subjects vs number of clusters (K)",
    title_fontsize=20,
    legend_fontsize=20,
    label_fontsize=20,
    tick_fontsize=20,
    figsize=(8, 5),
    show=True,
    save_path=None,
):
    """
    Plot NMI (Normalized Mutual Information) curves across subjects as a function of
    the number of clusters (K).
    The function expects one or more pandas DataFrames, each containing two columns:
    - "k": number of clusters (can be numeric; values are grouped and sorted)
    - "NMI": NMI values for the corresponding k
    For each DataFrame the function computes the per-k mean and standard deviation
    of NMI, then plots:
    - a curve of mean NMI vs k (first curve solid, subsequent curves dashed),
    - a marker for each k (markers cycle through a default set),
    - a discretized shaded band at each k representing mean ± 1 std (drawn as narrow
        bar-like patches, clipped to [0, 1]).
    Args:
            df_list (list[pandas.DataFrame] or tuple[pandas.DataFrame]):
                    Non-empty sequence of DataFrames. Each DataFrame must contain columns
                    "k" and "NMI". The data are grouped by "k" and aggregated by mean and std.
            labels (list[str], optional):
                    Labels for the plotted curves. If None, default labels "Curve 1", "Curve 2",
                    ... are generated. Length must equal len(df_list) when provided.
            colors (list[str] or list[tuple], optional):
                    Colors to use for each curve (any matplotlib-acceptable color spec).
                    If None, colors are chosen automatically. When provided, length must match
                    len(df_list).
            title (str, optional):
                    Plot title. Default: "NMI across subjects vs number of clusters (K)".
            title_fontsize (int or float, optional):
                    Font size for the title. Default: 20.
            legend_fontsize (int or float, optional):
                    Font size for the legend. Default: 20.
            label_fontsize (int or float, optional):
                    Font size for the x/y axis labels. Default: 20.
            tick_fontsize (int or float, optional):
                    Font size for axis tick labels. Default: 20.
            figsize (tuple(float, float), optional):
                    Figure size in inches (width, height). Default: (8, 5).
            show (bool, optional):
                    If True (default) call plt.show() after drawing. If False, the figure is
                    closed (plt.close(fig)) so no interactive output is produced.
            save_path (str or path-like, optional):
                    If provided, the figure is saved to this path using dpi=150 and
                    bbox_inches='tight' before showing/closing.
    Raises:
            ValueError:
                    - If df_list is not a non-empty list/tuple.
                    - If any DataFrame in df_list does not contain the required columns
                        {"k", "NMI"}.
                    - If labels is provided but its length does not match len(df_list).
                    - If colors is provided but its length does not match len(df_list).
    Notes / Plot details:
            - For each DataFrame, NaN standard deviations (e.g., when a single sample
                exists for a k) are replaced with 0.0.
            - The shaded uncertainty is drawn per-k as a bar with width equal to
                0.8 * min(diff(sorted(ks))). If only one k exists, a default step of 1.0
                is used.
            - Mean ± std are clipped to the interval [0, 1] for plotting.
            - The first curve is drawn with a solid line ('-'); subsequent curves use
                dashed lines ('--'). Markers cycle through a small set of symbols.
            - Axis limits: y is fixed to (0, 1). Grid lines are enabled with dashed style.
            - The function returns None.
    """

    if not isinstance(df_list, (list, tuple)) or len(df_list) == 0:
        raise ValueError("df_list must be a non-empty list of DataFrames.")

    n = len(df_list)
    if labels is None:
        labels = [f"Curve {i + 1}" for i in range(n)]
    if len(labels) != n:
        raise ValueError("labels length must match df_list length.")
    if colors is not None and len(colors) != n:
        raise ValueError("colors length must match df_list length or be None.")

    # Markers
    default_markers = ["o", "s", "D", "P", "X", "*", "<", ">", "h", "8"]
    markers_use = [default_markers[i % len(default_markers)] for i in range(n)]

    # Aggregate per df
    summaries = []
    for df in df_list:
        if not {"k", "NMI"}.issubset(df.columns):
            raise ValueError("Each DataFrame must have columns ['k', 'NMI'].")
        summary = (
            df.groupby("k")["NMI"]
            .agg(mean="mean", std="std")
            .reset_index()
            .sort_values("k")
        )
        summary["std"] = summary["std"].fillna(0.0)
        summaries.append(summary)

    fig, ax = plt.subplots(figsize=figsize)
    for i, (summary, label) in enumerate(zip(summaries, labels)):
        ks = summary["k"].astype(int).values
        mean_nmi = summary["mean"].values
        std_nmi = summary["std"].values

        lower = np.clip(mean_nmi - std_nmi, 0.0, 1.0)
        upper = np.clip(mean_nmi + std_nmi, 0.0, 1.0)

        line_kwargs = {}
        if colors is not None:
            line_kwargs["color"] = colors[i]

        # Style: first curve solid with filled markers, others dashed with hollow markers
        linestyle = "-" if i == 0 else "--"

        (h,) = ax.plot(
            ks,
            mean_nmi,
            linestyle=linestyle,
            marker=markers_use[i],
            markersize=6,
            markerfacecolor="auto",
            label=label,
            **line_kwargs,
        )
        this_color = h.get_color()

        # Discretized shaded part: bar-like bands for ±1 SD at each k
        if len(ks) > 1:
            step = float(np.min(np.diff(ks)))
        else:
            step = 1.0
        width = 0.8 * step  # a bit narrower than the spacing
        ax.bar(
            ks,
            height=(upper - lower),
            bottom=lower,
            width=width,
            color=this_color,
            alpha=0.2,
            align="center",
            edgecolor="none",
            linewidth=0,
        )

    ax.set_title(title, fontsize=title_fontsize)
    ax.set_xlabel("$K$ (number of clusters)", fontsize=label_fontsize)
    ax.set_ylabel("NMIS", fontsize=label_fontsize)
    ax.set_ylim((0, 1))
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=legend_fontsize)
    ax.tick_params(axis="both", which="major", labelsize=tick_fontsize)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return


def plot_correlation_scatters(
    df,
    metric,
    xlabel,
    ylabel,
    z_score_thr=2,
    figsize=(10, 5),
    fig_dir=None,
    plot_show=True,
):
    """Plot the correlation scatters of the peak values between males and females
    in each resting-state scan for specified metrics.

    df : pandas.DataFrame
        DataFrame containing the data to be plotted.

    metrics : list of str
        List of metric names to be analyzed.

    xlabels : list of str
        List of x-axis labels for the plots.

    ylabels : list of str
        List of y-axis labels for the plots.

    z_score_thr : int, optional
        Z-score threshold for outlier removal, by default 2.

    figsize : tuple, optional
        Size of the figure, by default (15, 5).

    fig_dir : str, optional
        Directory to save the figure, by default None.

    plot_show : bool, optional
        Whether to show the plot, by default True.

    Returns
    -------
    None
    """

    ncols = 1
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, gridspec_kw={"hspace": 0.5})

    metric_rest1 = metric + "_REST1"
    metric_rest2 = metric + "_REST2"
    df_overall_rest1 = df.loc[df["REST"] == 1][["Subject", metric]]
    df_overall_rest1 = remove_outliers_zscore(df_overall_rest1, metric, z_score_thr)
    df_overall_rest1 = df_overall_rest1.rename(columns={metric: metric_rest1})

    df_overall_rest2 = df.loc[df["REST"] == 2][["Subject", metric]]
    df_overall_rest2 = remove_outliers_zscore(df_overall_rest2, metric, z_score_thr)
    df_overall_rest2 = df_overall_rest2.rename(columns={metric: metric_rest2})

    df_overall = df_overall_rest2.merge(df_overall_rest1, how="inner")
    df_overall = df_overall.dropna()
    rest1_overall = np.array(df_overall[metric_rest1].to_list()).flatten()
    rest2_overall = np.array(df_overall[metric_rest2].to_list()).flatten()
    r, p = pearsonr(rest1_overall, rest2_overall)

    sns.regplot(
        ax=axes,
        x=rest1_overall,
        y=rest2_overall,
        color="steelblue",
        # label=r"$II$",
        scatter=True,
    )
    axes.set_title(
        f"R = {r:.2f} / $p$-value$_{{Bonferroni-corrected}}$ = {p * 2:.2g}",
        fontsize=20,
    )
    axes.set_xlabel(xlabel, fontsize=20)
    axes.set_ylabel(ylabel, fontsize=20)
    # axes.legend(fontsize=16)
    axes.tick_params(axis="both", which="major", labelsize=18)

    if fig_dir != None:
        plt.savefig(fig_dir)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return


def remove_outliers_zscore(df, column, threshold=2):
    """Remove outliers from the 'column' of the Pandas Dataframe 'df' using z-score threshold
    and return the corresponding collumn without outliers.

    Parameters
    ----------
    df: Pandas Dataframe

    column: string
    """
    x = df[column].to_list()
    x = np.array(x).flatten()
    z = np.abs(zscore(x))
    inliers = np.where(z <= threshold)[0]

    return df.iloc[inliers]


def plot_metrics_distribution(
    df, metrics, ylabels, zscore_thr=2, figsize=(15, 6), fig_dir=None, plot_show=True
):
    """Plot the distribution of the peak values between males and females in each resting-state scan at k=0,4 for II and TC metrics.

    df : pandas.DataFrame
        DataFrame containing the data to be plotted. Must include columns 'Subject', 'REST', 'Gender', and the metrics specified.

    metrics : list of str
        List of metric column names to be plotted.

    ylabels : list of str
        List of y-axis labels corresponding to each metric.

    zscore_thr : int, optional
        Z-score threshold for outlier removal (default is 2).

    figsize : tuple, optional
        Size of the figure (default is (15, 6)).

    fig_dir : str or None, optional
        Directory to save the figure. If None, the figure is not saved (default is None).

    plot_show : bool, optional
        If True, the plot is shown. If False, the plot is closed (default is True).

    Returns
    -------
    None
    """

    fig, ax = plt.subplots(
        nrows=1, ncols=2, figsize=figsize, gridspec_kw={"hspace": 0.5}
    )

    for idx, axes in enumerate(ax.flat):
        metric = metrics[idx]
        df_male_rest1_metric = df.loc[(df["Gender"] == "M") & (df["REST"] == 1)][
            ["Subject", "REST", "Gender", metric]
        ]
        df_female_rest1_metric = df.loc[(df["Gender"] == "F") & (df["REST"] == 1)][
            ["Subject", "REST", "Gender", metric]
        ]
        df_male_rest2_metric = df.loc[(df["Gender"] == "M") & (df["REST"] == 2)][
            ["Subject", "REST", "Gender", metric]
        ]
        df_female_rest2_metric = df.loc[(df["Gender"] == "F") & (df["REST"] == 2)][
            ["Subject", "REST", "Gender", metric]
        ]

        df_male_rest1_metric = remove_outliers_zscore(
            df_male_rest1_metric, metric, zscore_thr
        )
        df_female_rest1_metric = remove_outliers_zscore(
            df_female_rest1_metric, metric, zscore_thr
        )
        df_male_rest2_metric = remove_outliers_zscore(
            df_male_rest2_metric, metric, zscore_thr
        )
        df_female_rest2_metric = remove_outliers_zscore(
            df_female_rest2_metric, metric, zscore_thr
        )

        df_rest = pd.concat(
            [
                df_male_rest1_metric,
                df_female_rest1_metric,
                df_male_rest2_metric,
                df_female_rest2_metric,
            ],
            axis=0,
        )
        sns.violinplot(
            x="REST",
            y=metric,
            data=df_rest,
            hue="Gender",
            split=True,
            inner="quart",
            ax=axes,
            cut=2,  # não estende além dos dados
            bw_adjust=1,  # menos suavização -> menos 'cauda fantasma'
            gridsize=100,  # KDE mais precisa nos extremos
        )
        F_mann_rest1, p_mann_rest1 = mannwhitneyu(
            df_male_rest1_metric[metric], df_female_rest1_metric[metric]
        )
        F_mann_rest2, p_mann_rest2 = mannwhitneyu(
            df_male_rest2_metric[metric], df_female_rest2_metric[metric]
        )
        axes.set_title(
            f"REST 1: $p$-value$_{{Bonferroni-corrected}}$ = {p_mann_rest1 * 4:.2g}\nREST 2: $p$-value$_{{Bonferroni-corrected}}$ = {p_mann_rest2 * 4:.2g}",
            fontsize=20,
        )
        axes.set_ylabel(ylabels[idx], fontsize=20)
        axes.set_xlabel("rs-fMRI recording (REST)", fontsize=20)
        axes.legend(title="Sex", fontsize=18, title_fontsize=18)
        axes.tick_params(axis="both", which="major", labelsize=20)
        vals = df_rest[metric].dropna().values
        if vals.size > 0:
            vmin, vmax = vals.min(), vals.max()
            pad = 0.02 * (vmax - vmin if vmax > vmin else 1.0)
            axes.set_ylim(vmin - pad, vmax + pad)

    if fig_dir != None:
        fig.savefig(fig_dir)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return


def plot_models_silhouette_diagrams(
    list_kclusters,
    list_embeddings,
    list_kmeans_models,
    nodes_color=[],
    name_dataset="",
    fig_size=None,
    fig_dir=None,
    plot_show=True,
):
    """Plot the silhouette diagrams of K-Means models in 'list_kmeans_models'.

    Parameters
    ----------
    list_kclusters : list of int
        List of K clusters for which the silhouette diagram is to be plotted.

    list_embeddings : list of numpy.ndarray
        List of the corresponding embeddings.

    list_kmeans_models : list of KMeans
        List of K-Means objects.

    nodes_color : list of str, optional
        List of nodes color. Default is an empty list.

    name_dataset : str, optional
        Name of the dataset. Default is an empty string.

    fig_size : tuple of int, optional
        Figure size as (width, height). Default is None.

    fig_dir : str, optional
        Directory to save the figure. Default is None.

    plot_show : bool, optional
        Whether to show the plot. Default is True.

    Returns
    -------
    None

    """

    num_subplots = len(list_kclusters)
    ncols = 2
    nrows = (num_subplots + 1) // 2
    fig_width = 12
    fig_height = 4 * num_subplots
    if fig_size != None:
        fig_width, fig_height = fig_size[0], fig_size[1]
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(fig_width, fig_height), gridspec_kw={"hspace": 0.25}
    )

    percent = {}
    for i, ax in enumerate(axes.flat):
        if i < num_subplots:
            y_pred = list_kmeans_models.get(list_kclusters[i])
            embedding = list_embeddings.get(list_kclusters[i])
            silhouette_coefficients = silhouette_samples(embedding, y_pred)
            silhouette_avg = silhouette_coefficients.mean()
            df_sc = pd.DataFrame({"sc": silhouette_coefficients, "cluster": y_pred})
            if len(nodes_color) == len(y_pred):
                df_sc["color"] = nodes_color
            else:
                df_sc["color"] = [px.colors.qualitative.Light24[idx] for idx in y_pred]

            y_lower_global = 0
            for c in range(list_kclusters[i]):
                cth_cluster_silhouette_values = df_sc.loc[
                    df_sc["cluster"] == c
                ].sort_values(by="sc")
                size_cluster_c = len(cth_cluster_silhouette_values)
                y_lower_local = y_lower_global
                for j in range(size_cluster_c):
                    y_upper_local = y_lower_local + 20
                    ax.fill_betweenx(
                        np.arange(y_lower_local, y_upper_local),
                        0,
                        cth_cluster_silhouette_values.iloc[j]["sc"],
                        facecolor=cth_cluster_silhouette_values.iloc[j]["color"],
                        alpha=0.7,
                    )
                    y_lower_local = y_upper_local + 2

                # Label the silhouette plots with their cluster numbers at the middle
                ax.text(
                    -0.05,
                    y_lower_global + 0.5 * size_cluster_c * 20,
                    str(c),
                    fontsize=20,
                )

                # Compute the new y_lower for next plot
                y_lower_global = y_upper_local + 100  # 10 for the 0 samples

            percent[list_kclusters[i]] = (
                100 * len(df_sc[df_sc["sc"] > silhouette_avg]) / len(df_sc)
            )

            ax.set_title(
                f"Silhouette diagrams {name_dataset}, $K={list_kclusters[i]}$, $SC={silhouette_avg:.3f}$",
                fontsize=18,
            )
            ax.set_xlabel("Silhouette coefficients $\eta$", fontsize=20)
            ax.set_ylabel(r"Cluster labels", fontsize=20)
            # The vertical line for average silhouette score of all the values
            ax.axvline(x=silhouette_avg, color="black", linestyle="--")
            ax.set_yticks([])  # Clear the yaxis labels / ticks
            ax.set_xticks(np.arange(-0.1, 0.9, 0.1))
            ax.tick_params(axis="x", labelsize=16)

    if num_subplots % 2 != 0:
        if num_subplots > 1:
            axes[-1, -1].axis("off")
        else:
            axes[-1].axis("off")

    plt.tight_layout()
    if fig_dir != None:
        plt.savefig(fig_dir)
    if plot_show:
        plt.show()
    else:
        plt.close()

    return


def plot_node_clustering_distribution(
    nodes_cluster,
    nodes_color,
    nodes_subnet,
    fig_title="",
    file_dir=None,
    plot_show=True,
    ncols=3,
    hole_width=0.35,
    label_fontsize=14,  # fonte geral (não usada para label externa)
    pct_fontsize=16,  # tamanho da fonte do percentual local
    pct2_fontsize=12,  # tamanho da fonte do percentual global (abaixo)
    legend_fontsize=14,
    center_fontsize=18,  # número no centro
    center_fontweight="bold",
    center_color="black",
    figsize_per_col=4,
    figsize_per_row=4.2,
    dpi=300,
):
    df_brain = pd.DataFrame(
        {"cluster": nodes_cluster, "color": nodes_color, "subnet": nodes_subnet}
    )

    cluster_list = sorted(set(nodes_cluster))
    subnets_order = df_brain["subnet"].drop_duplicates().tolist()
    df = pd.DataFrame(0, index=cluster_list, columns=subnets_order)
    for subnet in subnets_order:
        for cl in cluster_list:
            df.loc[cl, subnet] = (
                (df_brain["subnet"] == subnet) & (df_brain["cluster"] == cl)
            ).sum()

    color_map = df_brain.groupby("subnet")["color"].first().reindex(subnets_order)
    colors = color_map.values.tolist()

    nodes_distribution = df.sum(axis=0).values
    nodes_distribution_safe = np.where(nodes_distribution == 0, 1, nodes_distribution)

    num_subplots = len(cluster_list)
    nrows = ceil(num_subplots / ncols)
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(figsize_per_col * ncols, figsize_per_row * nrows),
    )
    axes = np.atleast_2d(axes)
    axes_flat = axes.ravel()

    def _autopct(pct):
        return f"{pct:.0f}%" if pct >= 0.5 else ""

    for ax, cl in zip(axes_flat, cluster_list):
        values = df.loc[cl].values.astype(float)

        # remove setores 0%
        mask_nonzero = values > 0
        values = values[mask_nonzero]
        if len(values) == 0:
            ax.axis("off")
            continue

        colors_filtered = np.array(colors)[mask_nonzero]
        overall_pct = 100.0 * values / nodes_distribution_safe[mask_nonzero]

        wedges, texts, autotexts = ax.pie(
            values,
            labels=None,  # sem nomes das subnets
            labeldistance=1.1,
            pctdistance=0.68,
            startangle=90,
            counterclock=True,
            colors=colors_filtered,
            wedgeprops=dict(width=hole_width, edgecolor="black", linewidth=1),
            autopct=_autopct,
            textprops={"fontsize": pct_fontsize, "color": "black", "ha": "center"},
        )

        for w, t, g_pct in zip(wedges, autotexts, overall_pct):
            ang = (w.theta2 + w.theta1) / 2
            x = 0.6 * np.cos(np.deg2rad(ang))
            y = 0.6 * np.sin(np.deg2rad(ang)) - 0.3
            ax.text(
                x,
                y,
                f"({g_pct:.0f}%)",
                ha="center",
                va="center",
                fontsize=pct2_fontsize,
                color="black",
            )

        ax.text(
            0.0,
            0.0,
            f"{int(cl)}",
            ha="center",
            va="center",
            fontsize=center_fontsize,
            fontweight=center_fontweight,
            color=center_color,
        )

        ax.axis("equal")

    for j in range(len(cluster_list), len(axes_flat)):
        axes_flat[j].axis("off")

    fig.suptitle(fig_title, fontsize=label_fontsize + 4, y=0.98)
    handles = [
        mpatches.Patch(facecolor=c, edgecolor="black", label=s)
        for s, c in zip(subnets_order, colors)
    ]
    fig.legend(
        handles=handles,
        title="Brain functional\nsub-networks",
        fontsize=legend_fontsize,
        title_fontsize=legend_fontsize,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
    )
    fig.tight_layout(rect=[0.02, 0.02, 0.95, 0.95])

    if file_dir:
        plt.savefig(file_dir, dpi=dpi, bbox_inches="tight")
    if plot_show:
        plt.show()
    else:
        plt.close(fig)


def plot_hypergraph_modes_strength(
    hoi_weights_dir,
    hoi_weights_surrogate_dir=None,
    hoi_weights_synthetic_dir=None,
    hoi_labels=all_triangles,
    n_modes=51,
    file_name=None,
    labels=("Empirical", "Synthetic", "Surrogate"),
    colors=("C0", "C2", "C1"),
    show=True,
    label_fontsize=18,
    legend_fontsize=14,
    title_fontsize=18,
    tick_fontsize=12,
    figsize=(10, 10),
):
    """
    Plot per-mode hypergraph strength (sum of absolute upper-triangular weights) for A_ii and A_tc.
    Can overlay Original, Synthetic and Surrogate curves if their paths are provided.

    Parameters
    ----------
    hoi_weights_dir : str
        Path to original HOI weights (.npy).
    n_modes : int
        Number of modes to plot (will be clipped to available modes).
    file_name : str or None
        If provided, saves figure to file_name.
    hoi_weights_dir_surrogate : str or None
        Path to surrogate HOI weights (.npy).
    hoi_weights_dir_synthetic : str or None
        Path to synthetic HOI weights (.npy).
    labels : tuple
        Labels for (Original, Synthetic, Surrogate).
    colors : tuple
        Colors for (Original, Synthetic, Surrogate).
    show : bool
        Whether to show the plot.
    label_fontsize, legend_fontsize, title_fontsize, tick_fontsize : int
        Font sizes.
    """

    def mode_strengths(As):
        m = As.shape[2]
        sums = np.zeros(m, dtype=float)
        for mode in range(m):
            A_mode = np.abs(As[:, :, mode])
            sums[mode] = np.mean(pack_upper(A_mode))
        return sums

    # Load original tensors
    As_ii_o, As_tc_o = get_symmetrized_t_fft(hoi_weights_dir, hoi_labels)
    strengths = []
    meta = []

    # Original
    s_ii_o = mode_strengths(As_ii_o)
    s_tc_o = mode_strengths(As_tc_o)
    strengths.append(("orig", s_ii_o, s_tc_o))
    meta.append(
        (
            "orig",
            labels[0] if len(labels) > 0 else "Empirical",
            colors[0] if len(colors) > 0 else "C0",
            "o",
            "-",
        )
    )

    # Synthetic (optional)
    if hoi_weights_synthetic_dir is not None:
        As_ii_syn, As_tc_syn = get_symmetrized_t_fft(
            hoi_weights_synthetic_dir, hoi_labels
        )
        s_ii_syn = mode_strengths(As_ii_syn)
        s_tc_syn = mode_strengths(As_tc_syn)
        strengths.append(("synthetic", s_ii_syn, s_tc_syn))
        meta.append(
            (
                "synthetic",
                labels[1] if len(labels) > 1 else "Synthetic",
                colors[1] if len(colors) > 1 else "C2",
                "s",
                "--",
            )
        )

    # Surrogate (optional)
    if hoi_weights_surrogate_dir is not None:
        As_ii_surr, As_tc_surr = get_symmetrized_t_fft(
            hoi_weights_surrogate_dir, hoi_labels
        )
        s_ii_surr = mode_strengths(As_ii_surr)
        s_tc_surr = mode_strengths(As_tc_surr)
        strengths.append(("surrogate", s_ii_surr, s_tc_surr))
        idx = 2 if len(labels) > 2 else -1
        color_idx = 2 if len(colors) > 2 else -1
        meta.append(
            (
                "surrogate",
                labels[idx] if idx != -1 else "Surrogate",
                colors[color_idx] if color_idx != -1 else "C1",
                "D",
                "--",
            )
        )

    # Determine common number of modes
    available = [len(s_ii) for _, s_ii, _ in strengths]
    m_common = min([n_modes] + available) if n_modes is not None else min(available)
    modes = np.arange(m_common)

    # Truncate all to common modes
    strengths = [
        (name, s_ii[:m_common], s_tc[:m_common]) for (name, s_ii, s_tc) in strengths
    ]

    # Top-2 modes by strength on original after truncation
    s_ii_o_trunc = strengths[0][1]
    s_tc_o_trunc = strengths[0][2]
    top2_ii = np.argsort(s_ii_o_trunc)[-2:][::-1]
    top2_tc = np.argsort(s_tc_o_trunc)[-2:][::-1]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=False)
    ax1, ax2 = axes

    # Plot A_ii
    for (name, s_ii, _), (l, label, color, marker, linestyle) in zip(strengths, meta):
        if l == "orig":
            label += " $\\mathcal{G}_{OI}^{(k)}$"
        elif l == "synthetic":
            label += " $\\breve{\\mathcal{G}}_{OI}^{(k)}$"
        elif l == "surrogate":
            label += " $\\widetilde{\\mathcal{G}}_{OI}^{(k)}$"
        ax1.plot(
            modes,
            s_ii,
            marker=marker,
            linestyle=linestyle,
            label=label,
            color=color,
        )
    # Vertical lines for original top-2
    label_added = False
    for m in top2_ii:
        ax1.axvline(
            m,
            color=meta[0][2],
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            label=("Top-2 modes (empirical): " + str(list(map(int, top2_ii))))
            if not label_added
            else None,
        )
        label_added = True
    ax1.set_xlabel("Mode $k$", fontsize=label_fontsize)
    ax1.set_ylabel("Connection strength", fontsize=label_fontsize)
    ax1.set_title(
        r"Hypergraph mode strength $\mathcal{G}_{OI}^{(k)}$", fontsize=title_fontsize
    )
    ax1.tick_params(axis="both", labelsize=tick_fontsize)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(fontsize=legend_fontsize)

    # Plot A_tc
    for (name, _, s_tc), (l, label, color, marker, linestyle) in zip(strengths, meta):
        if l == "orig":
            label += " $\\mathcal{G}_{TC}^{(k)}$"
        elif l == "synthetic":
            label += " $\\breve{\\mathcal{G}}_{TC}^{(k)}$"
        elif l == "surrogate":
            label += " $\\widetilde{\\mathcal{G}}_{TC}^{(k)}$"
        ax2.plot(
            modes,
            s_tc,
            marker=marker,
            linestyle=linestyle,
            label=label,
            color=color,
        )
    # Vertical lines for original top-2
    label_added = False
    for m in top2_tc:
        ax2.axvline(
            m,
            color=meta[0][2],
            linestyle="--",
            alpha=0.8,
            linewidth=1.5,
            label=("Top-2 modes (empirical): " + str(list(map(int, top2_tc))))
            if not label_added
            else None,
        )
        label_added = True
    ax2.set_xlabel("Mode $k$", fontsize=label_fontsize)
    ax2.set_ylabel("Connection strength", fontsize=label_fontsize)
    ax2.set_title(
        r"Hypergraph mode strength $\mathcal{G}_{TC}^{(k)}$", fontsize=title_fontsize
    )
    ax2.tick_params(axis="both", labelsize=tick_fontsize)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(fontsize=legend_fontsize)

    plt.tight_layout()
    if file_name:
        plt.savefig(file_name, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)
