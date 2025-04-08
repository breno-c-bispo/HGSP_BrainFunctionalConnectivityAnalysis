# Basic data manipulation and visualisation libraries
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from plotly.subplots import make_subplots
from HCP_Data_Vis_Schaefer_100Parcels import G_den
import networkx as nx

mpl.rcParams["text.usetex"] = True
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["font.family"] = "Times New Roman"

# Scikit-learn libraries
from itertools import combinations
from math import ceil

from scipy import stats
from sklearn.metrics import silhouette_samples
from utils import (
    get_modularity,
    lower_diagonal_exclusive,
    remove_outliers_zscore,
)


def plot_multiple_stem(
    list_df,
    xlabels,
    ylabels,
    ncols,
    xticks=None,
    yticks=None,
    xlims=None,
    ylims=None,
    markersize=2,
    figsize=(15, 5),
    fig_name=None,
):
    """Plot multiple stem charts from a list of dataframes.

    list_df : list of pandas.DataFrame
        List of dataframes, each containing 'x' and 'y' columns for plotting.

    xlabels : list of str
        List of x-axis labels for each subplot.

    ylabels : list of str
        List of y-axis labels for each subplot.

    ncols : int
        Number of columns in the subplot grid.

    xticks : list of list of float, optional
        List of x-tick values for each subplot. Default is None.

    yticks : list of list of float, optional
        List of y-tick values for each subplot. Default is None.

    xlims : list of tuple of float, optional
        List of x-axis limits for each subplot. Each tuple contains (min, max). Default is None.

    ylims : list of tuple of float, optional
        List of y-axis limits for each subplot. Each tuple contains (min, max). Default is None.

    markersize : int, optional
        Size of the markers in the stem plot. Default is 2.

    figsize : tuple of int, optional
        Size of the figure. Default is (15, 5).

    fig_name : str, optional
        Name of the file to save the figure. If None, the figure is not saved. Default is None.

    Returns
    -------
    None
    """

    num_subplots = len(list_df)
    nrows = (num_subplots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for n, ax in enumerate(axes.flat):
        x = list_df[n]["x"]
        y = list_df[n]["y"]

        markerline, stemline, baseline = ax.stem(x, y, linefmt="k--", markerfmt="bo")

        plt.setp(stemline, linewidth=1.0)
        plt.setp(markerline, markersize=markersize)

        ax.set_ylabel(ylabels[n], fontsize=14)
        ax.set_xlabel(xlabels[n], fontsize=12)
        if xlims != None and xlims[n] != None:
            ax.set_xlim(xlims[n][0], xlims[n][1])
        if ylims != None and ylims[n] != None:
            ax.set_ylim(ylims[n][0], ylims[n][1])
        if yticks != None:
            ax.set_yticks(yticks[n])
        if xticks != None:
            ax.set_xticks(xticks[n])

    if fig_name != None:
        plt.savefig(fig_name)
    plt.show()
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

    dict_node_labels = {}
    dict_embeddings = {}
    for idx, kmeans_model in enumerate(list_kmeans_models):
        n_clusters = len(set(kmeans_model.labels_))
        dict_node_labels[n_clusters] = kmeans_model.labels_
        dict_embeddings[n_clusters] = list_embeddings[idx]
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
            y_pred = dict_node_labels.get(list_kclusters[i])
            embedding = dict_embeddings.get(list_kclusters[i])
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
                    fontsize=14,
                )

                # Compute the new y_lower for next plot
                y_lower_global = y_upper_local + 100  # 10 for the 0 samples

            percent[list_kclusters[i]] = (
                100 * len(df_sc[df_sc["sc"] > silhouette_avg]) / len(df_sc)
            )

            ax.set_title(
                f"Silhouette diagrams {name_dataset}, $K={list_kclusters[i]}$, $SC={silhouette_avg:.3f}$",
                fontsize=14,
            )
            ax.set_xlabel(f"Silhouette coefficients $\eta$", fontsize=14)
            ax.set_ylabel(r"Cluster labels", fontsize=14)
            # The vertical line for average silhouette score of all the values
            ax.axvline(x=silhouette_avg, color="black", linestyle="--")
            ax.set_yticks([])  # Clear the yaxis labels / ticks
            ax.set_xticks(np.arange(-0.1, 0.9, 0.1))

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


def plot_pie_clusters(
    models_list,
    k_cluster,
    nodes_color,
    nodes_subnet,
    fig_title="",
    file_dir=None,
    plot_show=True,
):
    """Plot a pie chart of the nodes distribution among the clusters.

    models_list : list of KMeans objects
        List of K-Means models.

    k_cluster : int
        Number of clusters to be plotted.

    nodes_color : list of str
        List of colors for each node.

    nodes_subnet : list of str
        List of subnets where each node belongs.

    fig_title : str, optional
        Title of the figure (default is an empty string).

    file_dir : str, optional
        Directory to save the plot image (default is None).

    plot_show : bool, optional
        Whether to display the plot (default is True).

    Returns
    -------
    None
    """

    for idx in range(len(models_list)):
        if len(set(models_list[idx].labels_)) == k_cluster:
            break

    nodes_cluster = models_list[idx].labels_
    df_brain = pd.DataFrame(
        {"cluster": nodes_cluster, "color": nodes_color, "subnet": nodes_subnet}
    )

    cluster_list = sorted(set(nodes_cluster))
    df = pd.DataFrame(
        0,
        index=cluster_list,
        columns=df_brain["subnet"].drop_duplicates().values.tolist(),
    )
    for cluster_name in list(df.columns):
        for cluster_label in df.index:
            df.loc[cluster_label, cluster_name] = len(
                df_brain[
                    (df_brain["subnet"] == cluster_name)
                    & (df_brain["cluster"] == cluster_label)
                ]
            )

    num_subplots = len(cluster_list)
    ncols = 3
    nrows = (num_subplots + 2) // 3

    fig = make_subplots(
        rows=nrows,
        cols=ncols,
        specs=np.full((nrows, ncols), {"type": "domain"}).tolist(),
    )
    row = 1
    col = 1
    nodes_distribution = np.sum(df.to_numpy(), axis=0)
    for i in df.index:
        cluster_node_info = df.loc[i]
        overall_percentual = 100 * cluster_node_info.values / nodes_distribution
        percentual = [f"({p:.0f}%)" for idx, p in enumerate(overall_percentual)]
        fig.add_trace(
            go.Pie(
                labels=list(cluster_node_info.index),
                values=cluster_node_info.values,
                name=f"Cluster {i}",
                textinfo="percent+text",
                text=percentual,
            ),
            row,
            col,
        )
        col += 1
        if col > ncols:
            row += 1
            col = 1
    # Use `hole` to create a donut-like pie chart
    fig.update_traces(
        textfont_color="black",
        textfont_size=18,
        textposition="inside",
        hole=0.4,
        hoverinfo="label+percent+name",
        texttemplate="%{percent:.2p}<br>%{text}",
        marker=dict(
            colors=df_brain["color"].drop_duplicates().values.tolist(),
            line=dict(color="black", width=1),
        ),
    )

    fig.update_layout(
        title_text=fig_title,
        autosize=False,
        width=800,
        font_family="Times New Roman",
        height=800,
        uniformtext_minsize=16,
        uniformtext_mode=None,
        legend=dict(font=dict(size=16), title_text="Brain functional\nsubnetworks"),
        margin=dict(l=0, r=0, b=0, t=50, pad=0),
    )

    if file_dir != None:
        fig.write_image(file_dir)

    # Code to plot the pie chart
    if plot_show:
        fig.show()
    else:
        plt.close()

    return


def plot_mean_tubal_scalars(
    As,
    k_indexes,
    title="",
    step_xticks=2,
    figsize=(10, 4),
    fig_dir=None,
    plot_show=True,
):
    """Plot the 'k_indexes' of the mean absolute value of tubal scalars of the 3-order tensor 'As'.

    As : numpy.ndarray
        A 3-order tensor (numpy array) from which the tubal scalars are computed.

    k_indexes : list of int
        List of integer indices to be plotted.

    title : str, optional
        Title of the plot (default is an empty string).

    step_xticks : int, optional
        Step size for x-axis ticks (default is 2).

    figsize : tuple of int, optional
        Size of the figure (default is (10, 4)).

    fig_dir : str or None, optional
        Directory to save the figure. If None, the figure is not saved (default is None).

    plot_show : bool, optional
        If True, the plot is shown. If False, the plot is closed (default is True).

    Returns
    -------
    None
    """

    N0, N1, N2 = As.shape
    l = list(combinations(range(N0), 2))
    tubal_scalars_list = []
    for c in l:
        tubal_scalars_list.append(np.squeeze(As[c[0], c[1], : ceil(N2 / 2) + 1]))
    tubal_scalars_list = np.abs(np.array(tubal_scalars_list))
    mean_tubal_scalars = np.mean(tubal_scalars_list, axis=0)
    df = pd.DataFrame(
        {"k": range(0, len(mean_tubal_scalars)), "mean": mean_tubal_scalars}
    )

    df = df.iloc[k_indexes]

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=figsize)
    # Plot a stem plot of the dafarame
    markerline, stemline, baseline = ax.stem(
        df["k"], df["mean"], linefmt="k--", markerfmt="bo"
    )
    plt.setp(stemline, linewidth=2.0)
    plt.setp(markerline, markersize=4)
    ax.set_xlabel(r"$k$", fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(range(0, len(df), step_xticks))
    if fig_dir != None:
        plt.savefig(fig_dir)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return


def plot_correlation_scatters(
    df,
    metrics,
    xlabels,
    ylabels,
    z_score_thr=2,
    figsize=(15, 5),
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

    ncols = 2
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, gridspec_kw={"hspace": 0.5})

    i = 0
    for idx, metric in enumerate(metrics):
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
        r, p = stats.pearsonr(rest1_overall, rest2_overall)

        if idx % 2 == 0:
            r_prev, p_prev = r, p
            df_overall_prev = df_overall.copy()
            metric_rest1_prev, metric_rest2_prev = metric_rest1, metric_rest2
        else:
            rest1_overall_prev = np.array(
                df_overall_prev[metric_rest1_prev].to_list()
            ).flatten()
            rest2_overall_prev = np.array(
                df_overall_prev[metric_rest2_prev].to_list()
            ).flatten()

            sns.regplot(
                ax=axes[i],
                x=rest1_overall_prev,
                y=rest2_overall_prev,
                color="steelblue",
                label=r"$II$",
                scatter=True,
            )
            sns.regplot(
                ax=axes[i],
                x=rest1_overall,
                y=rest2_overall,
                color="lightcoral",
                label=r"$TC$",
                scatter=True,
            )
            axes[i].set_title(
                f"$II$: R = {r_prev:.2f} / $p$-value$_{{Bonferroni-corrected}}$ = {p_prev * 2:.2g}\n$TC$: R = {r:.2f} / $p$-value$_{{Bonferroni-corrected}}$ = {p * 2:.2g}",
                fontsize=18,
            )
            axes[i].set_xlabel(xlabels[i], fontsize=18)
            axes[i].set_ylabel(ylabels[i], fontsize=18)
            axes[i].legend(fontsize=14)
            axes[i].tick_params(axis="both", which="major", labelsize=14)
            i += 1

    if fig_dir != None:
        plt.savefig(fig_dir)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return


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
        nrows=2, ncols=2, figsize=figsize, gridspec_kw={"hspace": 0.5}
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
        )
        F_mann_rest1, p_mann_rest1 = stats.mannwhitneyu(
            df_male_rest1_metric[metric], df_female_rest1_metric[metric]
        )
        F_mann_rest2, p_mann_rest2 = stats.mannwhitneyu(
            df_male_rest2_metric[metric], df_female_rest2_metric[metric]
        )
        axes.set_title(
            f"REST 1: $p$-value$_{{Bonferroni-corrected}}$ = {p_mann_rest1 * 4:.2g}\nREST 2: $p$-value$_{{Bonferroni-corrected}}$ = {p_mann_rest2 * 4:.2g}",
            fontsize=20,
        )
        axes.set_ylabel(ylabels[idx], fontsize=18)
        axes.set_xlabel("rs-fMRI recording (REST)", fontsize=18)
        axes.legend(title="Sex", fontsize=12, title_fontsize=14)
        axes.tick_params(axis="both", which="major", labelsize=14)

    if fig_dir != None:
        fig.savefig(fig_dir)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return


def _grid_communities(communities):
    """
    Generate boundaries of `communities`.
    Function based on the Netneurotools package (https://github.com/netneurolab/netneurotools.git).

    Parameters
    ----------
    communities : array_like
        Community assignment vector

    Returns
    -------
    bounds : list
        Boundaries of communities
    """
    communities = np.asarray(communities)
    if 0 in communities:
        communities = communities + 1

    comm = communities[np.argsort(communities)]
    bounds = []
    for i in np.unique(comm):
        ind = np.where(comm == i)
        if len(ind) > 0:
            bounds.append(np.min(ind))

    bounds.append(len(communities))

    return bounds


def _sort_communities(consensus, communities):
    """
    Sort `communities` in `consensus` according to strength.
    Function based on the Netneurotools package (https://github.com/netneurolab/netneurotools.git).

    Parameters
    ----------
    consensus : array_like
        Correlation matrix
    communities : array_like
        Community assignments for `consensus`

    Returns
    -------
    inds : np.ndarray
        Index array for sorting `consensus`
    """
    communities = np.asarray(communities)
    if 0 in communities:
        communities = communities + 1

    bounds = _grid_communities(communities)
    inds = np.argsort(communities)

    for n, f in enumerate(bounds[:-1]):
        i = inds[f : bounds[n + 1]]
        cco = i[consensus[np.ix_(i, i)].mean(axis=1).argsort()[::-1]]
        inds[f : bounds[n + 1]] = cco

    return inds


def plot_heatmap_communities(
    data,
    communities,
    *,
    inds=None,
    edgecolor="black",
    ax=None,
    title=None,
    figsize=(6.4, 4.8),
    xlabels=None,
    ylabels=None,
    xlabelrotation=90,
    ylabelrotation=0,
    cbar=True,
    cmap="viridis",
    square=True,
    xticklabels=None,
    yticklabels=None,
    mask_diagonal=True,
    **kwargs,
):
    """
    Plot `data` as heatmap with borders drawn around `communities`.
    Function based on the Netneurotools package (https://github.com/netneurolab/netneurotools.git)

    Parameters
    ----------
    data : (N, N) array_like
        Correlation matrix
    communities : (N,) array_like
        Community assignments for `data`
    inds : (N,) array_like, optional
        Index array for sorting `data` within `communities`. If None, these
        will be generated from `data`. Default: None
    edgecolor : str, optional
        Color for lines demarcating community boundaries. Default: 'black'
    ax : matplotlib.axes.Axes, optional
        Axis on which to plot the heatmap. If none provided, a new figure and
        axis will be created. Default: None
    figsize : tuple, optional
        Size of figure to create if `ax` is not provided. Default: (20, 20)
    {x,y}labels : list, optional
        List of labels on {x,y}-axis for each community in `communities`. The
        number of labels should match the number of unique communities.
        Default: None
    {x,y}labelrotation : float, optional
        Angle of the rotation of the labels. Available only if `{x,y}labels`
        provided. Default : xlabelrotation: 90, ylabelrotation: 0
    square : bool, optional
        Setting the matrix with equal aspect. Default: True
    {x,y}ticklabels : list, optional
        Incompatible with `{x,y}labels`. List of labels for each entry (not
        community) in `data`. Default: None
    cbar : bool, optional
        Whether to plot colorbar. Default: True
    mask_diagonal : bool, optional
        Whether to mask the diagonal in the plotted heatmap. Default: True
    kwargs : key-value mapping
        Keyword arguments for `plt.pcolormesh()`

    Returns
    -------
    ax : matplotlib.axes.Axes
        Axis object containing plot
    """
    for t, label in zip([xticklabels, yticklabels], [xlabels, ylabels]):
        if t is not None and label is not None:
            raise ValueError("Cannot set both {x,y}labels and {x,y}ticklabels")

    # get indices for sorting consensus
    if inds is None:
        inds = _sort_communities(data, communities)

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)

    # plot data re-ordered based on community and node strength
    if mask_diagonal:
        plot_data = np.ma.masked_where(np.eye(len(data)), data[np.ix_(inds, inds)])
        np.fill_diagonal(plot_data, np.nan)
    else:
        plot_data = data[np.ix_(inds, inds)]

    sns.heatmap(
        plot_data,
        ax=ax,
        cbar=cbar,
        square=square,
        mask=None,
        cmap=cmap,
        **kwargs,
    )
    column_bar = ax.collections[0].colorbar
    column_bar.ax.tick_params(labelsize=30)

    ax.set(xlim=(0, plot_data.shape[1]), ylim=(0, plot_data.shape[0]))
    ax.set_xlabel("ROI's cluster label$_{Subnetwork}$", fontsize=32)
    ax.set_ylabel("ROI's cluster label$_{Subnetwork}$", fontsize=32)

    if title is not None:
        ax.set_title(title, fontsize=34)

    for side in ["top", "right", "left", "bottom"]:
        ax.spines[side].set_visible(False)

    # invert the y-axis so it looks "as expected"
    ax.invert_yaxis()

    # draw borders around communities
    bounds = _grid_communities(communities)
    bounds[0] += 0.1
    bounds[-1] -= 0.1
    for n, edge in enumerate(np.diff(bounds)):
        ax.add_patch(
            mpatches.Rectangle(
                (bounds[n], bounds[n]),
                edge,
                edge,
                fill=False,
                linewidth=4,
                edgecolor=edgecolor,
            )
        )

    if xlabels is not None or ylabels is not None:
        # find the tick locations
        initloc = _grid_communities(communities)
        tickloc = []
        for loc in range(len(initloc) - 1):
            tickloc.append(np.mean((initloc[loc], initloc[loc + 1])))

        if xlabels is not None:
            # make sure number of labels match the number of ticks
            if len(tickloc) != len(xlabels):
                raise ValueError(
                    "Number of labels do not match the number of unique communities."
                )
            else:
                ax.set_xticks(tickloc)
                ax.set_xticklabels(labels=xlabels, rotation=xlabelrotation)
                ax.tick_params(left=False, bottom=False)
        if ylabels is not None:
            # make sure number of labels match the number of ticks
            if len(tickloc) != len(ylabels):
                raise ValueError(
                    "Number of labels do not match the number of unique communities."
                )
            else:
                ax.set_yticks(tickloc)
                ax.set_yticklabels(labels=ylabels, rotation=ylabelrotation)
                ax.tick_params(left=False, bottom=False)

    if xticklabels is not None:
        labels_ind = [xticklabels[i] for i in inds]
        ax.set_xticks(np.arange(len(labels_ind)) + 0.5)
        ax.set_xticklabels(labels_ind, rotation=90)
    if yticklabels is not None:
        labels_ind = [yticklabels[i] for i in inds]
        ax.set_yticks(np.arange(len(labels_ind)) + 0.5)
        ax.set_yticklabels(labels_ind, rotation=0)

    return ax


def generate_binarized_adjacency_matrices(
    matrix,
    density,
    subnet_labels,
    atlas_cluster_labels,
    sc_cluster_labels,
    danmf_cluster_labels,
    matrix_name="",
    fig_name=None,
    plot_show=False,
):
    """Compute modularity of a network structured by the 'matrix' with specific 'density' according to the cluster labels from the
    spectral clustering model 'sc_cluster_labels' and DANMF clustering model 'danmf_cluster_labels'.

    Parameters
    ----------

    matrix: numpy array matrix

    density: float

    subnet_labels: list of string

    atlas_cluster_labels: list of integer

    sc_cluster_labels: list of integer

    danmf_cluster_labels: list of integer

    matrix_name: string

    """

    matrix_bin = G_den(matrix, density)
    matrix_bin = nx.to_numpy_array(matrix_bin)
    matrix_bin[matrix_bin > 0] = 1
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 45))

    title = f"Adjacency matrix of {matrix_name} - Density = {int(density * 100)}\%\nReordered based on the Schaefer's atlas"
    node_labels = atlas_cluster_labels.astype(str) + "$_{" + subnet_labels + "}$"
    ax[0] = plot_heatmap_communities(
        matrix_bin,
        atlas_cluster_labels,
        xticklabels=node_labels.to_list(),
        yticklabels=node_labels.to_list(),
        figsize=(35, 35),
        cmap="viridis",
        ax=ax[0],
        title=title,
    )

    mod_sc = get_modularity(matrix_bin, sc_cluster_labels)
    title = f"Adjacency matrix of {matrix_name} - Density = {int(density * 100)}\%\nClustered and reordered based on the spectral clustering outcome\nModularity$\:={mod_sc:.3f}$"
    node_labels = sc_cluster_labels.astype(str) + "$_{" + subnet_labels + "}$"
    plot_heatmap_communities(
        matrix_bin,
        sc_cluster_labels,
        xticklabels=node_labels.to_list(),
        yticklabels=node_labels.to_list(),
        figsize=(35, 35),
        cmap="viridis",
        ax=ax[1],
        title=title,
    )

    mod_danmf = get_modularity(matrix_bin, danmf_cluster_labels)
    title = f"Adjacency matrix of {matrix_name} - Density = {int(density * 100)}\%\nClustered and reordered based on the DANMF-based clustering outcome\nModularity$\:={mod_danmf:.3f}$"
    node_labels = danmf_cluster_labels.astype(str) + "$_{" + subnet_labels + "}$"
    plot_heatmap_communities(
        matrix_bin,
        danmf_cluster_labels,
        xticklabels=node_labels.to_list(),
        yticklabels=node_labels.to_list(),
        figsize=(35, 35),
        cmap="viridis",
        ax=ax[2],
        title=title,
    )
    plt.tight_layout()
    if fig_name != None:
        plt.savefig(fig_name)
    if plot_show:
        plt.show()
    else:
        plt.close()
    return mod_sc, mod_danmf


def plot_edge_weights_and_node_degrees_distributions(
    list_matrices, titles, figsize=(15, 10), fig_dir=None, plot_show=True
):
    """Plot the distribution of the edge weights and node degrees for a given list of adjacency matrices

    Parameters
    ----------
    list_matrices: list of numpy array matrices
    titles: list of strings
    figsize: tuple
    fig_dir: string
    """
    num_matrices = len(list_matrices)
    ncols = 2
    nrows = num_matrices
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for i, matrix in enumerate(list_matrices):
        # Plot edge weight distribution
        sns.histplot(
            lower_diagonal_exclusive(matrix),
            bins=50,
            color="blue",
            ax=axes[i, 0],
            kde=True,
        )
        axes[i, 0].set_title(f"Edge weight distribution of {titles[i]}", fontsize=20)
        axes[i, 0].set_xlabel("Edge Weight", fontsize=16)
        axes[i, 0].set_ylabel("Frequency", fontsize=16)
        axes[i, 0].tick_params(axis="both", which="major", labelsize=14)

        # Plot node degree distribution
        sns.histplot(
            np.sum(matrix, axis=1), bins=50, color="red", ax=axes[i, 1], kde=True
        )
        axes[i, 1].set_title(f"Node degree distribution of {titles[i]}", fontsize=20)
        axes[i, 1].set_xlabel("Node Degree", fontsize=16)
        axes[i, 1].set_ylabel("Frequency", fontsize=16)
        axes[i, 1].tick_params(axis="both", which="major", labelsize=14)

    plt.tight_layout()
    if fig_dir is not None:
        plt.savefig(fig_dir)

    if plot_show:
        plt.show()
    else:
        plt.close()

    return


def plot_heatmaps(list_matrix, titles, ax=None, figsize=(20, 15), cmap="viridis"):
    """
    Plot heatmaps of the adjacency matrices.

    Parameters:
    - list_matrix: list of numpy.ndarray
        List of adjacency matrices.
    - titles: list of str
        List of titles for each heatmap.
    - figsize: tuple of int, optional
        Size of the figure. Default is (20, 20).
    - cmap: str, optional
        Colormap to use for the heatmaps. Default is 'viridis'.
    - plot_show: bool, optional
        If True, the plot is shown. Otherwise, the plot is saved. Default is True.
    """
    n = len(list_matrix)

    if ax is None:
        fig, ax = plt.subplots(nrows=n, ncols=1, figsize=figsize)

    if n == 1:
        ax = [ax]

    for i, (matrix, title) in enumerate(zip(list_matrix, titles)):
        m = matrix.copy()
        np.fill_diagonal(m, np.nan)
        sns.heatmap(m, ax=ax[i], cmap=cmap)
        ax[i].set_title(title, fontsize=44)
        ax[i].set_xlabel("ROI", fontsize=38)
        ax[i].set_ylabel("ROI", fontsize=38)
        cbar = ax[i].collections[0].colorbar
        cbar.ax.tick_params(labelsize=34)
        ax[i].tick_params(axis="both", which="major", labelsize=12)

    return ax
