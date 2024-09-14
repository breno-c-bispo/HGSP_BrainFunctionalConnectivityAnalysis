# Basic data manipulation and visualisation libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['text.usetex'] = True
plt.rcParams["font.family"] = "Times"

# Scikit-learn libraries
from sklearn.metrics import silhouette_samples

from itertools import combinations
from scipy import stats

from math import ceil

from utils import remove_outliers_zscore, get_binarized_matrix, get_modularity


def plot_multiple_stem(list_df, xlabels, ylabels, ncols, xticks=None, yticks=None, xlims=None, ylims=None, markersize=2, figsize=(15, 5), fig_name=None):
    """Plot multiple charts from 'list_df'.

        Parameters
        ----------

        list_df: Pandas Dataframe

        xlabels: list of strings

        ylabels: list of strings

        ncols: integer

        fig_name: string
    """

    num_subplots = len(list_df)
    nrows = (num_subplots + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)

    for n, ax in enumerate(axes.flat):
        x = list_df[n]['x']
        y = list_df[n]['y']

        markerline, stemline, baseline = ax.stem(x, y,linefmt='k--',markerfmt='o')

        plt.setp(stemline, linewidth = 1.0)
        plt.setp(markerline, markersize = markersize)
        
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

def plot_models_silhouette_diagrams(list_kclusters, list_embeddings, list_kmeans_models, nodes_color=[], name_dataset='', fig_size=None, fig_dir=None):
    """Plot the silhoueete diagrams of K-Means models 'list_kmeans_models'

        Paramters
        ---------

        list_kclusters: list of integer
                    list of K clusters for which the silhouette diagram is to be plotted
        
        list_embeddings: list of numpy array matrix
                    list of the corresponding embedding

        list_kmeans_models: list of K-Means objects

        nodes_color: list of string
                    list of nodes color
    
    """
    
    dict_node_labels = {}
    dict_embeddings = {}
    for idx, kmeans_model in enumerate(list_kmeans_models):
        n_clusters =  len(set(kmeans_model.labels_))
        dict_node_labels[n_clusters] = kmeans_model.labels_
        dict_embeddings[n_clusters] = list_embeddings[idx]
    num_subplots = len(list_kclusters)
    ncols = 2
    nrows = (num_subplots + 1) // 2
    fig_width = 12
    fig_height = 4 * num_subplots
    if fig_size != None:
        fig_width, fig_height = fig_size[0], fig_size[1]
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), gridspec_kw={'hspace': 0.25})
        
    percent = {}
    for i, ax in enumerate(axes.flat):
        if i < num_subplots:
            y_pred = dict_node_labels.get(list_kclusters[i])
            embedding = dict_embeddings.get(list_kclusters[i])
            silhouette_coefficients = silhouette_samples(embedding, y_pred)
            silhouette_avg = silhouette_coefficients.mean()
            df_sc = pd.DataFrame({'sc': silhouette_coefficients, 'cluster': y_pred})
            if len(nodes_color) == len(y_pred):
                df_sc['color'] = nodes_color
            else:
                df_sc['color'] = [px.colors.qualitative.Light24[idx] for idx in y_pred]
                
            y_lower_global = 0
            for c in range(list_kclusters[i]):
                cth_cluster_silhouette_values = df_sc.loc[df_sc['cluster'] == c].sort_values(by='sc')
                size_cluster_c = len(cth_cluster_silhouette_values)
                y_lower_local = y_lower_global
                for j in range(size_cluster_c):
                    y_upper_local = y_lower_local + 20
                    ax.fill_betweenx(
                        np.arange(y_lower_local, y_upper_local),
                        0,
                        cth_cluster_silhouette_values.iloc[j]['sc'],
                        facecolor=cth_cluster_silhouette_values.iloc[j]['color'],
                        alpha=0.7,
                    )
                    y_lower_local = y_upper_local + 2
                
                # Label the silhouette plots with their cluster numbers at the middle
                ax.text(-0.05, y_lower_global + 0.5 * size_cluster_c*20, str(c), fontsize=14)

                # Compute the new y_lower for next plot
                y_lower_global = y_upper_local + 100  # 10 for the 0 samples

            percent[list_kclusters[i]] = 100*len(df_sc[df_sc['sc'] > silhouette_avg])/len(df_sc)
            
            ax.set_title(f"Silhouette diagrams {name_dataset}, $K={list_kclusters[i]}$, $SC={silhouette_avg:.3f}$", fontsize=14)
            ax.set_xlabel(f"Silhouette coefficients $\eta$", fontsize=14)
            ax.set_ylabel(r"Cluster labels", fontsize=14)
            # The vertical line for average silhouette score of all the values
            ax.axvline(x=silhouette_avg, color="black", linestyle="--")
            ax.set_yticks([])  # Clear the yaxis labels / ticks
            ax.set_xticks(np.arange(-0.1,0.9,0.1))

    if num_subplots % 2 != 0:
        if num_subplots > 1:
            axes[-1, -1].axis('off')
        else:
            axes[-1].axis('off')
        
    plt.tight_layout()
    if fig_dir != None:
        plt.savefig(fig_dir)
    plt.show()

    return

def plot_pie_clusters(models_list, k_cluster, nodes_color, nodes_subnet, fig_title='', file_dir=None):
    """Plot a pie chart of the nodes distribution among the clusters

        Parameters
        ----------

        models_list: list of K-Means objects

        k_cluster: integer
                model with K cluster to be plotted
        
        nodes_color: list of string
                    list of nodes color
        
        nodes_subnet: list os string
                    list of subnets where each node belongs
    """

    for idx in range(len(models_list)):
        if len(set(models_list[idx].labels_)) == k_cluster:
            break

    nodes_cluster = models_list[idx].labels_
    df_brain = pd.DataFrame({'cluster':nodes_cluster, 'color': nodes_color, 'subnet':nodes_subnet})

    cluster_list = sorted(set(nodes_cluster))
    df = pd.DataFrame(0, index=cluster_list, columns=df_brain['subnet'].drop_duplicates().values.tolist())
    for cluster_name in list(df.columns):
        for cluster_label in df.index:
            df.loc[cluster_label, cluster_name] = len(df_brain[(df_brain['subnet'] == cluster_name) & (df_brain['cluster'] == cluster_label)])

    num_subplots = len(cluster_list)
    ncols = 3
    nrows = (num_subplots + 2) // 3

    fig = make_subplots(rows=nrows, cols=ncols, specs=np.full((nrows, ncols), {'type':'domain'}).tolist())
    row = 1
    col = 1
    nodes_distribution = np.sum(df.to_numpy(), axis=0)
    for i in df.index:
        cluster_node_info = df.loc[i]
        overall_percentual = 100 * cluster_node_info.values / nodes_distribution
        percentual = [f'({p:.0f}%)' for idx, p in enumerate(overall_percentual)]
        fig.add_trace(go.Pie(labels=list(cluster_node_info.index), 
                            values=cluster_node_info.values, 
                            name=f"Cluster {i}",
                            textinfo='percent+text',
                            text=percentual),
                            row, col)
        col += 1
        if col > ncols:
            row += 1
            col = 1
    # Use `hole` to create a donut-like pie chart
    fig.update_traces(textfont_color='black', textfont_size=18,
                      textposition='inside',
                        hole=.4, hoverinfo="label+percent+name", texttemplate='%{percent:.2p}<br>%{text}',
                        marker=dict(
                            colors=df_brain['color'].drop_duplicates().values.tolist(),
                            line=dict(color='black', width=1))
                        )
    
    fig.update_layout(
        title_text=fig_title, autosize=False, width=800,
        font_family='Times New Roman',
        height=800,
        uniformtext_minsize=16, uniformtext_mode=None,
        legend=dict(font=dict(size=16), title_text='Brain parcels'),
        margin=dict(l=0, r=0, b=0, t=50, pad=0))
    
    if file_dir != None:
        fig.write_image(file_dir)
    fig.show()
    return

def plot_mean_tubal_scalars(As, k_indexes, title='', step_xticks=2, figsize=(10,4), fig_dir=None):
    """Plot the 'k_indexes' of the mean absolute value of tubal scalars of the 3-order tensor 'As'

        Parameters
        ----------

        As: numpy array tensor

        k_indexes: list of integer

        title: string
    """

    N0,N1,N2 = As.shape
    l = list(combinations(range(N0), 2))
    tubal_scalars_list = []
    for c in l:
        tubal_scalars_list.append(np.squeeze(As[c[0],c[1],:ceil(N2/2)+1]))
    tubal_scalars_list = np.abs(np.array(tubal_scalars_list))
    mean_tubal_scalars = np.mean(tubal_scalars_list,axis=0)
    df = pd.DataFrame({'k':range(0,len(mean_tubal_scalars)), 'mean':mean_tubal_scalars})

    df = df.iloc[k_indexes]

    fig, ax = plt.subplots(nrows=1,ncols=1, figsize=figsize)
    df.plot( ax=ax,x='k', y='mean', c='brown', fontsize=16)
    ax.set_xlabel(r'$k$', fontsize=16)
    ax.set_title(title, fontsize=18)
    ax.set_xticks(range(0,len(df),step_xticks))
    if fig_dir != None:
        plt.savefig(fig_dir)
    plt.show()
    return

def plot_correlation_scatters(df, z_score_thr=2, figsize=(15,5), fig_dir=None):
    """Plot of the correlation scatters of the peak peak values between males and females
        in each resting-state scan at k=0,4 for II and TC metrics.

        Parameters
        ----------

        df: Pandas Dataframe

        z_score_thr: integer
    """

    ncols = 2
    nrows = 1
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, gridspec_kw={'hspace': 0.5})

    list_metrics = ['peak_ii_0', 'peak_tc_0', 'peak_ii_4', 'peak_tc_4']
    list_xlabel = ['Peak value at $k=0$ for the 1$\\textsuperscript{o}$ rs-scan', 'Peak value at $k=4$ for the 1$\\textsuperscript{o}$ rs-scan']
    list_ylabel = ['Peak value at $k=0$ for the 2$\\textsuperscript{o}$ rs-scan', 'Peak value at $k=4$ for the 2$\\textsuperscript{o}$ rs-scan']
    i = 0
    for idx, metric in enumerate(list_metrics):
        metric_rest1 = metric + '_REST1'
        metric_rest2 = metric + '_REST2'
        df_overall_rest1 = df.loc[df['REST'] == 1][['Subject', metric]]
        df_overall_rest1 = remove_outliers_zscore(df_overall_rest1, metric, z_score_thr)
        df_overall_rest1 = df_overall_rest1.rename(columns={metric: metric_rest1})

        df_overall_rest2 = df.loc[df['REST'] == 2][['Subject', metric]]
        df_overall_rest2 = remove_outliers_zscore(df_overall_rest2, metric, z_score_thr)
        df_overall_rest2 = df_overall_rest2.rename(columns={metric: metric_rest2})

        df_overall = df_overall_rest2.merge(df_overall_rest1, how='inner')
        df_overall = df_overall.dropna()
        rest1_overall = np.array(df_overall[metric_rest1].to_list()).flatten()
        rest2_overall = np.array(df_overall[metric_rest2].to_list()).flatten()
        r, p = stats.pearsonr(rest1_overall, rest2_overall)

        if (idx % 2 == 0):
            r_prev, p_prev = r, p
            df_overall_prev = df_overall.copy()
            metric_rest1_prev, metric_rest2_prev = metric_rest1, metric_rest2
        else:
            rest1_overall_prev = np.array(df_overall_prev[metric_rest1_prev].to_list()).flatten()
            rest2_overall_prev = np.array(df_overall_prev[metric_rest2_prev].to_list()).flatten()
           
            sns.regplot(ax=axes[i],x=rest1_overall_prev, y=rest2_overall_prev, color='steelblue', label=r'$II$', scatter=True)
            sns.regplot(ax=axes[i],x=rest1_overall, y=rest2_overall, color='lightcoral', label=r'$TC$', scatter=True)
            axes[i].set_title(f'$II$: R = {r_prev:.2f} / $p$-value$_{{Bonferroni-corrected}}$ = {p_prev*2:.2g}\n$TC$: R = {r:.2f} / $p$-value$_{{Bonferroni-corrected}}$ = {p*2:.2g}', fontsize=14)
            axes[i].set_xlabel(list_xlabel[i], fontsize=14)
            axes[i].set_ylabel(list_ylabel[i], fontsize=14)
            axes[i].legend(fontsize=12)
            axes[i].tick_params(axis='both', which='major', labelsize=14)
            i += 1

    if fig_dir != None:
        plt.savefig(fig_dir)
    plt.show()
    return

def plot_metrics_distribution(df, zscore_thr=2, figsize=(15,6), fig_dir=None):
    """Plot the distribution of the peak values between males and females in each resting-state scan at k=0,4 for II and TC metrics

        Parameters
        ----------

        df: Pandas Dataframe

        z_score_thr: integer
    """

    fig, ax = plt.subplots(nrows=2,ncols=2, figsize=figsize, gridspec_kw={'hspace': 0.5})
    list_metrics = ['peak_ii_0', 'peak_tc_0', 'peak_ii_4', 'peak_tc_4']
    list_ylabel = [r'Peak value at $k=0$ for $II$',
               r'Peak value at $k=0$ for $TC$',
               r'Peak value at $k=4$ for $II$',
               r'Peak value at $k=4$ for $TC$']
    
    for idx, axes in enumerate(ax.flat):
        metric = list_metrics[idx]
        df_male_rest1_metric = df.loc[(df['Gender'] == 'M') & (df['REST'] == 1)][['Subject', 'REST', 'Gender', metric]]
        df_female_rest1_metric = df.loc[(df['Gender'] == 'F') & (df['REST'] == 1)][['Subject', 'REST', 'Gender', metric]]
        df_male_rest2_metric = df.loc[(df['Gender'] == 'M') & (df['REST'] == 2)][['Subject', 'REST', 'Gender', metric]]
        df_female_rest2_metric = df.loc[(df['Gender'] == 'F') & (df['REST'] == 2)][['Subject', 'REST', 'Gender', metric]]

        df_male_rest1_metric = remove_outliers_zscore(df_male_rest1_metric, metric, zscore_thr)
        df_female_rest1_metric = remove_outliers_zscore(df_female_rest1_metric, metric, zscore_thr)
        df_male_rest2_metric = remove_outliers_zscore(df_male_rest2_metric, metric, zscore_thr)
        df_female_rest2_metric = remove_outliers_zscore(df_female_rest2_metric, metric, zscore_thr)

        
        df_rest = pd.concat([df_male_rest1_metric, df_female_rest1_metric, df_male_rest2_metric, df_female_rest2_metric], axis=0)
        sns.violinplot(x="REST", y=metric, data=df_rest, hue="Gender", split=True, inner="quart", ax=axes)
        F_mann_rest1, p_mann_rest1 = stats.mannwhitneyu(df_male_rest1_metric[metric], df_female_rest1_metric[metric])
        F_mann_rest2, p_mann_rest2 = stats.mannwhitneyu(df_male_rest2_metric[metric], df_female_rest2_metric[metric])
        axes.set_title(f'1$\\textsuperscript{{o}}$ rs-scan: $p$-value$_{{Bonferroni-corrected}}$ = {p_mann_rest1*4:.2g}\n2$\\textsuperscript{{o}}$ rs-scan: $p$-value$_{{Bonferroni-corrected}}$ = {p_mann_rest2*4:.2g}', fontsize=16)
        axes.set_ylabel(list_ylabel[idx],fontsize=16)
        axes.set_xlabel('Resting-state scan',fontsize=16)
        axes.legend(title='Sex', fontsize=12, title_fontsize=12)
        axes.tick_params(axis='both', which='major', labelsize=12)
    
    if fig_dir != None:
        fig.savefig(fig_dir)
    plt.show()
    return    

def get_adjacency_matrix_heatmap(A, node_labels, title, axes):
    """ Return a axes with the heatmap of the matrix A

        Paramters
        ---------

        A: numpy array matrix

        node_labels: list of strings

        title: string

        axes: matplotlib Axes object
    
    """
    
    matrix = A.copy()
    np.fill_diagonal(matrix,np.nan)
    Pdmatrix = pd.DataFrame(matrix)
    Pdmatrix.columns = node_labels
    Pdmatrix.index = node_labels
    Pdmatrix = Pdmatrix.sort_index(0).sort_index(1)
    sns.heatmap(Pdmatrix, ax=axes, cbar=True, square=False, mask=None,xticklabels=True, yticklabels=True) 
    cbar = axes.collections[0].colorbar
    cbar.ax.tick_params(labelsize=18)
    axes.set_title(title, size=22)
    return axes

def generate_binarized_adjacency_matrices(matrix, density, subnet_labels, atlas_cluster_labels, sc_cluster_labels, danmf_cluster_labels, matrix_name='', fig_name=None, plot_show=False):
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
        
    matrix_bin = get_binarized_matrix(matrix, density)
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20,60))

    title = f'Adjacency matrix of {matrix_name} - Density = {int(density*100)}\%\nReordered based on the Schaefer\'s atlas'
    node_labels = atlas_cluster_labels.astype(str) + '$_{' + subnet_labels + '}$'
    ax[0] = get_adjacency_matrix_heatmap(matrix_bin, node_labels.to_list(), title, ax[0])

    mod_sc = get_modularity(matrix_bin, sc_cluster_labels)
    title = f'Adjacency matrix of {matrix_name} - Density = {int(density*100)}\%\nClustered and reordered based on the Spectral clustering outcome\nModularity$\:={mod_sc:.3f}$'
    node_labels = sc_cluster_labels.astype(str) + '$_{' + subnet_labels + '}$'
    ax[1] = get_adjacency_matrix_heatmap(matrix_bin, node_labels.to_list(), title, ax[1])

    mod_danmf = get_modularity(matrix_bin, danmf_cluster_labels)
    title = f'Adjacency matrix of {matrix_name} - Density = {int(density*100)}\%\nClustered and reordered based on the DANMF-based clustering outcome\nModularity$\:={mod_danmf:.3f}$'
    node_labels = danmf_cluster_labels.astype(str) + '$_{' + subnet_labels + '}$'
    ax[2] = get_adjacency_matrix_heatmap(matrix_bin, node_labels.to_list(), title, ax[2])

    if fig_name != None:
        plt.savefig(fig_name)
    if plot_show:
        plt.show()
    return mod_sc, mod_danmf