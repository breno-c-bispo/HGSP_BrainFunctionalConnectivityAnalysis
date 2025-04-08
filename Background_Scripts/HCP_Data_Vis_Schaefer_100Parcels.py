#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Costumized code from the https://github.com/multinetlab-amsterdam/network_TDA_tutorial.git
for 3D brain visualization
"""

__author__ = "Fernando Nobrega & Breno Bispo"
__contact__ = "f.a.nobregasantos@uva.nl & breno.bispo@ufpe.br"
__date__ = "2023/09/12"
__status__ = "Production"


####################
# Review History   #
####################

# Reviewed and Updated by Eduarda Centeno 20201103


####################
# Libraries        #
####################

# Third party imports
import networkx as nx  # version 2.4
import numpy as np  # version 1.18.5
import pandas as pd  # version 1.1.3
import plotly.express as px  # version 4.6.0
import plotly.graph_objs as go  # version 4.6.0
from plotly.offline import init_notebook_mode, iplot
from trimesh import load_mesh
from sklearn.preprocessing import minmax_scale


########################
# Pre-defined settings #
########################

init_notebook_mode(connected=True)  # Define Notebook Mode

# Pre-defined paths and constants
path_areas = "./Schaefer_100Parcels_Atlas/Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_region_names_full.txt"
path_pos = "./Schaefer_100Parcels_Atlas/Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_positions.txt"
path_subnet_names = "./Schaefer_100Parcels_Atlas/Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_subnet_order_names.txt"
path_subnet_colors = "./Schaefer_100Parcels_Atlas/Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_subnet_order_colors.txt"
path_subnet_labels = "./Schaefer_100Parcels_Atlas/Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_subnet_colors_number.txt"
path_brainobj = "./Figures/brain.obj"

schaefer_nodes_label = np.genfromtxt(path_subnet_labels).astype(int) - 1
schaefer_nodes_color = open(path_subnet_colors, "r").read().split("\n")
schaefer_nodes_subnet = open(path_subnet_names, "r").read().split("\n")
schaefer_coordinates = np.loadtxt(path_pos)
schaefer_areas_full = open(path_areas, "r").read().split("\n")


# Define Functions ------------------------------------------------------------


def matplotlib_to_plotly(cmap, pl_entries):
    """Create matplotlib color scales for plotly

    Parameters
    ----------
    cmap : colormap
        A colormap in matplotly  - Ex: jet_cmap
    pl_entries: list
        Number of entries

    Returns
    -------
    pl_colorsacle: list
        A color scale from matplotlib that is readble in ploty

    """

    h = 1.0 / (pl_entries - 1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k * h)[:3]) * 255))
        pl_colorscale.append([float(k * h), "rgb" + str((C[0], C[1], C[2]))])

    return pl_colorscale


def openatlas(path_pos=path_pos):
    """Open an atlas file with its coordinates

    Parameters
    ----------
    path_pos: string
        Path to the file with atlas coordinates

    Returns
    -------
    data: list
        A list of coordinates

    """

    positions = pd.read_csv(path_pos, header=None, delim_whitespace=True)

    data = [list(row.values) for _, row in positions.iterrows()]

    return data


def dictpos(areas, path_pos=path_pos):
    """Creates a dictionary with 3D positions for a given atlas
    This function creates a transparent shell. This is necessary for hoverinfo,
    i.e. you can find the name of the ROI using your mouse.

    Parameters
    ----------
    path_pos: string
        Path to the file with atlas coordinates

    Returns
    -------
    trace: plotly.graph_objs._mesh3d.Mesh3d
        Plotly graphical object
    x: list
        X-axis coordinates
    y: list
        Y-axis coordinates
    z: list
        Z-axis coordinates

    """

    data = openatlas(path_pos)
    x = []
    y = []
    z = []
    pos3d = {}
    for i in range(0, len(data)):
        pos3d[i] = (data[i][0], data[i][1], data[i][2])
        x.append(data[i][0])
        y.append(data[i][1])
        z.append(data[i][2])

    xs = []
    ys = []
    zs = []
    for i in range(0, len(data)):
        pos3d[i] = (data[i][0], data[i][1], data[i][2])
        xs.append(1.01 * data[i][0])
        ys.append(1.01 * data[i][1])
        zs.append(1.01 * data[i][2])

    trace1 = go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        alphahull=4.2,
        opacity=0.0005,
        color="gray",
        text=areas,
        hoverinfo="text",
    )

    return trace1, x, y, z


# uncover=False
def shell_brain(brain_mesh):
    """Returns a brain gray shell from a fixed brain.obj file

    Parameters
    ----------
    brain_mesh: meshio mesh object

    Returns
    -------
    mesh: plotly.graph_objs._mesh3d.Mesh3d

    """

    vertices = brain_mesh.vertices
    triangles = brain_mesh.faces
    x, y, z = vertices.T
    I, J, K = triangles.T
    # showlegend=True gives the option to uncover the shell
    mesh = go.Mesh3d(
        x=x,
        y=y,
        z=z,
        color="grey",
        i=I,
        j=J,
        k=K,
        opacity=0.1,
        hoverinfo=None,
        showlegend=True,
        name="Brain 3D",  # colorscale=pl_mygrey, #intensity=z,
        # flatshading=True, #showscale=False
    )

    return mesh  # iplot(fig)


def G_tre(matrix, e):
    """Returns a thresholded networkx Graph from a adjacency matrix

    Parameters
    ----------
    matrix: matrix
        A matrix of values - connectivity matrix

    e: float
        Threshold value for matrix

    Returns
    -------
        NetworkX graph

    Notes
    -------
        Calculation: (1-e) = 0.0

    """

    # "If you want to normalize just uncomment here"
    # ScdatanGA=np.array(normalize(Aut[i]))

    ScdatanGA = matrix
    matcopy = np.copy(np.abs(ScdatanGA))  # be careful to always assing  copy of data,
    # othewise will change the data as well
    matcopy[(np.copy(np.abs(ScdatanGA))) <= (1 - e)] = 0.0
    Gfinal = nx.from_numpy_matrix(matcopy[:, :])

    return Gfinal


def G_den(matrix, d, verbose=False):
    """Returns a networkx Graph from a adjacency matrix, with a given density d

    Parameters
    ----------
    matrix: matrix
        A matrix of values - connectivity matrix

    d: float
        Density value for matrix binaziring

    Returns
    -------
        NetworkX graph

    """

    # matrix i, density d. i is a matrix - ravel flatten the matrix
    np.fill_diagonal(matrix, 0)
    temp = sorted(matrix.ravel(), reverse=True)  # will flatten it and rank corr values
    size = len(matrix)
    cutoff = np.ceil(d * (size * (size - 1)))  # number of links with a given density
    tre = temp[int(cutoff)]
    G0 = nx.from_numpy_matrix(matrix)
    G0.remove_edges_from(list(nx.selfloop_edges(G0)))
    G1 = nx.from_numpy_matrix(matrix)
    for u, v, a in G0.edges(data=True):
        if (a.get("weight")) <= tre:
            G1.remove_edge(u, v)
    finaldensity = nx.density(G1)
    if verbose == True:
        print(finaldensity)

    return G1


def Plot_Brain_Subnets(
    scale=15,
    title="Schaefer's atlas",
    top_view=False,
    printscreen=None,
    movie=None,
    plot_show=True,
):
    """Plot Schaerfer's brain atlas

    Parameters
    ----------

    scale: int
        Scaling factor for node size

    title: string

    top_view: boolean
        If True, it will return a top view of the brain. Otherwise, it will return a side view of the brain.

    printscreen: string
        If it is desired to export a printscreen of the 3D plot, enter the file name.

    movie: string
        If it is desired to export a .HTML interactive 3D visualization of the brain, enter the HTML file name

    plot_show: boolean
        If True, it will show the plot. Otherwise, it will return the plot object.


    """

    data = [brain_trace]

    df = df_schaefer.copy()

    for c in set(list(df["label"])):
        df_temp = df.loc[df["label"] == c]
        df_temp = df_temp.sort_values(
            by="color", key=lambda x: x.map(x.value_counts()), ascending=False
        )
        trace2 = go.Scatter3d(
            x=np.array(df_temp["x"]),
            y=np.array(df_temp["y"]),
            z=np.array(df_temp["z"]),
            mode="markers",
            marker=dict(
                sizemode="diameter",
                symbol="circle",
                showscale=False,
                opacity=1,
                size=scale,
                cauto=True,
                color=df_temp["color"],
                line=dict(width=2, color="DarkSlateGrey"),
            ),
            showlegend=True,
            name=str(df_temp["subnet"].values[0]),
            text=df_temp["region_name"],
            customdata=pd.DataFrame(
                {"node": list(df_temp.index), "subnet": df_temp["subnet"]}
            ),
            hovertemplate="<b>Node: </b>%{customdata[0]}<br>"
            + "<b>Region: </b>%{text}<br>"
            + "<b>Subnet: </b>%{customdata[1]}"
            + "<extra></extra>",
        )
        data.append(trace2)

    # view
    if top_view:
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=1.8),
        )
    else:
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.8, y=0, z=0),
        )

    layout = go.Layout(
        autosize=False,  # To export we need to make False and uncomment the details bellow
        width=800,
        height=800,  # This is to make an centered html file. To export as png, one needs to include this margin
        margin=go.Margin(l=0, r=0, b=0, t=0, pad=4),
        title=title,
        font=dict(size=18, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            camera=camera,
            xaxis=dict(nticks=5, tickfont=dict(size=16, color="black"), visible=False),
            yaxis=dict(nticks=7, tickfont=dict(size=16, color="black"), visible=False),
            zaxis=dict(nticks=7, tickfont=dict(size=16, color="black"), visible=False),
        ),
    )

    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(
        autosize=False, width=800, height=800, margin=dict(l=50, r=50, b=75, t=75)
    )

    fig.update_layout(
        font_family="times", legend_font_size=18, title_x=0.5, legend=dict(x=0)
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
            yaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
            zaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
        )
    )

    if movie != None:
        fig.write_html(movie)
    if printscreen != None:
        fig.write_image(printscreen)

    if plot_show:
        return iplot(fig)
    else:
        return


def Plot_Brain_Graph_Signal(
    graph_signal,
    title="Brain graph signal",
    scale=75,
    node_colors=None,
    size_nodes_const=False,
    top_view=False,
    printscreen=None,
    movie=None,
    plot_show=True,
):
    """Plot a brain graph signal on the Brain 3D plot

    Parameters
    ----------
    graph_signal: list
        A list of values with a node property

    path_pos: stringtr
        Path to the file with atlas coordinates

    title: string

    scale: int
        Scaling factor for node size

    node_colors: list of strings
        A list of strings containing the nodes color

    size_nodes_const: boolean
        If True, the size of the circle on the 3D brain plot are fixed. Otherwise, it'll be modulated according to the corresponding value on the node.

    top_view: boolean
        If True, it will return a top view of the brain. Otherwise, it will return a side view of the brain.

    printscreen: string
        If it is desired to export a printscreen of the 3D plot, enter the file name.

    movie: string
        If it is desired to export a .HTML interactive 3D visualization of the brain, enter the HTML file name

    Returns
    -------
      A 3D plot with customized nodes

    """

    sizec = np.array(graph_signal)
    sizec = np.abs(sizec)
    # restriction in the size
    sizec = 1 / max(sizec) * sizec
    if size_nodes_const:
        sizec = 1.5
    if node_colors != None:
        colorV = node_colors
    else:
        colorV = np.array(graph_signal)
    trace2 = go.Scatter3d(
        x=df_schaefer["x"].to_list(),
        y=df_schaefer["y"].to_list(),
        z=df_schaefer["z"].to_list(),
        mode="markers",
        marker=dict(
            sizemode="diameter",
            symbol="circle",
            showscale=True,
            colorbar=dict(
                title="Values",
                thickness=30,
                x=0.0,  # 0.95,
                len=0.8,
                tickmode="array",
                tick0=0,
                dtick=1,
                nticks=4,
            ),
            opacity=1,
            size=scale * sizec,
            color=colorV,
            colorscale="Jet",
            cauto=True,
            cmin=np.min(colorV),
            cmax=np.max(colorV),
            line=dict(width=2, color="DarkSlateGrey"),
        ),
        showlegend=True,
        name="Brain Graph Signal",
        text=df_schaefer["region_name"],
        customdata=pd.DataFrame(
            {
                "node": list(df_schaefer.index),
                "value": graph_signal,
                "subnet": df_schaefer["subnet"],
            }
        ),
        hovertemplate="<b>Node: </b>%{customdata[0]}<br>"
        + "<b>Node value: </b>%{customdata[1]:.3f}<br>"
        + "<b>Region: </b>%{text}<br>"
        + "<b>Subnet: </b>%{customdata[2]}"
        + "<extra></extra>",
    )
    data = [brain_trace, trace2]
    fig = go.Figure(data=data)
    # view
    if top_view:
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=1.4),
        )
    else:
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.4, y=0, z=0),
        )

    layout = go.Layout(
        autosize=False,
        width=1200,
        height=800,
        margin=dict(l=50, r=50, b=75, t=75, pad=4),
        title=title,
        font_family="Times New Roman",
        font=dict(size=18, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="white",
        scene=dict(
            camera=camera,
            xaxis=dict(nticks=5, tickfont=dict(size=14, color="black"), visible=False),
            yaxis=dict(nticks=7, tickfont=dict(size=14, color="black"), visible=False),
            zaxis=dict(nticks=7, tickfont=dict(size=14, color="black"), visible=False),
        ),
    )

    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
            yaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
            zaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
        )
    )
    if movie != None:
        fig.write_html(movie)
    if printscreen != None:
        fig.write_image(printscreen)

    if plot_show:
        return iplot(fig)
    else:
        return


def Plot_Brain_Clusters(
    models_list,
    k_cluster,
    nodes_color=schaefer_nodes_color,
    scale=15,
    title="Brain Clusters",
    top_view=False,
    printscreen=None,
    movie=None,
    plot_show=True,
):
    """Clusters on the Brain 3D plot

    Parameters
    ----------
    models_list: list of K-Means objects

    k_cluster: integer
            Select the K-Means model containing 'k_cluster' clusters

    node_colors: list of strings
            A list of strings containing the nodes color

    scale: int
        Scaling factor for node size

    title: string

    top_view: boolean
        If True, it will return a top view of the brain. Otherwise, it will return a side view of the brain.

    printscreen: string
        If it is desired to export a printscreen of the 3D plot, enter the file name.

    movie: string
        If it is desired to export a .HTML interactive 3D visualization of the brain, enter the HTML file name

    plot_show: boolean
        If True, it will show the plot. Otherwise, it will return the plot object.

    """

    for idx in range(len(models_list)):
        if len(set(models_list[idx].labels_)) == k_cluster:
            break
    nodes_cluster = models_list[idx].labels_
    if len(set(nodes_cluster)) != k_cluster:
        ValueError(
            f"K-Means clustering model not found containing {k_cluster} clusters!"
        )

    df = df_schaefer.copy()
    df["cluster"] = nodes_cluster
    df["color"] = nodes_color
    df["name"] = ["Cluster {}".format(i) for i in nodes_cluster]

    data = [brain_trace]
    for c in set(list(df["cluster"])):
        df_temp = df.loc[df["cluster"] == c]
        df_temp = df_temp.sort_values(
            by="color", key=lambda x: x.map(x.value_counts()), ascending=False
        )
        trace2 = go.Scatter3d(
            x=np.array(df_temp["x"]),
            y=np.array(df_temp["y"]),
            z=np.array(df_temp["z"]),
            mode="markers",
            marker=dict(
                sizemode="diameter",
                symbol="circle",
                showscale=False,
                opacity=1,
                size=scale,
                cauto=True,
                color=df_temp["color"],
                line=dict(width=2, color="DarkSlateGrey"),
            ),
            showlegend=True,
            name=str(df_temp["name"].values[0]),
            text=df_temp["region_name"],
            customdata=pd.DataFrame(
                {"node": list(df_temp.index), "subnet": df_temp["subnet"]}
            ),
            hovertemplate="<b>Node: </b>%{customdata[0]}<br>"
            + "<b>Region: </b>%{text}<br>"
            + "<b>Subnet: </b>%{customdata[1]}"
            + "<extra></extra>",
        )
        data.append(trace2)

    # view
    if top_view:
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=1.8),
        )
    else:
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.8, y=0, z=0),
        )

    layout = go.Layout(
        autosize=False,  # To export we need to make False and uncomment the details bellow
        width=800,
        height=800,  # This is to make an centered html file. To export as png, one needs to include this margin
        margin=go.Margin(l=0, r=0, b=0, t=0, pad=4),
        title=title,
        font=dict(size=18, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            camera=camera,
            xaxis=dict(nticks=5, tickfont=dict(size=16, color="black"), visible=False),
            yaxis=dict(nticks=7, tickfont=dict(size=16, color="black"), visible=False),
            zaxis=dict(nticks=7, tickfont=dict(size=16, color="black"), visible=False),
        ),
    )

    fig = go.Figure(data=data, layout=layout)

    fig.update_layout(
        autosize=False, width=800, height=800, margin=dict(l=50, r=50, b=75, t=75)
    )

    fig.update_layout(
        font_family="times", legend_font_size=18, title_x=0.5, legend=dict(x=0)
    )

    fig.update_layout(
        scene=dict(
            xaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
            yaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
            zaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
        )
    )

    if movie != None:
        fig.write_html(movie)
    if printscreen != None:
        fig.write_image(printscreen)

    # Code to show the plot
    if plot_show:
        return iplot(fig)
    else:
        return


def nodes_cliques_participation(cliques_list, k, df_atlas, node_feature_range=(1, 75)):
    """This function returns the nodal participation in k-cliques (in percentage)

    Parameters
    ----------
    Graph: NetworkX graph

    cl: int
        Clique size

    scale: int
        Scaling factor for node size

    Returns
    -------
    final: list
        A list with the participation in k-cliques per node (in percentage)

    """
    node_score = [0] * len(df_atlas)
    cliques_list_k = [sublist for sublist in cliques_list if len(sublist) == k]
    for clique in cliques_list_k:
        for node in clique:
            node_score[node] += 1

    if node_feature_range != None:
        node_score = np.array(node_score)
        non_zero_scores = node_score[node_score > 0]
        rescaled_scores = minmax_scale(non_zero_scores, node_feature_range)
        node_score[node_score > 0] = rescaled_scores
    return node_score


def Plot_Brain_Interactions(
    interaction_weights,
    density,
    node_feature_range=(1, 75),
    top_view=False,
    movie=None,
    printscreen=None,
    plot_show=True,
):
    """
    Plots brain interactions in a 3D plot using Plotly.
    Parameters:
    ------------

    interaction_weights (dict): A dictionary where keys are tuples representing cliques and values are the interaction weights.

    density (float): The density of interactions to be plotted, as a fraction of the total number of interactions.

    node_feature_range (tuple, optional): A tuple specifying the range of node feature sizes. Default is (1, 75).

    top_view (bool, optional): If True, sets the camera to a top view. Default is False.

    movie (str, optional): File path to save the plot as an HTML file. Default is None.

    printscreen (str, optional): File path to save the plot as an image. Default is None.

    plot_show (bool, optional): If True, displays the plot. Default is True.

    Returns:
    --------
    None or plotly.graph_objs.Figure: If plot_show is True, returns the plotly figure object for interactive plotting. Otherwise, returns None.

    Raises:
    ValueError: If interaction_weights is not a dictionary or if the clique size is less than or equal to 1.
    Notes:
    - The function assumes that `df_schaefer` is a DataFrame containing the coordinates and other information of brain regions.
    - The function uses Plotly for 3D plotting.
    """

    # Verify if the interaction weights are a dictionary and find the length of of all keys
    if not isinstance(interaction_weights, dict):
        raise ValueError("The interaction weights must be a dictionary.")
    k = len(list(interaction_weights.keys())[0])
    if k == 1:
        raise ValueError("The clique size must be greater than 1.")

    data = [brain_trace]

    # Calculate the number of items to keep
    num_items_to_keep = max(1, int(len(interaction_weights) * density))

    # Sort the dictionary by absolute value of the values in descending order
    sorted_items = sorted(
        interaction_weights.items(), key=lambda item: abs(item[1]), reverse=True
    )

    # Keep only the top 10% items
    interaction_weights = dict(sorted_items[:num_items_to_keep])

    cliques_list = list(interaction_weights.keys())

    # NEED TO MAKE A SINGLE DICTIONARY FROM ALL POLYGONS - THE GO MESH OBJECT.DATA IS A DICTIONARY
    for clique in cliques_list:  # Running over all cliques
        # Create a vector with the positions for each clique
        xk = []
        yk = []
        zk = []
        for j in range(0, k):
            # including the positions of the cliques
            xk.append(df_schaefer.loc[clique[j], "x"])
            yk.append(df_schaefer.loc[clique[j], "y"])
            zk.append(df_schaefer.loc[clique[j], "z"])

        if k > 2:
            # We have to pertubate a bit one of the nodes to make a mesh3d object
            xk.append(df_schaefer.loc[clique[0], "x"] + 0.05)
            yk.append(df_schaefer.loc[clique[0], "y"] + 0.05)
            zk.append(df_schaefer.loc[clique[0], "z"] + 0.05)
            # These are the poligons
            data.append(
                go.Mesh3d(
                    x=xk,
                    y=yk,
                    z=zk,
                    alphahull=0.075,
                    opacity=0.05,
                    color="blue",
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # These are the lines of the cliques
        data.append(
            go.Scatter3d(
                x=xk,
                y=yk,
                z=zk,
                mode="lines",
                line=dict(color="black", width=4, colorscale=None),
                opacity=0.15,
                hoverinfo="skip",
                showlegend=False,
            )
        )

    data.append(
        go.Scatter3d(
            x=np.array(df_schaefer["x"]),
            y=np.array(df_schaefer["y"]),
            z=np.array(df_schaefer["z"]),
            mode="markers",
            marker=dict(
                sizemode="diameter",
                symbol="circle",
                showscale=False,
                opacity=1,
                size=nodes_cliques_participation(
                    cliques_list, k, df_schaefer, node_feature_range
                ),
                color=df_schaefer["color"],
            ),
            showlegend=True,
            name="Participation",
            text=df_schaefer["region_name"],
            customdata=pd.DataFrame(
                {"node": list(df_schaefer.index), "subnet": df_schaefer["subnet"]}
            ),
            hovertemplate="<b>Node: </b>%{customdata[0]}<br>"
            + "<b>Region: </b>%{text}<br>"
            + "<b>Subnet: </b>%{customdata[1]}"
            + "<extra></extra>",
        )
    )

    # view
    if top_view:
        camera = dict(
            up=dict(x=0, y=1, z=0),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=0, y=0, z=1.8),
        )
    else:
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=1.8, y=0, z=0),
        )

    layout = go.Layout(
        title=f"Brain 3D Plot - {k}-order Interactions",
        font=dict(size=18, color="black"),
        paper_bgcolor="white",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            camera=camera,
            xaxis=dict(nticks=5, tickfont=dict(size=16, color="black"), visible=False),
            yaxis=dict(nticks=7, tickfont=dict(size=16, color="black"), visible=False),
            zaxis=dict(nticks=7, tickfont=dict(size=16, color="black"), visible=False),
        ),
    )

    fig = go.Figure(data=data, layout=layout)
    fig.update_layout(
        autosize=False,
        width=1000,
        height=800,
        margin=dict(l=50, r=50, b=75, t=75, pad=4),
    )
    fig.update_layout(
        font_family="times", legend_font_size=18, title_x=0.5, legend=dict(x=1)
    )
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
            yaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
            zaxis=dict(
                ticklen=20, showticklabels=False, zeroline=False, showbackground=False
            ),
        )
    )
    if movie != None:
        fig.write_html(movie)
    if printscreen != None:
        fig.write_image(printscreen)

    if plot_show:
        return iplot(fig)
    else:
        return


#######################
# Running Script      #
#######################

# Get names of areas
list_areas = pd.read_csv(path_areas, header=None).values
n_rois = len(list_areas)
areas = [list_areas[0:n_rois, 0][i] for i in range(0, n_rois)]

## Create gray shell
# brain_mesh =  meshio.read(path_brainobj) # Reading a brain.obj file
brain_mesh = load_mesh(path_brainobj)
brain_trace = shell_brain(brain_mesh)

df_schaefer = pd.DataFrame(
    {
        "label": schaefer_nodes_label,
        "color": schaefer_nodes_color,
        "subnet": schaefer_nodes_subnet,
        "x": schaefer_coordinates[:, 0],
        "y": schaefer_coordinates[:, 1],
        "z": schaefer_coordinates[:, 2],
        "region_name": schaefer_areas_full,
    }
)
