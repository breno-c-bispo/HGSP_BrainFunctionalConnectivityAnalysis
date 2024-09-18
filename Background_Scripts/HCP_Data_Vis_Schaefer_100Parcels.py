#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Costumized code from the https://github.com/multinetlab-amsterdam/network_TDA_tutorial.git
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
import pandas as pd # version 1.1.3
import plotly.graph_objs as go # version 4.6.0 
import matplotlib # version 3.3.2
import numpy as np # version 1.18.5
import networkx as nx # version 2.4
import meshio # version 4.0.16
from plotly.offline import iplot, init_notebook_mode
import plotly.express as px # version 4.6.0 

########################
# Pre-defined settings #
########################

init_notebook_mode(connected=True) # Define Notebook Mode

# Pre-defined paths and constants
path_areas = './Schaefer_100Parcels_Atlas/Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_region_names_short.txt'
path_pos = './Schaefer_100Parcels_Atlas/Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_positions.txt'
path_subnet_names = './Schaefer_100Parcels_Atlas/Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_subnet_order_names.txt'
path_subnet_colors = './Schaefer_100Parcels_Atlas/Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_subnet_order_colors.txt'
path_subnet_labels = './Schaefer_100Parcels_Atlas/Schaefer2018_100Parcels_7Networks_Tian_Subcortex_S1_3T_subnet_colors_number.txt'
path_brainobj = './Figures/brain.obj'

schaefer_nodes_label = np.genfromtxt(path_subnet_labels).astype(int) - 1
schaefer_nodes_color = open(path_subnet_colors, "r").read().split('\n')
schaefer_nodes_subnet = open(path_subnet_names, "r").read().split('\n')



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
    
    h = 1.0/(pl_entries-1)
    pl_colorscale = []

    for k in range(pl_entries):
        C = list(map(np.uint8, np.array(cmap(k*h)[:3])*255))
        pl_colorscale.append([float(k*h), 'rgb' + str((C[0], C[1], C[2]))])

    return pl_colorscale


def openatlas(path_pos = path_pos):
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
    
    positions = pd.read_csv(path_pos,header=None, delim_whitespace=True)
    
    data = [list(row.values) for _, row in positions.iterrows()]
  
    return data


def dictpos(areas, path_pos = path_pos):
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
    x=[]
    y=[]
    z=[]
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
        xs.append(1.01*data[i][0])
        ys.append(1.01*data[i][1])
        zs.append(1.01*data[i][2])
   
    trace1 = go.Mesh3d(x=xs, y=ys,z=zs, alphahull=4.2, opacity=0.0005,
                       color='gray', text=areas, hoverinfo='text')
    
    return trace1, x, y, z

#uncover=False
def shell_brain(brain_mesh):
    """Returns a brain gray shell from a fixed brain.obj file
    
    Parameters
    ----------
    brain_mesh: meshio mesh object
    
    Returns
    -------
    mesh: plotly.graph_objs._mesh3d.Mesh3d
        
    """
    
    vertices = brain_mesh.points
    triangles = brain_mesh.cells[0][1]
    x, y, z = vertices.T
    I, J, K = triangles.T
    #showlegend=True gives the option to uncover the shell
    mesh = go.Mesh3d(x=x, y=y, z=z, color='grey', i=I, j=J, k=K, opacity=0.1,
                     hoverinfo=None,showlegend = True, name ='Brain 3D'  #colorscale=pl_mygrey, #intensity=z,
                     #flatshading=True, #showscale=False
                     )
  
    return mesh #iplot(fig)


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
    
    #"If you want to normalize just uncomment here"
    #ScdatanGA=np.array(normalize(Aut[i]))
    
    ScdatanGA = matrix
    matcopy = (np.copy(np.abs(ScdatanGA))) # be careful to always assing  copy of data, 
                                          # othewise will change the data as well
    matcopy[(np.copy(np.abs(ScdatanGA))) <= (1-e)] = 0.0
    Gfinal = nx.from_numpy_matrix(matcopy[:,:])
    
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
    
    #matrix i, density d. i is a matrix - ravel flatten the matrix
    np.fill_diagonal(matrix,0)
    temp = sorted(matrix.ravel(), reverse=True) # will flatten it and rank corr values
    size = len(matrix)
    cutoff = np.ceil(d * (size * (size-1))) # number of links with a given density
    tre = temp[int(cutoff)]
    G0 = nx.from_numpy_matrix(matrix)
    G0.remove_edges_from(list(nx.selfloop_edges(G0)))
    G1 = nx.from_numpy_matrix(matrix)
    for u,v,a in G0.edges(data=True):
        if (a.get('weight')) <= tre:
            G1.remove_edge(u, v)
    finaldensity = nx.density(G1)
    if verbose == True:
        print(finaldensity)
        
    return G1

def Plot_Brain_Subnets(nodes_cluster=schaefer_nodes_label, nodes_color=schaefer_nodes_color, nodes_name=schaefer_nodes_subnet, path_pos=path_pos, scale=1, title='Brain Clusters', top_view=False, printscreen=None, movie=None):
    """ Clusters on the Brain 3D plot
    
    Parameters
    ----------
    models_list: list of K-Means objects
            
    k_cluster: integer
            Select the K-Means model containing 'k_cluster' clusters
    
    node_colors: list of strings
            A list of strings containing the nodes color
    
    path_pos: string
        Path to the file with atlas coordinates

    scale: int
        Scaling factor for node size
    
    title: string

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
      
    trace1, x, y, z = dictpos(areas, path_pos)
    sizec = 1.5 * scale
    df = pd.DataFrame({'cluster': nodes_cluster, 'color': nodes_color, 'name': nodes_name})
    df = df.assign(x=x, y=y, z=z)
    data = [brain_trace, trace1]
    for c in set(list(df['cluster'])):
        df_temp = df.loc[df['cluster'] == c]
        df_temp = df_temp.sort_values(by='color', key=lambda x: x.map(x.value_counts()), ascending=False)
        trace2 = go.Scatter3d(x=np.array(df_temp['x']),
                        y=np.array(df_temp['y']),
                        z=np.array(df_temp['z']),
                        mode='markers',
                        marker=dict(sizemode='diameter', symbol='circle',
                                    showscale=False, opacity=1, size=10*sizec, cauto=True, color=df_temp['color']),
                        showlegend=True, text=None, hoverinfo=None, name=str(df_temp['name'].values[0]))
        data.append(trace2)
    
    # view
    if top_view:
        camera = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0, z=1.8))
    else:
        camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.8, y=0, z=0))  
    
    layout = go.Layout(autosize=False, # To export we need to make False and uncomment the details bellow
                        width=800, height=800, # This is to make an centered html file. To export as png, one needs to include this margin
                        margin=go.Margin(l=0, r=0, b=0, t=0, pad=4),
                        title=title,
                        font=dict(size=18, color='black'),
                        paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)',
                        
                        scene=dict(camera=camera, 
                                   xaxis=dict(nticks=5, tickfont=dict(size=16, color='black'), visible=False),
                                   yaxis=dict(nticks=7, tickfont=dict(size=16, color='black'), visible=False),
                                   zaxis=dict(nticks=7, tickfont=dict(size=16, color='black'), visible=False)))
                        
    
    fig = go.Figure(data=data, layout=layout)
    
    fig.update_layout(autosize=False, width=800, height=800, margin=dict(l=50, r=50, b=75, t=75))
    
    fig.update_layout(font_family="times", legend_font_size=18, title_x=0.5, legend=dict(x=0))

    fig.update_layout(scene=dict(xaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False), 
                                 yaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False),
                                 zaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False)))
                                               
    if movie!=None:
        fig.write_html(movie)
    if printscreen!=None:
        fig.write_image(printscreen)
    return iplot(fig)


def Plot_Brain_Graph_Signal(graph_signal, path_pos=path_pos, title='Brain graph signal', scale=75, node_colors=None, size_nodes_const=False, top_view=False, printscreen=None, movie=None):
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
        
    news = 1 #This is here in case we need to rescale the coordinates - not in use now
    trace1, x, y, z = dictpos(areas, path_pos)
    sizec = np.array(graph_signal)
    sizec = np.abs(sizec)
    # restriction in the size
    sizec=1/max(sizec)*sizec
    if size_nodes_const:
        sizec = 1.5
    if node_colors != None:
        colorV = node_colors
    else:
        colorV = np.array(graph_signal)
    trace2 = go.Scatter3d(x=news*np.array(x),
                          y=news*np.array(y),
                          z=news*np.array(z),
                          mode='markers', 
                          marker=dict(sizemode='diameter',symbol='circle',
                                      showscale=True,
                                      colorbar=dict(title='Values',
                                                    thickness=30, x=1.0,#0.95,
                                                    len=0.8, tickmode = 'array',
                                                    tick0=0, dtick=1, nticks=4),
                                      opacity=1, size=scale*sizec, color=colorV, 
                                      colorscale='Jet', cauto=True, 
                                      cmin=np.min(colorV), cmax=np.max(colorV)),
                          showlegend=False, text=areas, hoverinfo=None)
    data=[brain_trace, trace1, trace2]
    fig = go.Figure(data=data)
    # view
    if top_view:
        camera = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0, z=1.4))
    else:
        camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.4, y=0, z=0))
    
    
    
    
    layout = go.Layout(autosize=True, # To export we need to make False and uncomment the details bellow
                       #width=780, #height=540, # This is to make an centered html file. To export as png, one needs to include this margin
                        #margin=go.Margin(l=5, r=5, b=5, t=0, pad=0),
                        title=title,
                        font=dict(size=18, color='black'), 
                        paper_bgcolor='white', plot_bgcolor='white',
                        #showline=False, #zaxis=dict(title='x Axis', 
                                                     #titlefont=dict(
                                                     #family='Courier New, monospace',
                                                     #size=80, color='#7f7f7f')),
                        scene=dict(camera=camera, 
                                   xaxis=dict(nticks=5, tickfont=dict(size=14, color='black'), visible=False),
                                   yaxis=dict(nticks=7, tickfont=dict(size=14, color='black'), visible=False),
                                   zaxis=dict(nticks=7, tickfont=dict(size=14, color='black'), visible=False)))
                        
    
    fig = go.Figure(data=data, layout=layout)
    
    fig.update_layout(autosize=False, width=800, height=800, 
                      margin=dict(l=50, r=50, b=100, t=100))
    fig.update_layout(font_family="Times New Roman")

    fig.update_layout(scene=dict(xaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False), 
                                 yaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False),
                                 zaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False)))
    if movie!=None:
        fig.write_html(movie)
    if printscreen!=None:
        fig.write_image(printscreen)
    return iplot(fig)

def Plot_Brain_Clusters(models_list, k_cluster, nodes_color, path_pos=path_pos, scale=1, title='Brain Clusters', top_view=False, printscreen=None, movie=None):
    """ Clusters on the Brain 3D plot
    
    Parameters
    ----------
    models_list: list of K-Means objects
            
    k_cluster: integer
            Select the K-Means model containing 'k_cluster' clusters
    
    node_colors: list of strings
            A list of strings containing the nodes color
    
    path_pos: string
        Path to the file with atlas coordinates

    scale: int
        Scaling factor for node size
    
    title: string

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
      
    trace1, x, y, z = dictpos(areas, path_pos)
    sizec = 1.5 * scale
    for idx in range(len(models_list)):
        if len(set(models_list[idx].labels_)) == k_cluster:
            break
    nodes_cluster = models_list[idx].labels_
    if len(set(nodes_cluster)) != k_cluster:
        ValueError(f'K-Means clustering model not found containing {k_cluster} clusters!')

    df = pd.DataFrame({'cluster': nodes_cluster, 'color': nodes_color, 'name': ['Cluster {}'.format(i) for i in nodes_cluster]})
    df = df.assign(x=x, y=y, z=z)
    data = [brain_trace, trace1]
    for c in set(list(df['cluster'])):
        df_temp = df.loc[df['cluster'] == c]
        df_temp = df_temp.sort_values(by='color', key=lambda x: x.map(x.value_counts()), ascending=False)
        trace2 = go.Scatter3d(x=np.array(df_temp['x']),
                        y=np.array(df_temp['y']),
                        z=np.array(df_temp['z']),
                        mode='markers',
                        marker=dict(sizemode='diameter', symbol='circle',
                                    showscale=False, opacity=1, size=10*sizec, cauto=True, color=df_temp['color']),
                        showlegend=True, text=None, hoverinfo=None, name=str(df_temp['name'].values[0]))
        data.append(trace2)
    
    # view
    if top_view:
        camera = dict(up=dict(x=0, y=1, z=0), center=dict(x=0, y=0, z=0), eye=dict(x=0, y=0, z=1.8))
    else:
        camera = dict(up=dict(x=0, y=0, z=1), center=dict(x=0, y=0, z=0), eye=dict(x=1.8, y=0, z=0))  
    
    layout = go.Layout(autosize=False, # To export we need to make False and uncomment the details bellow
                        width=800, height=800, # This is to make an centered html file. To export as png, one needs to include this margin
                        margin=go.Margin(l=0, r=0, b=0, t=0, pad=4),
                        title=title,
                        font=dict(size=18, color='black'),
                        paper_bgcolor='white', plot_bgcolor='rgba(0,0,0,0)',
                        
                        scene=dict(camera=camera, 
                                   xaxis=dict(nticks=5, tickfont=dict(size=16, color='black'), visible=False),
                                   yaxis=dict(nticks=7, tickfont=dict(size=16, color='black'), visible=False),
                                   zaxis=dict(nticks=7, tickfont=dict(size=16, color='black'), visible=False)))
                        
    
    fig = go.Figure(data=data, layout=layout)
    
    fig.update_layout(autosize=False, width=800, height=800, margin=dict(l=50, r=50, b=75, t=75))
    
    fig.update_layout(font_family="times", legend_font_size=18, title_x=0.5, legend=dict(x=0))

    fig.update_layout(scene=dict(xaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False), 
                                 yaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False),
                                 zaxis=dict(ticklen=20, showticklabels=False, 
                                            zeroline=False, showbackground=False)))
                                               
    if movie!=None:
        fig.write_html(movie)
    if printscreen!=None:
        fig.write_image(printscreen)
    return iplot(fig)
    
    
#######################
# Running Script      #
#######################

# Creating colormaps that are compatible with plotly 
magma_cmap = matplotlib.cm.get_cmap('magma')
viridis_cmap = matplotlib.cm.get_cmap('viridis')
plasma_cmap = matplotlib.cm.get_cmap('plasma')
jet_cmap = matplotlib.cm.get_cmap('jet')
inferno_cmap = matplotlib.cm.get_cmap('inferno')
Spectral_cmap = matplotlib.cm.get_cmap('Spectral')
Dark2_cmap = matplotlib.cm.get_cmap('Dark2')


# This creates a palette with 255 points.
magma = matplotlib_to_plotly(magma_cmap, 255)
viridis = matplotlib_to_plotly(viridis_cmap, 255)
plasma = matplotlib_to_plotly(plasma_cmap, 255)
jet = matplotlib_to_plotly(jet_cmap, 255)
inferno = matplotlib_to_plotly(inferno_cmap, 255)
Spectral = matplotlib_to_plotly(Spectral_cmap, 255)

# Get names of areas
list_areas = pd.read_csv(path_areas,header=None).values
n_rois = len(list_areas)
areas = [list_areas[0:n_rois,0][i] for i in range(0,n_rois)] 

## Create gray shell
brain_mesh =  meshio.read(path_brainobj) # Reading a brain.obj file
brain_trace = shell_brain(brain_mesh)
trace1, _, _, _ = dictpos(areas, path_pos) # Transparent shell

# Simplicial plot
data = openatlas(path_pos) # This creates a dictionary with the positions in 3D of the nodes 

x = []
y = []
z = []
pos3d = {}
for i in range(0, len(data)):
    pos3d[i] = (data[i][0], data[i][1], data[i][2])
    x.append(data[i][0])
    y.append(data[i][1])
    z.append(data[i][2])



    
