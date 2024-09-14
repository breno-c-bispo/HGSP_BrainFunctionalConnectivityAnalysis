
import numpy as np
import numpy.linalg as lg
import pandas as pd

# Network library
import networkx as nx

# Scikit-learn libraries
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.cluster import KMeans

# Karateclub library
import karateclub as kc

from itertools import permutations
from scipy import stats
from ast import literal_eval

from ttensor import *
from HCP_Data_Vis_Schaefer_100Parcels import G_den

def pre_processing_data(raw_data, n_features, scaling=True, normalizing=True):
    """Standardize and/or normalize features
        
        Parameters
        ----------
        raw_data: numpy array matrix
        
        n_features: integer

        scaling: boolean

        normalizing: boolean

        Returns
        ----------
        X: numpy array matrix
    """

    # Scaling the Data 
    if scaling:
        X = StandardScaler().fit_transform(raw_data[:,:n_features])
    else:
        X = raw_data[:,:n_features]
    
    # Normalizing the Data
    if normalizing: 
        X = normalize(X)
    
    return X
    
def graph_spectrum_components(A, shift_operator='laplacian', norm_type='sym', eig_type=None):
    """Compute the eigenvalues/eigenvectors of the adjacency matrix A or its Laplacian

        Parameters
        ---------
        A: numpy array matrix

        shift_operator: string ('adjacency' or 'laplacian')
                        if equals to 'adjacency' it returns the eigendecomposition of the (normmalized)
                        adjacency matrix A. If equals to 'laplacian' it returns the eigendecomposition of the (normmalized)
                        Laplacian matrix of A
        norm_type: string ('rw' or 'sym' or None)
                    if 'rw', the function returns the eigendecomposition of the random normalized version of A (or its Laplacian).
                    if 'sym', the function returns the eigendecomposition of the symmetrically normalized version of A (or its Laplacian).
                    if None, the function returns the eigendecomposition of A (or its Laplacian).
        eig_type: string ('U' or 'L' or None)
                    If A (or Laplacian) is real symmetric matrix, specifies whether the eigendecomposition is done with the lower triangular part of a (‘L’) or the upper triangular part (‘U’).
                    If None, it returns the eigendecomposition of ordinary matrix A (or Laplacian).
    """
    
    A = np.squeeze(A)

    if norm_type == 'sym' or norm_type == 'rw' or shift_operator == 'laplacian':
        vector_degree = np.sum(A, axis=0)
        D = np.diag(vector_degree)
        D_inv = np.diag(1/vector_degree)

    if shift_operator == 'adjacency':
        F = A
    elif shift_operator == 'laplacian':
        F = D - A
    else:
        ValueError("Chosse shift_operator= 'adjacency' or 'laplacian'")

    if norm_type == 'sym':
        D_inv_sqrt = np.sqrt(D_inv)
        F = D_inv_sqrt @ F @ D_inv_sqrt
    elif norm_type == 'rw':
        F = D_inv @ F
    elif norm_type == None:
        F = F
    else:
        ValueError("Choose norm_type= None or 'sym' or 'rw'")

    if eig_type == None:
        Lambda_F, U_F = lg.eig(F)
    elif eig_type == 'U' or eig_type == 'L':
        Lambda_F, U_F = lg.eigh(F, UPLO=eig_type)
    else:
        ValueError("Choose eig_type= None or 'U' or 'L'")
    
    idx_frequency_F = np.argsort(Lambda_F)
    if shift_operator == 'adjacency':
        idx_frequency_F = np.flip(idx_frequency_F)

    U_F = U_F[:,idx_frequency_F]
    Lambda_F = Lambda_F[idx_frequency_F]

    return Lambda_F, U_F
        

def graph_spectral_clustering(A, k_clusters, shift_operator='laplacian', norm_type='sym', eig_type=None, scaling=True, normalizing=True, n_init=50):
    """Returns 'k_clusters' spectral clustering K-Means models 'kmeans_models' using the (normalized) adjacency matrix A (or its Laplacian),
        list of silhouette scores of the 'kmeans_models' and list of embeddings 'embeddings'.

        Parameters
        ----------
        A: numpy array matrix

        k_clusters: list of integer
                    list of K clusters for which the function computes the spectral clustering models
        
        shift_operator: string ('adjacency' or 'laplacian')
                        if equals to 'adjacency' it computes the spectral clustering models of the (normmalized)
                        adjacency matrix A. If equals to 'laplacian' it computes the spectral clustering models of the (normmalized)
                        Laplacian matrix of A
        
        norm_type: string ('rw' or 'sym' or None)
                    if 'rw', the function computes the spectral clustering models of the random normalized version of A (or its Laplacian).
                    if 'sym', the function computes the spectral clustering models of the symmetrically normalized version of A (or its Laplacian).
                    if None, the function computes the spectral clustering models of A (or its Laplacian).
        
        eig_type: string ('U' or 'L' or None)
                    if A (or Laplacian) is real symmetric matrix, specifies whether the eigendecomposition is done with the lower triangular part of a (‘L’) or the upper triangular part (‘U’).
                    If None, it returns the eigendecomposition of ordinary matrix A (or Laplacian).
        
        scaling: boolean
                if True, standardize features of the embedding of A (or its Laplacian).
        
        normalizing: boolean
                if True, normalize features of the embedding of A (or its Laplacian).

        n_init: integer
                number of times the K-Means algorithm is run with different centroid seeds.
    """
    
    kmeans_models = []
    silhouette_score_models = []
    embeddings = []
    _, U_F = graph_spectrum_components(A, shift_operator, norm_type, eig_type)
    for i, k in enumerate(k_clusters):
        X = pre_processing_data(U_F, n_features=k, scaling=scaling, normalizing=normalizing)
        model = KMeans(n_clusters=k, random_state=42, n_init=n_init).fit(X)
        silhouette_score_model = silhouette_score(X, model.labels_)
        kmeans_models.append(model)
        silhouette_score_models.append(silhouette_score_model)
        embeddings.append(X)

    return kmeans_models, silhouette_score_models, embeddings

def danmf_clustering(A, k_clusters, layers, scaling=True, normalizing=True, pre_iterations=100, iterations=100, lamb=0.01, n_init=50):
    """Returns 'k_clusters' DANMF-based K-Means models 'kmeans_models' using the (normalized) adjacency matrix A, list of silhouette
        scores of the 'kmeans_models' and list of embeddings 'embeddings'.

        Parameters
        ----------
        A: numpy array matrix

        k_clusters: list of integer
                    list of K clusters for which the function computes the DANMF clustering models
        
        scaling: boolean
                if True, standardize features of the embedding of A (or its Laplacian).
        
        normalizing: boolean
                if True, normalize features of the embedding of A (or its Laplacian).

        pre_iterations: integer
                    number of pre-training epochs of the DANMF algorithm.

        iterations: integer
                    number of DANMF training epochs of the DANMF algorithm.

        lamb: float
                regularization parameter of the DANMF algorithm.
        
        n_init: integer
                number of times the K-Means algorithm is run with different centroid seeds.
    """
    
    G = nx.from_numpy_matrix(A)
    kmeans_models = []
    silhouette_score_models = []
    embeddings = []
    for i, k in enumerate(k_clusters):
        layers_model = layers + [k]
        danmf_model = kc.DANMF(layers=layers_model, pre_iterations=pre_iterations, iterations=iterations, lamb=lamb)
        danmf_model.fit(G)
        embeddings_danmf = danmf_model.get_embedding()

        X = pre_processing_data(embeddings_danmf, n_features=embeddings_danmf.shape[1], scaling=scaling, normalizing=normalizing)
        model = KMeans(n_clusters=k, random_state=42, n_init=n_init).fit(X)
        silhouette_score_model = silhouette_score(X, model.labels_)
        kmeans_models.append(model)
        silhouette_score_models.append(silhouette_score_model)
        embeddings.append(X)
        print('KMeans Clusters %(d)s - Done!' % {'d':k})
    
    return kmeans_models, silhouette_score_models, embeddings

def build_adjacency_tensors(file):
    """Generate the 3-order adjacency tensors 'A_tensor_ii' and 'A_tensor_tc' from the 'file',
        containing the hyperedges weights of the 3-uniform hypergraphs
    """
    
    M = 3
    N = 116

    triplet_weights = pd.read_csv(file)

    Zero_tensor = np.zeros(np.full(M, N))
    A_tensor_ii = Zero_tensor.copy()
    A_tensor_tc = Zero_tensor.copy()
    triplet_labels = [tuple(np.array(literal_eval(t))) for t in triplet_weights['triplets'].tolist()]
    for i, ind in enumerate(triplet_labels):
        permut = list(permutations(ind))
        for p in permut:
            A_tensor_ii[p] = triplet_weights['II'][i]
            A_tensor_tc[p] = triplet_weights['TC'][i]
    
    return A_tensor_ii, A_tensor_tc

def get_symmetrized_ttensors(file):
    """Generate the 3-order adjacency tensors 'A_tensor_ii' and 'A_tensor_tc' from the 'file',
        symmetrize the tensors and create the ttensors objects 'As_ii' and 'As_tc'.
    """

    A_tensor_ii, A_tensor_tc = build_adjacency_tensors(file)

    As_ii = tensor2sym(A_tensor_ii)
    As_ii = ttensor(data=As_ii)
    As_tc = tensor2sym(A_tensor_tc)
    As_tc = ttensor(data=As_tc)

    return As_ii, As_tc

def remove_outliers_zscore(df,column,threshold=2):
    """Remove outliers from the 'column' of the Pandas Dataframe 'df' using z-score threshold
        and return the corresponding collumn without outliers.

        Parameters
        ----------
        df: Pandas Dataframe

        column: string
    """
    x = df[column].to_list()
    x = np.array(x).flatten()
    z = np.abs(stats.zscore(x))
    inliers = np.where(z <= threshold)[0]
    
    return df.iloc[inliers]

def get_clusters_label(list_kmeans_models, kcluster):
    """Return the cluster labels from the list of K-Means models 'list_kmeans_models'
        which has 'kcluster' clusters.

        Parameters
        ----------
        list_kmeans_models: list of K-Means objects

        kcluster: integer
    """
    for model in list_kmeans_models:
        if (len(set(model.labels_)) == kcluster):
            return model.labels_
    raise ValueError("Model not found!")

def get_binarized_matrix(matrix, density):
    """Binarize a matrix according to the desired density.

        Parameters
        ----------
        matrix: numpy array matrix

        density: float
    """
    G = G_den(matrix, density, verbose=False)
    matrix_bin = nx.to_numpy_array(G)
    matrix_bin[matrix_bin > 0] = 1
    return matrix_bin

def get_modularity(matrix, cluster_labels):
    """Return the modularity of a network according to the cluster label

        Parameters
        ----------
        matrix: numpy array matrix

        cluster_labels: list of integer
                cluster label of each node of the network
    """
    G_binarized = nx.from_numpy_matrix(matrix)
    df = pd.DataFrame({'node': range(len(cluster_labels)), 'cluster_label': cluster_labels})
    partitions = []
    for c in list(set(cluster_labels)):
        partitions.append(set(df.loc[df['cluster_label']==c]['node'].values))
    return nx.community.modularity(G_binarized, partitions)

def laplacian(A):
    """Return the laplacian matrix of 'A'

    Parameters
    ----------
    A: numpy array matrix
    
    """
    return np.diag(np.sum(A, axis=0)) - A

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
    D_inv = np.diag(1/vector_degree)

    if sym:
        D_inv_sqrt = np.sqrt(D_inv)
        A_norm = D_inv_sqrt @ A @ D_inv_sqrt
    else:
        A_norm = D_inv @ A
    
    return A_norm

