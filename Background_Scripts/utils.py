
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

def get_simmetrized_ttensors(file):

    A_tensor_ii, A_tensor_tc = build_adjacency_tensors(file)

    As_ii = tensor2sym(A_tensor_ii)
    As_ii = ttensor(data=As_ii)
    As_tc = tensor2sym(A_tensor_tc)
    As_tc = ttensor(data=As_tc)

    return As_ii, As_tc

def remove_outliers_zscore(df,column,threshold=2):
    ''' Detection '''
    x = df[column].to_list()
    x = np.array(x).flatten()
    z = np.abs(stats.zscore(x))
    inliers = np.where(z <= threshold)[0]
    
    return df.iloc[inliers]

def get_clusters_label(list_kmeans_models, kcluster):
    for model in list_kmeans_models:
        if (len(set(model.labels_)) == kcluster):
            return model.labels_
    raise ValueError("Model not found!")

def get_binarized_matrix(matrix, density):
    G = G_den(matrix, density, verbose=False)
    matrix_bin = nx.to_numpy_array(G)
    matrix_bin[matrix_bin > 0] = 1
    return matrix_bin

def get_modularity(matrix, cluster_labels):
    G_binarized = nx.from_numpy_matrix(matrix)
    df = pd.DataFrame({'node': range(len(cluster_labels)), 'cluster_label': cluster_labels})
    partitions = []
    for c in list(set(cluster_labels)):
        partitions.append(set(df.loc[df['cluster_label']==c]['node'].values))
    return nx.community.modularity(G_binarized, partitions)

