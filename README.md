# Jupyter Notebook of the paper entitled "From Pairwise to Higher-order Clustering: A (Hyper-)graph Signal Processing Approach on Brain Functional Connectivity Analysis"

Authors: Breno Bispo, José Neto, Juliano Lima and Fernando Santos 

Contact information: <juliano.lima@ufpe.br> or <f.a.nobregasantos@uva.nl>

## Table of contents
1. [General information](#general_information)
2. [Requirements](#requirements)
3. [Content](#content)
4. [Acknowledgements](#acknowledgements)


### <a id='general_information'></a> General information:

 The primary purpose of this project is to reproduce the results depicted in the paper entitled "From Pairwise to Higher-order Clustering: A (Hyper-)graph Signal Processing Approach on Brain Functional Connectivity Analysis".

### <a id='content'></a> Content:

This folder consist in the following folders / files:

- **t_hgsp_fmri.ipynb**: the main Jupyter Notebook that reproduce the results presented in the papaer;
- **df_peak_ii_tc_0_4_zscore_mean_tubal_scalars.pkl**: a Pandas Dataframe that consists of peak values at $k=0,4$ of the tubal scalars average of $\left(\widehat{\mathcal{M}_1}_s^{abs}\right)^{[n]}, \left(\widehat{\mathcal{T}_1}_s^{abs}\right)^{[n]}, \left(\widehat{\mathcal{M}_2}_s^{abs}\right)^{[n]}, \left(\widehat{\mathcal{T}_2}_s^{abs}\right)^{[n]}$ for each $n$-individual brain hypergraph;
- **Background_Scripts**: a folder that consists of auxiliary Python modules related to (H-)GSP tools, clustering algorithms and plotting functions;
- **Schaefer_100Parcels_Atlas**: a folder that consists of spacial coordinates, region names/colors, subnet names/colors, etc, of the 116 regions-of-interest (ROIs) of the brain (based on the Schaefer's brain atlas https://doi.org/10.1093/cercor/bhx179), the averaged correlation matrix $\mathbf{A}$ (given by the average of $\mathbf{A}^{[0,1,\cdots,1977]}$), and the averaged zscored hyperedges weights of the triplets using *interaction information* ($II$) and *total correlation* ($TC$) metrics;
- **Binarized_Adjacency_Matrices**: a folder that consists of binarized adjacency matrices of $\mathcal{G}$, $\mathcal{G}^{(0)}\_{II}$, $\mathcal{G}^{(0)}\_{TC}$, $\mathcal{G}^{(4)}\_{II}$, $\mathcal{G}^{(4)}\_{TC}$ (for different values of density $D \in \\{0,5,15,20\\}$%), with rows and columns ordered based on the clustering outcomes;
- **Figures**: a folder that consists of a 3D brain object for plotting purposes.

### <a id='requirements'></a> Requirements:
The essential core packages are depicted below. The Anaconda environment setup follows the instructions provided in this repository: https://github.com/multinetlab-amsterdam/network_TDA_tutorial.git.

     - Python version       : 3.7.16
     - meshio      : 4.0.16
     - scikit-learn: 1.0.2
     - jupyterlab  : 3.3.4
     - notebook    : 6.5.2
     - networkx  : 2.6.3
     - karateclub: 1.3.3
     - seaborn   : 0.11.0
     - scipy     : 1.5.0
     - sys       : 3.7.16
     - matplotlib: 3.3.2
     - numpy     : 1.18.5
     - plotly    : 4.6.0
     - pandas    : 1.1.3

### <a id='acknowledgements'></a>Acknowledgements:

This work was supported in part by Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq) under grants 140151/2022-2, 442238/2023-1, 312935/2023-4 and 405903/2023-5, Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) under grant 88881.311848/2018-01, 88887.899136/2023-00, and Fundação de Amparo à Ciência e Tecnologia do Estado de Pernambuco (FACEPE) under grant APQ-1226-3.04/22.