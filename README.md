# Notebooks of the papers entitled "Emergence of Higher-Order Functional Brain Connectivity with Hypergraph Signal Processing" and "From Pairwise to Higher-order Clustering: A (Hyper-)graph Signal Processing Approach on Brain Functional Connectivity Analysis"

Authors: Breno Bispo, José Neto, Juliano Lima and Fernando Santos 

Contact information: <breno.bispo@ufpe.br> or <juliano.lima@ufpe.br> or <f.a.nobregasantos@uva.nl>

## Table of contents
1. [General information](#general_information)
2. [Content](#content)
3. [Data availability](#data)
4. [Requirements](#requirements)
5. [Citation](#citation)
6. [Acknowledgements](#acknowledgements)


### <a id='general_information'></a> General information:

 The primary purpose of this project is to reproduce the results depicted in the paper entitled **Emergence of Higher-Order Functional Brain Connectivity with Hypergraph Signal Processing**, published in *32nd European Signal Processing Conference (EUSIPCO 2024)*, and **From Pairwise to Higher-order Clustering: A (Hyper-)graph Signal Processing Approach on Brain Functional Connectivity Analysis**, submitted on *IEEE Journal of Biomedical and Health Informatics (JBHI)*.

### <a id='content'></a> Content:

This folder consist in the following folders / files:

- **hgsp_spectrum_analysis_fmri.ipynb**: a Jupyter Notebook that reproduce the results presented in the paper *Emergence of Higher-Order Functional Brain Connectivity with Hypergraph Signal Processing*, published in EUSIPCO 2024 (DOI: https://doi.org/10.23919/EUSIPCO63174.2024.10715376);
- **hgsp_clustering_analysis_fmri.ipynb**: a Jupyter Notebook that reproduce the results presented in the paper *From Pairwise to Higher-order Clustering: A (Hyper-)graph Signal Processing Approach on Brain Functional Connectivity Analysis*, submitted on IEEE JBHI;
- **hgsp_brain_dataset_setup.py**: a script to pre-process the correlation matrices and HOI weights of each rs-fMRI scan. Moreover, it computes the mean individual correlation matrix and the HOI weights;
- **Background_Scripts**: a folder that consists of auxiliary Python modules related to (Hyper-)Graph Signal Processing tools, clustering algorithms and plotting functions;
- **Schaefer_100Parcels_Atlas**: a folder that consists of spacial coordinates, region names/colors, subnet names/colors, etc, of 116 regions-of-interest (ROIs) of the brain (based on the Schaefer's brain atlas https://doi.org/10.1093/cercor/bhx179), the mean individual correlation matrix $\mathbf{A}$ (given by the average of $\mathbf{A}^{[0,1,\cdots,1977]}$), and the mean individual HOI weights using *interaction information* ($II$) and *total correlation* ($TC$) metrics;
- **Figures**: a folder that consists the following subfolders:
  - *Emergence_of_Higher-Order_Functional_Brain_Connectivity*: contains the figures of the paper *Emergence of Higher-Order Functional Brain Connectivity with Hypergraph Signal Processing*
  - *From_Pairwise_to_HIgher-order_Clustering*: it contains the following subfolders:
    - brain_hypergraph_3D_models: a folder that contains 3D representations of the brain graph $\mathcal{G}$, and the brain hypergraphs $\mathcal{H}\_{II}$ and $\mathcal{H}\_{TC}$;
    - clustering_outcomes: a folder that contains the clustering outcomes from the brain graph $\mathcal{G}$, and brain hypergraph modes $\mathcal{G}^{(k=0,4)}\_{II}$, $\mathcal{G}^{(k=0,4)}\_{TC}$;
    - binarized_adjacency_matrices: a folder that contains binarized adjacency matrices of $\mathcal{G}$, $\mathcal{G}^{(k=0,4)}\_{II}$, $\mathcal{G}^{(k=0,4)}\_{TC}$ (for different values of connection density $D \in \\{0\\% ,5\\%,15\\%,20\\%\\}$), with rows and columns ordered based on the clustering outcomes;
    - null_models: a folder that contains 3D illustrations of brain clusters from the null models of $\mathcal{G}^{(k=0,4)}\_{II}$, $\mathcal{G}^{(k=0,4)}\_{TC}$. Additionally, it contains heatmaps of the adjacency matrices of the original versions of brain hypergraph modes and their corresponding rewired (null model) counterparts;
    - brain_signatures: a folder that contains the results from the analysis of brain signatures described in the paper.

### <a id='data'></a> Data availability:

The volunteers' correlation matrices from the rs-fMRI time series are available at https://doi.org/10.5281/zenodo.6770120. Additionally, the hyperedge weights of the triangles, calculated using $II$ and $TC$ metrics, are available at https://doi.org/10.5281/zenodo.14606768.

### <a id='requirements'></a> Usage and Requirements:

This project uses **Anaconda and Python version 3.7**. In this way, we recommend creating a new environment in Anaconda dedicated for the use of this notebook (primarly tested on Windows) following the instructions provided in this repository: https://github.com/multinetlab-amsterdam/network_TDA_tutorial.

To install these dependencies, follow these steps:

1. Activate the new environment in the command line (Anaconda prompt):

```bash
conda activate envname
```

You should now see the name of your virtual environment in your terminal prompt, indicating that the virtual environment is active.

2. Change to the notebook's directory:

```bash
cd path\to\notebookfolder
```

3. Install the required packages:

** with environment-specific python.exe (Windows)
```bash
path\to\anaconda3\envs\envname\python.exe -m pip install -r requirements.txt
```
** MacOS users do not need to give the path to the environment's python.exe
```bash
pip install -r requirements.txt
```

This command will install all the packages listed in `requirements.txt`:

  - scikit-learn: 1.0.2
  - jupyterlab  : 3.3.4
  - notebook    : 6.5.2
  - networkx  : 2.6.3
  - karateclub: 1.3.3
  - seaborn   : 0.11.0
  - scipy     : 1.5.0
  - matplotlib: 3.3.2
  - numpy     : 1.18.5
  - plotly    : 4.6.0
  - pandas    : 1.1.3
  - bctpy     : 0.6.0


Now, you are ready to run the project.
     

### <a id='citation'></a> Citation:

Here are some of the key papers where the data visualisation of this project is based on:

- [Centeno, E. G. Z., Moreni, G., Vriend, C., Douw, L., & Santos, F. A. N. (2022). A hands-on tutorial on network and topological neuroscience. Brain Structure and Function, 227(3), 741-762.](https://link.springer.com/article/10.1007/s00429-021-02435-0)

- [Santos, F. A., Tewarie, P. K., Baudot, P., Luchicchi, A., Barros de Souza, D. A., Girier, G., ... & Quax, R. (2023). Emergence of High-Order Functional Hubs in the Human Brain. bioRxiv, 2023-02.](https://www.biorxiv.org/content/10.1101/2023.02.10.528083v1)


### <a id='acknowledgements'></a>Acknowledgements:

- Breno C. Bispo would like to acknowledge support from Dutch Institute for Emergent Phenomena (DIEP), Institute for Advanced Studies at University of Amsterdam;

- This work was supported in part by Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq) under grants 140151/2022-2, 442238/2023-1, 312935/2023-4 and 405903/2023-5, Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) under grant 88881.311848/2018-01, 88887.899136/2023-00, and Fundação de Amparo à Ciência e Tecnologia do Estado de Pernambuco (FACEPE) under grant APQ-1226-3.04/22.