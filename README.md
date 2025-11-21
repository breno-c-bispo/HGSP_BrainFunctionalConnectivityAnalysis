# Notebook of the paper entitled "From Pairwise to Higher-Order Brain Community Detection: A Hypergraph Signal Processing Approach on Brain Functional Connectivity Analysis"

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

 The primary purpose of this project is to reproduce the results depicted in the paper entitled **From Pairwise to Higher-Order Brain Community Detection: A Hypergraph Signal Processing Approach on Brain Functional Connectivity Analysis**, submitted in *Computers in Biology and Medicine* journal.

### <a id='content'></a> Content:

This folder consist in the following folders / files:

- **hgsp_brain_communities.ipynb**: a Jupyter Notebook that reproduce the results presented in the paper *From Pairwise to Higher-Order Brain Community Detection: A Hypergraph Signal Processing Approach on Brain Functional Connectivity Analysis*, submitted on Computers in Biology and Medicine journal;
- **Background_Scripts**: a folder that consists of auxiliary Python modules related to (Hyper-)Graph Signal Processing tools, clustering algorithms and plotting functions;
- **Schaefer_100Parcels_Atlas**: a folder that consists of spacial coordinates, region names/colors, subnet names/colors, etc, of 116 regions-of-interest (ROIs) of the brain (based on the Schaefer's brain atlas https://doi.org/10.1093/cercor/bhx179);
- **3D_Brain_Model**: a folder that contains the 3D brain surface object.

### <a id='data'></a> Data availability:

The volunteers' datasets are available at https://doi.org/10.5281/zenodo.17538433.

### <a id='requirements'></a> Usage and Requirements:

This project uses **Anaconda and Python version 3.9**. In this way, we recommend creating a new environment in Anaconda dedicated for the use of this notebook (primarly tested on Windows) following the instructions provided in this repository: https://github.com/multinetlab-amsterdam/network_TDA_tutorial.

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

- permetrics: 2.0.0
- scipy     : 1.8.0
- tqdm      : 4.67.1
- trimesh   : 4.7.0
- seaborn   : 0.11.2
- sys       : 3.9.23
- networkx  : 2.4
- numpy     : 1.22.3
- joblib    : 1.5.1
- thoi      : 0.2.37
- re        : 2.2.1
- pandas    : 1.3.5
- karateclub: 1.3.3
- plotly    : 4.6.0
- matplotlib: 3.5.2
- sklearn   : 1.6.1
- IPython   : 8.4.0


Now, you are ready to run the project.
     

### <a id='citation'></a> Citation:

Here are some of the key papers where the data visualisation of this project is based on:

- [Centeno, E. G. Z., Moreni, G., Vriend, C., Douw, L., & Santos, F. A. N. (2022). A hands-on tutorial on network and topological neuroscience. Brain Structure and Function, 227(3), 741-762.](https://link.springer.com/article/10.1007/s00429-021-02435-0)

- [Santos, F. A., Tewarie, P. K., Baudot, P., Luchicchi, A., Barros de Souza, D. A., Girier, G., ... & Quax, R. (2023). Emergence of High-Order Functional Hubs in the Human Brain. bioRxiv, 2023-02.](https://www.biorxiv.org/content/10.1101/2023.02.10.528083v1)


### <a id='acknowledgements'></a>Acknowledgements:

- Breno C. Bispo would like to acknowledge support from Dutch Institute for Emergent Phenomena (DIEP), Institute for Advanced Studies at University of Amsterdam;

- This work was supported in part by Conselho Nacional de Desenvolvimento Científico e Tecnológico (CNPq) under grants 140151/2022-2, 442238/2023-1, 312935/2023-4 and 405903/2023-5, Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES) under grant 88881.311848/2018-01, 88887.899136/2023-00, and Fundação de Amparo à Ciência e Tecnologia do Estado de Pernambuco (FACEPE) under grant APQ-1226-3.04/22.