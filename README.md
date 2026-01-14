# ProPaST

# Contents
  - [Overview](#overview)
  - [Architecture](#architecture)
  - [Installation](#installation)
  - [Data availability](#data-availability)
  - [Usage](#usage)
  - [Key Functions](#key-functions)
  - [Results](#results)
  - [Contact](#contact)



# Overview
Spatial transcriptomics enables the characterization of tissue microenvironments, cell–cell interactions, and aspects of disease pathology. Unlike traditional single-cell RNA sequencing (scRNA-seq), which lacks spatial context, it retains the relative positions of cells, providing additional information about tissue organization. Integrating spatial location with gene expression profiles in an informative and robust manner remains challenging. Overemphasizing spatial information may oversmooth transcriptional heterogeneity and obscure small-scale functional domains, whereas relying primarily on gene expression can yield clusters with limited spatial coherence, potentially missing intrinsic tissue structures. To address these issues, we propose ProPaST, a prototype-aware graph contrastive learning framework aimed at identifying cellular niches while preserving local microenvironmental features. Benchmarking across multiple datasets indicates that ProPaST achieves improved performance compared with current state-of-the-art methods in spatial domain identification. Analyses including multi-sample batch integration and cell differentiation trajectories suggest that ProPaST generalizes effectively across datasets. Application to metastatic breast cancer samples revealed potential biological features within the tumor microenvironment, highlighting aspects of disease mechanisms. Overall, ProPaST provides a robust framework for dissecting tissue architecture and understanding cellular organization in complex tissues.   

# Architecture
![dasdsa](/framework.pdf)

# Installation
## Requirements
```bash
Python == 3.8
torch==2.4.1+cu118
torch-geometric==2.6.1
numpy==1.22.3
scipy==1.8.1
pandas==1.5.3
scikit-learn==1.1.1
scanpy==1.9.8
anndata==0.8.0
networkx==3.0
python-igraph==0.11.8
leidenalg==0.10.2
louvain==0.8.2
POT==0.9.5
matplotlib==3.7.1
seaborn==0.13.2
tqdm==4.64.0
rpy2==3.4.1
```
## Installation
```bash
conda create -n ProPaST_env python=3.8
conda activate ProPaST_env
pip install -r requirements.txt
```
# Data availability
The datasets are freely available at [data](XXXXX).

# Usage
Tutorial for Spatial Domain Identification: [Run.ipynb](https://github.com/QinlinMa/ProPaST/blob/main/Run%20.ipynb)  
Tutorial for Multi-Section Vertical Integration: [integration_p.ipynb](https://github.com/QinlinMa/ProPaST/blob/main/integration_p.ipynb)  
Tutorial for Multi-Section Horizontal Integration: [Horizontal_integration.ipynb](https://github.com/QinlinMa/ProPaST/blob/main/Horizontal_integration.ipynb)  
Tutorial for Cell Differentiation Trajectory Prediction: [Trajectory.ipynb](https://github.com/QinlinMa/ProPaST/blob/main/Trajectory.ipynb)  

# Key Functions
<!-- 内容 -->

# Results
<!-- 内容 -->

# Contact
<!-- 内容 -->
