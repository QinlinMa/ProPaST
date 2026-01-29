# ProPaST:Niche-State Anchoring of Spatial Transcriptomics Uncovers Context-Specific Programs of Tissue Compartmentalization and Morphogenesis

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
Spatial transcriptomics (ST) is rapidly expanding across diverse sequencing- and imaging-based platforms, enabling in situ transcriptome profiling at unprecedented scale and resolution. However, platform-specific differences in gene coverage, spatial resolution, and noise often yield inconsistent spatial domain maps and hinder cross-dataset comparability. We present ProPaST, a graph representation learning framework that learns platform-invariant, structure-preserving spot-level representations from expression and spatial adjacency, enabling accurate spatial domain mapping within each technology and cross-dataset alignment by anchoring local measurements to mesoscale spatial niche prototypes that represent functional units of tissue organization. ProPaST uses graph convolutional networks to integrate gene expression with spatial context, leverages complementary feature- and graph-level perturbations to enhance robustness, and preserves tissue structure through a reconstruction constraint. It further employs a spot–neighborhood context contrastive objective that pulls each spot representation toward its aggregated neighborhood context while pushing it away from mismatched contexts sampled elsewhere in the tissue, thereby sharpening domain boundaries. In addition, ProPaST performs niche-level prototype contrast and anchoring, treating domain centroids as mesoscale niche prototypes that regularize spot representations and promote consistent domain semantics across datasets. Across diverse ST technologies, ProPaST consistently improves spatial domain clustering and achieves top overall performance in multi-platform benchmarks. Beyond single-dataset clustering, ProPaST enables robust horizontal and vertical integration of multi-sample and multi-section data and supports cross-platform alignment while retaining anatomical continuity. 


# Architecture
![framework](https://github.com/QinlinMa/ProPaST/blob/main/framework.png)

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
The datasets are freely available at [data](https://zenodo.org/records/18253702).

# Usage
Tutorial for Spatial Domain Identification: [Run.ipynb](https://github.com/QinlinMa/ProPaST/blob/main/Run%20.ipynb)  
Tutorial for Multi-Section Vertical Integration: [integration_p.ipynb](https://github.com/QinlinMa/ProPaST/blob/main/integration_p.ipynb)  
Tutorial for Multi-Section Horizontal Integration: [Horizontal_integration.ipynb](https://github.com/QinlinMa/ProPaST/blob/main/Horizontal_integration.ipynb)  
Tutorial for Cell Differentiation Trajectory Prediction: [Trajectory.ipynb](https://github.com/QinlinMa/ProPaST/blob/main/Trajectory.ipynb)  


# Contact
For correspondence and requests for materials, please contact:qinlinma528@nenu.edu.cn

# License
<!-- 内容 -->
