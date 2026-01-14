from pathlib import Path
from typing import Dict, List, Tuple, Union
import scipy.sparse as sp
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from scipy.sparse import load_npz

import itertools
from typing import List

import numpy as np

def compute_cluster_adj(A_niche: np.ndarray,
                        niche_cluster_assign_df: pd.DataFrame,
                        normalize: bool = True) -> np.ndarray:
   
    S = niche_cluster_assign_df.values  # N_niche x N_cluster

    # 根据 A_niche 是否为稀疏矩阵选择计算方法
    if sp.issparse(A_niche):
        S_sparse = sp.csr_matrix(S)
        A_cluster_sparse = S_sparse.T @ A_niche @ S_sparse  # C x C
        A_cluster = A_cluster_sparse.toarray()
    else:
        A_cluster = S.T @ A_niche @ S  # C x C

    # 可选归一化
    if normalize:
        d = np.sqrt(np.sum(A_cluster, axis=1, keepdims=True)) + 1e-15
        A_cluster_normalized = A_cluster / d / d.T
    else:
        A_cluster_normalized = A_cluster

    print("Cluster-level adjacency matrix shape:", A_cluster_normalized.shape)
    return A_cluster_normalized
def brute_force(conn_matrix: np.ndarray) -> List[int]:
    """
    Brute force method to find the optimal path with the highest connectivity.
    """
    max_connectivity = float('-inf')
    optimal_path = []
    for path in itertools.permutations(range(len(conn_matrix))):
        connectivity = 0
        for i in range(len(path) - 1):
            connectivity += conn_matrix[path[i], path[i + 1]]
        if connectivity > max_connectivity:
            max_connectivity = connectivity
            optimal_path = list(path)
    return optimal_path



def get_niche_trajectory_path(trajectory_construct_method: str, niche_adj_matrix: ndarray) -> List[int]:
    """
    Get niche trajectory path
    :param trajectory_construct_method: str, the method to construct trajectory
    :param adj_matrix: non-negative ndarray, adjacency matrix of the graph
    :return: List[int], the niche trajectory
    """

    niche_adj_matrix = (niche_adj_matrix + niche_adj_matrix.T) / 2

    if trajectory_construct_method == 'BF':
       

        niche_trajectory_path = brute_force(niche_adj_matrix)

    return niche_trajectory_path


def trajectory_path_to_NC_score(niche_trajectory_path: List[int],
                                niche_clustering_sum: ndarray) -> ndarray:
    """
    Convert niche cluster trajectory path to NTScore
    :param niche_trajectory_path: List[int], the niche trajectory path
    :param niche_clustering_sum: ndarray, the sum of each niche cluster
    :param equal_space: bool, whether the niche clusters are equally spaced in the trajectory
    :return: ndarray, the NTScore
    """

    
    niche_NT_score = np.zeros(len(niche_trajectory_path))
    
    values = np.linspace(0, 1, len(niche_trajectory_path))
    for i, index in enumerate(niche_trajectory_path):
        # debug(f'i: {i}, index: {index}')
        niche_NT_score[index] = values[i]
    return niche_NT_score


def get_niche_NTScore(trajectory_construct_method: str,
                      niche_level_niche_cluster_assign_df: DataFrame,
                      niche_adj_matrix: ndarray) -> Tuple[ndarray, DataFrame]:
    """
    Get niche-level niche trajectory and cell-level niche trajectory
    :param trajectory_construct_method: str, the method to construct trajectory
    :param niche_level_niche_cluster_assign_df: DataFrame, the niche-level niche cluster assignment. #niche x #niche_cluster
    :param adj_matrix: ndarray, the adjacency matrix of the graph
    :return: Tuple[ndarray, DataFrame], the niche-level niche trajectory and cell-level niche trajectory
    """



    niche_trajectory_path = get_niche_trajectory_path(trajectory_construct_method=trajectory_construct_method,
                                                      niche_adj_matrix=niche_adj_matrix)

    niche_clustering_sum = niche_level_niche_cluster_assign_df.values.sum(axis=0)
    niche_cluster_score = trajectory_path_to_NC_score(niche_trajectory_path=niche_trajectory_path,
                                                      niche_clustering_sum=niche_clustering_sum)
    niche_level_NTScore_df = pd.DataFrame(niche_level_niche_cluster_assign_df.values @ niche_cluster_score,
                                          index=niche_level_niche_cluster_assign_df.index,
                                          columns=['Niche_NTScore'])
    return niche_cluster_score, niche_level_NTScore_df

def niche_to_cell_NTScore(niche_level_NTScore, niche_weight_matrix, cell_ids=None):
    # DataFrame → ndarray
    if isinstance(niche_level_NTScore, pd.DataFrame):
        niche_level_NTScore = niche_level_NTScore.values

    # 强制列向量 (N_niche, 1)
    if niche_level_NTScore.ndim == 1:
        niche_level_NTScore = niche_level_NTScore.reshape(-1, 1)
    elif niche_level_NTScore.shape[0] == 1 and niche_level_NTScore.shape[1] != 1:
        niche_level_NTScore = niche_level_NTScore.T

    if niche_weight_matrix.shape[0] != niche_level_NTScore.shape[0]:
        raise ValueError(f"Inconsistent number of niches. "
                         f"Niche weight matrix: {niche_weight_matrix.shape[0]}, "
                         f"NTScore: {niche_level_NTScore.shape[0]}")

    niche_to_cell_matrix = (niche_weight_matrix / niche_weight_matrix.sum(axis=0)).T
    cell_level_NTScore = niche_to_cell_matrix @ niche_level_NTScore

    if cell_ids is None:
        cell_ids = [f"Cell_{i}" for i in range(cell_level_NTScore.shape[0])]

    cell_level_NTScore_df = pd.DataFrame(cell_level_NTScore, index=cell_ids, columns=['Cell_NTScore'])
    return cell_level_NTScore_df
