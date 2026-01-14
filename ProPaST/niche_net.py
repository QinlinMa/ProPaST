from typing import Optional, Tuple
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.spatial import cKDTree

# 计算最近邻
def build_knn_network(sample_meta_df: pd.DataFrame,
                      n_neighbors: int = 50,
                      n_local: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = sample_meta_df[['x', 'y']].values
    kdtree = cKDTree(data=coordinates)
    dis_matrix, indices_matrix = kdtree.query(x=coordinates, k=np.max([n_neighbors, n_local]) + 1)
    return coordinates, dis_matrix, indices_matrix


# 构建网络
def calc_edge_index(sample_meta_df: pd.DataFrame,
                    indices_matrix: np.ndarray,
                    n_neighbors: int = 50) -> np.ndarray:
    N = sample_meta_df.shape[0]
    src_indices = np.repeat(np.arange(N), n_neighbors)
    dst_indices = indices_matrix[:, 1:n_neighbors + 1].flatten()
    adj_matrix = csr_matrix((np.ones(dst_indices.shape[0]), (src_indices, dst_indices)), shape=(N, N))
    adj_matrix = adj_matrix + adj_matrix.transpose()
    edge_index = np.argwhere(adj_matrix > 0)
    return edge_index


# 计算高斯权重
def gauss_dist_1d(dist: np.ndarray, n_local: int) -> float:
    return np.exp(-(dist / dist[n_local])**2)


# 构建权重稀疏邻接矩阵 N x N
def calc_niche_weight_matrix(sample_meta_df: pd.DataFrame,
                             dis_matrix: np.ndarray,
                             indices_matrix: np.ndarray,
                             n_neighbors: int = 50,
                             n_local: int = 20) -> csr_matrix:
    N = sample_meta_df.shape[0]
    niche_weight_matrix = np.apply_along_axis(func1d=gauss_dist_1d, axis=1, arr=dis_matrix,
                                              n_local=n_local)[:, :n_neighbors + 1]
    src_indices = np.repeat(np.arange(N), n_neighbors + 1)
    dst_indices = indices_matrix[:, :n_neighbors + 1].flatten()
    niche_weight_matrix_csr = csr_matrix((niche_weight_matrix.flatten(), (src_indices, dst_indices)),
                                         shape=(N, N))
    return niche_weight_matrix_csr


# 计算细胞类型组成
def calc_cell_type_composition(niche_weight_matrix: csr_matrix,
                               ct_coding_matrix: np.ndarray) -> np.ndarray:
    cell_to_niche_matrix = niche_weight_matrix / niche_weight_matrix.sum(axis=1)
    cell_type_composition = cell_to_niche_matrix @ ct_coding_matrix
    return cell_type_composition

def construct_niche_network_sample_return(sample_meta_df: pd.DataFrame,
                                          sample_ct_coding: pd.DataFrame,
                                          n_neighbors: int = 50,
                                          n_local: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray, csr_matrix, np.ndarray]:

    # 1) build kNN network
    coordinates, dis_matrix, indices_matrix = build_knn_network(sample_meta_df=sample_meta_df,
                                                                n_neighbors=n_neighbors,
                                                                n_local=n_local)

    # 2) calculate edge index
    edge_index = calc_edge_index(sample_meta_df=sample_meta_df,
                                 indices_matrix=indices_matrix,
                                 n_neighbors=n_neighbors)

    # 3) calculate niche weight matrix
    niche_weight_matrix = calc_niche_weight_matrix(sample_meta_df=sample_meta_df,
                                                   dis_matrix=dis_matrix,
                                                   indices_matrix=indices_matrix,
                                                   n_neighbors=n_neighbors,
                                                   n_local=n_local)

    # 4) calculate cell type composition
    cell_type_composition = calc_cell_type_composition(niche_weight_matrix=niche_weight_matrix,
                                                       ct_coding_matrix=sample_ct_coding.values)

    # 5) return all results (不保存文件)
    return coordinates, dis_matrix, indices_matrix, niche_weight_matrix, cell_type_composition
'''
1. **`coordinates`**：每个细胞的空间坐标，用于计算细胞之间的距离。(N, 2)
2. **`dis_matrix`**：每个细胞与其最近邻细胞的距离矩阵。(N, k+1),k = max(n_neighbors, n_local)
3. **`indices_matrix`**：每个细胞的最近邻细胞索引矩阵，对应 `dis_matrix` 的距离。(N, k+1)
4. **`niche_weight_matrix`**：细胞间的加权邻接矩阵，表示每个细胞与邻居细胞的关系强度。(N, N)
5. **`cell_type_composition`**：每个细胞在其微环境中各细胞类型的比例分布。(N, C),C 是细胞类型数量

'''