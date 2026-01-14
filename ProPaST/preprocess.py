import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import scanpy as sc
import ot
import scipy.sparse as sp

from torch.backends import cudnn
from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix


def permutation(feature):#打乱样本，生成负样本
    # fix_seed(FLAGS.random_seed) 
    ids = np.arange(feature.shape[0])
    ids = np.random.permutation(ids)
    feature_permutated = feature[ids]
    
    return feature_permutated 

def construct_interaction(adata, n_neighbors=3):
    """Constructing spot-to-spot interactive graph"""
    position = adata.obsm['spatial']#获取所有点的坐标
    
    # calculate distance matrix
    distance_matrix = ot.dist(position, position, metric='euclidean')#构建出N*N的距离矩阵，N为样本数
    n_spot = distance_matrix.shape[0]#获取样本数
    
    adata.obsm['distance_matrix'] = distance_matrix
    
    # find k-nearest neighbors
    interaction = np.zeros([n_spot, n_spot]) #构建一个N*N的0阵 
    for i in range(n_spot):
        vec = distance_matrix[i, :]#取出第i行，即第i+1个样本与所有样本点的距离
        distance = vec.argsort()#返回排序后的索引值
        for t in range(1, n_neighbors + 1):#取出n_neighbors个样本点，从1开始是因为distance第0个是self-self的距离
            y = distance[t]
            interaction[i, y] = 1
         
    adata.obsm['graph_neigh'] = interaction
    
    #transform adj to symmetrical adj
    adj = interaction
    adj = adj + adj.T
    adj = np.where(adj>1, 1, adj)#为了保持邻接矩阵的对称性，加上T矩阵
    
    adata.obsm['adj'] = adj
    
def preprocess(adata,hvg):#1.筛选高变基因 → 2. 归一化文库大小 → 3. 对数变换 → 4. 标准化缩放
    sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=hvg)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, zero_center=False, max_value=10)
    
def get_feature(adata, deconvolution=False):
    if deconvolution:
       adata_Vars = adata
    else:   
       adata_Vars =  adata[:, adata.var['highly_variable']]
       
    if isinstance(adata_Vars.X, csc_matrix) or isinstance(adata_Vars.X, csr_matrix):
       feat = adata_Vars.X.toarray()[:, ]
    else:
       feat = adata_Vars.X[:, ] 
    
    feat_a = permutation(feat)
    
    adata.obsm['feat'] = feat
    adata.obsm['feat_a'] = feat_a    
    
def add_contrastive_label(adata):

    n_spot = adata.n_obs
    one_matrix = np.ones([n_spot, 1])
    zero_matrix = np.zeros([n_spot, 1])
    label_CSL = np.concatenate([one_matrix, zero_matrix], axis=1)
    adata.obsm['label_CSL'] = label_CSL         
    
def normalize_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return adj.toarray()

def preprocess_adj(adj):
    adj_normalized = normalize_adj(adj)+np.eye(adj.shape[0])
    return adj_normalized 
def fix_seed(seed):#固定种子
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8' 
    
    



def drop_edges_from_adj(adj, drop_prob=0.2,seed = 42):
    if seed is not None:
        np.random.seed(seed)
    assert adj.shape[0] == adj.shape[1] 
    assert np.allclose(adj, adj.T)

    adj = adj.copy() 

    upper_tri = np.triu(adj, k=1)
    edge_indices = np.array(upper_tri.nonzero()).T 

    num_edges = edge_indices.shape[0]
    keep_mask = np.random.rand(num_edges) > drop_prob
    kept_edges = edge_indices[keep_mask]

   
    rows = np.concatenate([kept_edges[:, 0], kept_edges[:, 1]])
    cols = np.concatenate([kept_edges[:, 1], kept_edges[:, 0]])
    data = np.ones_like(rows)

    adj_dropped = sp.coo_matrix((data, (rows, cols)), shape=adj.shape)

  
    adj_dropped_dense = adj_dropped.toarray()
    np.fill_diagonal(adj_dropped_dense, 1.0)

    adj_normalized = normalize_adj(adj_dropped_dense)
    return adj_normalized


def drop_edges_from_adj_niche(adj, drop_prob=0.2, seed=42):
    if seed is not None:
        np.random.seed(seed)
        
    assert adj.shape[0] == adj.shape[1] 
    
    adj = (adj + adj.T) / 2.0 
    


    adj = adj.copy() 

    upper_tri = np.triu(adj, k=1)
    edge_indices = np.array(upper_tri.nonzero()).T 

    num_edges = edge_indices.shape[0]
    keep_mask = np.random.rand(num_edges) > drop_prob
    kept_edges = edge_indices[keep_mask]

   
    rows = np.concatenate([kept_edges[:, 0], kept_edges[:, 1]])
    cols = np.concatenate([kept_edges[:, 1], kept_edges[:, 0]])
    data = np.ones_like(rows)

    adj_dropped = sp.coo_matrix((data, (rows, cols)), shape=adj.shape)

  
    adj_dropped_dense = adj_dropped.toarray()
    np.fill_diagonal(adj_dropped_dense, 1.0)


    adj_normalized = normalize_adj(adj_dropped_dense)
    return adj_normalized

def compute_centers(x, psedo_labels,num_clusters):
    n_samples = x.size(0)
    if len(psedo_labels.size()) > 1:
        weight = psedo_labels.T
    else:
        weight = torch.zeros(num_clusters, n_samples).to(x)  # L, N
        weight[psedo_labels, torch.arange(n_samples)] = 1
    weight = F.normalize(weight, p=1, dim=1) 
    centers = torch.mm(weight, x)
    centers = F.normalize(centers, dim=1)
    return centers
        
def preprocess_adj_sparse(adj):
    adj = sp.coo_matrix(adj)
    adj_ = adj + sp.eye(adj.shape[0])
    rowsum = np.array(adj_.sum(1))
    degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())
    adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
    return sparse_mx_to_torch_sparse_tensor(adj_normalized)
    
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.
    """
    # 将稀疏矩阵转换为COO格式（坐标格式）并确保数据类型为float32
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    
    # 构建索引矩阵：
    # 1. 使用vstack将行索引和列索引堆叠成(2, nnz)的数组
    # 2. 转换为int64类型（PyTorch稀疏张量要求的索引类型）
    # 3. 转换为torch张量
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    
    # 将非零值数据转换为torch张量
    values = torch.from_numpy(sparse_mx.data)
    
    # 获取原始稀疏矩阵的形状并转换为torch.Size格式
    shape = torch.Size(sparse_mx.shape)
    
    # 创建并返回PyTorch稀疏张量
    # 参数说明：
    # indices: 形状为(2, nnz)的索引张量
    # values: 形状为(nnz,)的值张量
    # shape: 稀疏张量的整体形状
    return torch.sparse.FloatTensor(indices, values, shape)

def compute_cluster_losss(q_centers,k_centers,temperature,psedo_labels,num_clusters):

    device = q_centers.device  

    d_q = q_centers.mm(q_centers.T) / temperature
    d_k = (q_centers * k_centers).sum(dim=1) / temperature
    d_q = d_q.float()

    arange_idx = torch.arange(num_clusters, device=device)
       

    d_q[arange_idx, arange_idx] = d_k

    unique_labels = torch.unique(psedo_labels).to(device)
       
    one_hot_labels = F.one_hot(unique_labels, num_clusters).to(device)
       
    zero_classes = arange_idx[torch.sum(one_hot_labels, dim=0) == 0]
       

    mask = torch.zeros((num_clusters, num_clusters), dtype=torch.bool, device=device)
    mask[:, zero_classes] = 1
    d_q.masked_fill_(mask, -10)

    pos = d_q.diag(0)
       

    mask_neg = torch.ones((num_clusters, num_clusters), device=device)
    mask_neg = mask_neg.fill_diagonal_(0).bool()
    neg = d_q[mask_neg].reshape(-1, num_clusters - 1)
      

    loss = -pos + torch.logsumexp(torch.cat([pos.reshape(num_clusters, 1), neg], dim=1), dim=1)
    loss[zero_classes] = 0.
    loss = loss.sum() / (num_clusters - len(zero_classes))

      

    return loss
    
