import torch
import numpy as np
import pandas as pd

from tqdm import tqdm
from torch import nn
import torch.nn.functional as F
from sklearn.cluster import KMeans

from .preprocess import preprocess_adj_sparse, permutation, fix_seed
from .niche_net import construct_niche_network_sample_return
from .niche_trajectory import (
    compute_cluster_adj,
    get_niche_NTScore,
    niche_to_cell_NTScore
)

class ProPaST_niche():
    def __init__(self, 
        adata,
        device,
        num_clusters,
        gamma = 0.1, 
        learning_rate=0.001,
        epochs=600,
        dim_input=3000,
        dim_output=8,
        random_seed = 42,
        alpha = 10,
        beta = 1,
        theta = 0.1,
        lamda1 = 10,
        lamda2 = 1,
        temperature=0.2
        ):
        
        self.adata = adata.copy()
        self.device = device
        self.learning_rate=learning_rate
        self.epochs=epochs
        self.num_clusters = num_clusters
        self.random_seed = random_seed
        self.alpha = alpha
        self.beta = beta
        self.theta = theta
        self.lamda1 = lamda1
        self.lamda2 = lamda2
        self.temperature= temperature
        self.gamma = gamma
        fix_seed(self.random_seed)
        sample_meta_df = pd.DataFrame({
            'Cell_ID': adata.obs['Cell_ID'],   
            'x': adata.obs['x'],
            'y': adata.obs['y']
        })
        sample_ct_coding = pd.DataFrame(
            adata.X, 
            index=adata.obs['Cell_ID'],
            columns=adata.var_names  # 确保列名对应细胞类型
            )


        coordinates, dis_matrix, indices_matrix, niche_weight_matrix, cell_type_composition = \
        construct_niche_network_sample_return(
        sample_meta_df=sample_meta_df,
        sample_ct_coding=sample_ct_coding,
        n_neighbors=50,
        n_local=20
        )


        self.coordinates = coordinates
        self.dis_matrix = dis_matrix
        self.indices_matrix = indices_matrix
        self.niche_weight_matrix = niche_weight_matrix
        self.cell_type_composition = cell_type_composition
     
    
    
        self.features = torch.FloatTensor(self.cell_type_composition.copy()).to(self.device)
        self.dim_input = self.features.shape[1] 
        self.dim_output = dim_output
       
        self.features_a = self.features.clone()
        
        self.adj_template = self.niche_weight_matrix
        print('Building sparse matrix for GNN...')
        self.adj = preprocess_adj_sparse(self.adj_template).to(self.device)
        
        N = self.features.shape[0]

        binary_adj = (self.niche_weight_matrix > 0).astype(float)
        self.graph_neigh = torch.FloatTensor(binary_adj.todense() + np.eye(N)).to(self.device)
        

        one_matrix = np.ones([N, 1])
        zero_matrix = np.zeros([N, 1])
        label_CSL_np = np.concatenate([one_matrix, zero_matrix], axis=1)
        self.label_CSL = torch.FloatTensor(label_CSL_np).to(self.device) # 形状 [5763, 2]
    def train(self):
      
        from .model import Trajectory
        
        self.model = Trajectory(self.dim_input, self.dim_output, self.graph_neigh, self.num_clusters, self.device, self.adj_template).to(self.device)
        
        self.loss_CSL = nn.BCEWithLogitsLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.learning_rate, weight_decay=0.00) # weight_decay 默认为 0
        
        print('Begin to train Niche data for embedding...')
        self.model.train()
        
        for epoch in tqdm(range(self.epochs)): 
            self.model.train()
      
            self.features_a = permutation(self.features)

            self.hiden_feat, self.emb, ret, ret_a ,cluster_contrastive_loss = self.model(self.features, self.features_a, self.adj)
            

            self.loss_sl_1 = self.loss_CSL(ret, self.label_CSL)
            self.loss_sl_2 = self.loss_CSL(ret_a, self.label_CSL)
            

            self.loss_feat = F.mse_loss(self.features, self.emb)


            loss =  self.alpha*self.loss_feat + self.beta*(self.loss_sl_1 + self.loss_sl_2) + cluster_contrastive_loss * self.gamma
            
            self.optimizer.zero_grad()
            loss.backward() 
            self.optimizer.step()
            
        print("Optimization finished for Niche data!")
        
        with torch.no_grad():
            self.model.eval()
 
            self.emb_rec = self.model(self.features, self.features_a, self.adj)[1]
            self.emb_rec = F.normalize(self.emb_rec, p=2, dim=1).detach().cpu().numpy()
            self.adata.obsm['emb'] = self.emb_rec


            X_to_cluster = self.adata.obsm['emb']
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(X_to_cluster)
            self.adata.obs['NicheCluster'] = pd.Categorical(cluster_labels.astype(str))

            N_niche = self.adata.obs['NicheCluster'].nunique()
            niche_ids = [f'Niche_{i}' for i in range(N_niche)]
            cluster_map = {str(i): niche_ids[i] for i in range(N_niche)}
            cluster_series = self.adata.obs['NicheCluster'].astype(str).map(cluster_map)
            niche_cluster_assign_df = pd.get_dummies(cluster_series, prefix='NicheCluster')
            niche_cluster_assign_df.index = self.adata.obs_names
            self.adata.obsm['Niche_One_Hot'] = niche_cluster_assign_df  

            A_cluster_normalized = compute_cluster_adj(
                self.adj_template, self.adata.obsm['Niche_One_Hot']
            )


            trajectory_construct_method = 'BF'
            niche_cluster_score, niche_level_NTScore_df = get_niche_NTScore(
                trajectory_construct_method=trajectory_construct_method,
                niche_level_niche_cluster_assign_df=self.adata.obsm['Niche_One_Hot'],
                niche_adj_matrix=A_cluster_normalized
            )


            cell_level_NTScore_df = niche_to_cell_NTScore(
                niche_level_NTScore_df,
                self.niche_weight_matrix,
                cell_ids=self.adata.obs_names
            )


            self.adata.obs['Cell_NTScore'] = cell_level_NTScore_df['Cell_NTScore']
            self.adata.obs['Niche_NTScore'] = niche_level_NTScore_df['Niche_NTScore']

            return self.adata
