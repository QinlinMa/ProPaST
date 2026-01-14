import torch
import torch.nn as nn
import math
from .preprocess import drop_edges_from_adj,compute_centers,compute_cluster_losss,drop_edges_from_adj_niche
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from .torch_clustering import PyTorchKMeans, FaissKMeans, PyTorchGaussianMixture, evaluate_clustering
class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)  #将c调整形状

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)#拼接分数成为1个变量【sc_1,sc_2】

        return logits
    
class AvgReadout(nn.Module):    
    """实现了一个简单的图级嵌入生成器，使用加权平均方法将图中所有有效节点的嵌入聚合成一个全局嵌入，并进行了归一化。它通常用于图神经网络中，以生成图级别的特征表示。"""
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, emb, mask=None):
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.expand((vsum.shape[1], row_sum.shape[0])).T
        global_emb = vsum / row_sum 
        return F.normalize(global_emb, p=2, dim=1) 
    
class Encoder(Module):
    def __init__(self, in_features, out_features, graph_neigh,num_clusters,device,adj_template,dropout=0.0, act=F.relu):
        """
            in_features : 输入特征的维度。
            out_features: 输出特征的维度，表示经过该层处理后节点嵌入的维度。
            graph_neigh : 图的邻接矩阵，通常用来表示图中节点间的连接关系。
            dropout     :dropout 概率，用于正则化，防止过拟合。
            act         :激活函数，默认为 ReLU。

        """
        super(Encoder, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
       
        self.dropout = dropout
        self.num_clusters = num_clusters
        self.act = act
        self.temperature = 0.5
        self.count = 0
        self.device = device
        self.adj_template = adj_template
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))#定义两个可训练参数,并指出形状
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()#该方法用于初始化模型中的权重，采用 Xavier 均匀分布初始化方法：
        
        self.disc = Discriminator(self.out_features)
        
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()


      
        self.kwargs = {
            'metric': 'cosine',  # euclidean if not l2_normalize
            'distributed': False,
            'random_state': 41,
            'n_clusters': self.num_clusters,
            'verbose': False
}
        self.gmm = PyTorchGaussianMixture(covariance_type='diag', reg_covar=1e-6, init='k-means++', **self.kwargs)
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)


    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.mm(adj, z)
       
        hiden_emb = z #图卷积和线性变换后的节点嵌入。
        
        h = torch.mm(z, self.weight2) #通过 weight2 将节点嵌入从 out_features 映射回 in_features。
        h = torch.mm(adj, h) #使用邻接矩阵 adj 对新的节点特征 h 进行加权求和，得到第二次图卷积后的结果
      
        emb = self.act(z)
      
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.mm(adj, z_a)
        emb_a = self.act(z_a)

        if self.count >= 100 and self.count % 15 == 0:  # 100/200
            adj_drop = drop_edges_from_adj(self.adj_template)
            #adj_drop = self.adj_template
            adj_drop = torch.FloatTensor(adj_drop).to(self.device)
            z_b = F.dropout(feat,self.dropout,self.training)
            z_b = torch.mm(z_b,self.weight1)
            z_b = torch.mm(adj_drop,z_b)
            with torch.no_grad(): 
                z_clu = self.gmm.fit_predict(z)     # (N x K) soft labels
                z_clu = torch.argmax(z_clu, dim=1)
            
            z_center = compute_centers(z,z_clu,self.num_clusters)
            z_b_center = compute_centers(z_b,z_clu,self.num_clusters)
            clustering_loss = compute_cluster_losss(z_center,z_b_center,self.temperature,z_clu,self.num_clusters)

        else:
            clustering_loss = torch.tensor(0.0) 
        self.count += 1

        g = self.read(emb, self.graph_neigh) 
        g = self.sigm(g)  

        g_a = self.read(emb_a, self.graph_neigh)
        g_a = self.sigm(g_a)  

        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb) 
        return hiden_emb, h, ret, ret_a,clustering_loss
    
class Trajectory(Module):

    def __init__(self, in_features, out_features, graph_neigh,num_clusters,device,adj_template,dropout=0.0, act=F.relu):
        super(Trajectory, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.graph_neigh = graph_neigh
        self.dropout = dropout
        self.act = act
        self.num_clusters = num_clusters
        self.adj_template = adj_template
        self.device = device
        self.temperature = 0.5
        self.weight1 = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.weight2 = Parameter(torch.FloatTensor(self.out_features, self.in_features))
        self.reset_parameters()
        
        self.disc = Discriminator(self.out_features)
        self.count = 0
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

        self.kwargs = {
            'metric': 'cosine',  # euclidean if not l2_normalize
            'distributed': False,
            'random_state': 41,
            'n_clusters': self.num_clusters,
            'verbose': False
}
        self.gmm = PyTorchGaussianMixture(covariance_type='diag', reg_covar=1e-6, init='k-means++', **self.kwargs)
        
    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight1)
        torch.nn.init.xavier_uniform_(self.weight2)

    def forward(self, feat, feat_a, adj):
        z = F.dropout(feat, self.dropout, self.training)
        z = torch.mm(z, self.weight1)
        z = torch.spmm(adj, z)
        
        hiden_emb = z
        
        h = torch.mm(z, self.weight2)
        h = torch.spmm(adj, h)
        
        emb = self.act(z)
        
        z_a = F.dropout(feat_a, self.dropout, self.training)
        z_a = torch.mm(z_a, self.weight1)
        z_a = torch.spmm(adj, z_a)
        emb_a = self.act(z_a)
        
        if self.count >= 20 and self.count % 15 == 0:                 #traj : 20/15
            adj_template_dense = self.adj_template.todense()
            adj_drop = drop_edges_from_adj_niche(adj_template_dense)
            adj_drop = torch.FloatTensor(adj_drop).to(self.device)
            
            z_a_c = F.dropout(feat,self.dropout,self.training)
            z_a_c = torch.mm(z_a_c,self.weight1)
            z_a_c = torch.spmm(adj_drop,z_a_c)
            with torch.no_grad(): 
                z_clu = self.gmm.fit_predict(z)     # (N x K) soft labels
                z_clu = torch.argmax(z_clu, dim=1)
            
            z_center = compute_centers(z,z_clu,self.num_clusters)
            z_a_c_center = compute_centers(z_a_c,z_clu,self.num_clusters)
            clustering_loss = compute_cluster_losss(z_center,z_a_c_center,self.temperature,z_clu,self.num_clusters)

        else:
            clustering_loss = torch.tensor(0.0) 
        self.count += 1


        g = self.read(emb, self.graph_neigh)
        g = self.sigm(g)
        
        g_a = self.read(emb_a, self.graph_neigh)
        g_a =self.sigm(g_a)       
       
        ret = self.disc(g, emb, emb_a)  
        ret_a = self.disc(g_a, emb_a, emb)
        
        return hiden_emb, h, ret, ret_a,clustering_loss
    