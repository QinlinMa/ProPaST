import numpy as np
import pandas as pd
from sklearn import metrics
import scanpy as sc
import ot
from sklearn.decomposition import PCA

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(adata, n_clusters=7, radius=50, method='mclust', start=0.1, end=3.0, increment=0.01, refinement=False,pca_dim=20):

    n_components = min(pca_dim,adata.obsm['emb'].shape[1])
    pca = PCA(n_components=n_components, random_state=42) 
    embedding = pca.fit_transform(adata.obsm['emb'].copy())
    adata.obsm['emb_pca'] = embedding
    
    if method == 'mclust':
       adata = mclust_R(adata, used_obsm='emb_pca', num_cluster=n_clusters)
       adata.obs['domain'] = adata.obs['mclust']
    elif method == 'leiden':
       res = search_res_binary(adata, n_clusters, use_rep='emb_pca', method=method, start=start, end=end)
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs['domain'] = adata.obs['leiden']
    if refinement:  
       new_type = refine_label(adata, radius, key='domain')
       adata.obs['domain'] = new_type 
       
def refine_label(adata, radius=50, key='label'):
    n_neigh = radius
    new_type = []
    old_type = adata.obs[key].values
    
    #calculate distance
    position = adata.obsm['spatial']
    distance = ot.dist(position, position, metric='euclidean')
           
    n_cell = distance.shape[0]
    
    for i in range(n_cell):
        vec  = distance[i, :]
        index = vec.argsort()
        neigh_type = []
        for j in range(1, n_neigh+1):
            neigh_type.append(old_type[index[j]])
        max_type = max(neigh_type, key=neigh_type.count)
        new_type.append(max_type)
        
    new_type = [str(i) for i in list(new_type)]     
    return new_type
 

def search_res_binary(adata, n_clusters, use_rep='emb', 
                      start=0.1, end=3.0, tol=0.001, max_iter=50):
    print('Searching resolution with binary search...')
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)

    low, high = start, end
    best_res = None
    found_exact = False

    for i in range(max_iter):
        mid = (low + high) / 2

     
        sc.tl.leiden(adata, random_state=0, resolution=mid)
        cluster_count = adata.obs['leiden'].nunique()


        print(f"Iter {i+1}: res={mid:.4f}, clusters={cluster_count}")

        if cluster_count == n_clusters:
            best_res = mid
            found_exact = True
            break
        elif cluster_count > n_clusters:
            high = mid  
        else:
            low = mid   

        best_res = mid

        if abs(high - low) < tol:
            break

    if not found_exact:
        print(f"Warning: exact match not found, best approx res={best_res:.4f}")

    return best_res












