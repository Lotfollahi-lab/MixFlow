from geomloss import SamplesLoss
import sys
import anndata as ad
import scanpy as sc
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr
import torch
from pykeops.torch import generic_logsumexp

from . import metric_MMD


def W1(x, y):
    loss_fn = SamplesLoss(loss="sinkhorn", p=1, blur=0.01, backend="tensorized")
    return loss_fn(x, y).item()

def W2(x, y):
    loss_fn = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, backend="tensorized")
    return loss_fn(x, y).item()

def MMD(x, y):
    loss_fn = SamplesLoss(loss="gaussian", blur=0.5, backend="tensorized")
    return loss_fn(x, y).item()


def scanpy_preprocessing(adata):
    adata = adata.copy()
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)  # Library size normalization
    sc.pp.log1p(adata)  # Log-transform
    sc.pp.highly_variable_genes(adata, n_top_genes=2000)
    return adata

def scanpy_pca(adata):
    adata = adata.copy() # Create a copy to avoid modifying the original AnnData
    sc.pp.pca(adata, n_comps=2)
    return adata


def W1_complete(x, y, preprocess = False):
    if preprocess:
        x = scanpy_preprocessing(x)
        y = scanpy_preprocessing(y)

    x_reduced = scanpy_pca(x)
    y_reduced = scanpy_pca(y)

    # Extract PCA embeddings and convert to torch tensors
    x_pca = torch.tensor(x_reduced.obsm['X_pca'], dtype=torch.float32)
    y_pca = torch.tensor(y_reduced.obsm['X_pca'], dtype=torch.float32)
    
    return W1(x_pca, y_pca)


def W2_complete(x, y, preprocess = False):
    if preprocess:
        x = scanpy_preprocessing(x)
        y = scanpy_preprocessing(y)

    x_reduced = scanpy_pca(x)
    y_reduced = scanpy_pca(y)

    # Extract PCA embeddings and convert to torch tensors
    x_pca = torch.tensor(x_reduced.obsm['X_pca'], dtype=torch.float32)
    y_pca = torch.tensor(y_reduced.obsm['X_pca'], dtype=torch.float32)
    
    return W2(x_pca, y_pca)


def MMD_complete(x, y, preprocess = False):
    if preprocess:
        x = scanpy_preprocessing(x)
        y = scanpy_preprocessing(y)

    x_reduced = scanpy_pca(x)
    y_reduced = scanpy_pca(y)

    # Extract PCA embeddings and convert to torch tensors
    x_pca = torch.tensor(x_reduced.obsm['X_pca'], dtype=torch.float32)
    y_pca = torch.tensor(y_reduced.obsm['X_pca'], dtype=torch.float32)
    
    return MMD(x_pca, y_pca)


def get_deg_genes(
    adata: ad.AnnData,
    groupby: str = "condition_ID",
    method: str = "wilcoxon",
    alpha: float = 0.05
):
    sc.tl.rank_genes_groups(
        adata,
        groupby=groupby,
        method=method,
        use_raw=False,
        n_genes=adata.shape[1]
    )
    
    degs = set()
    # The rank_genes_groups results are stored in adata.uns['rank_genes_groups']
    rg_results = adata.uns["rank_genes_groups"]
    
    # Each group is a key in rg_results["names"].dtype.names
    for group in rg_results["names"].dtype.names:
        pvals_adj = rg_results["pvals_adj"][group]
        genes = rg_results["names"][group]
        
        for gene, pval in zip(genes, pvals_adj):
            if pval < alpha:
                degs.add(gene)
    
    return degs


def get_avg_expression(
    adata: ad.AnnData,
    genes: set
) -> pd.Series:
    """
    Computes the average (mean) expression for the specified genes
    across all cells in the AnnData object.
    Returns a Series of average expression, indexed by gene.
    """
    # Intersect the requested gene set with what is actually in the data
    common_genes = list(set(adata.var_names).intersection(genes))
    if len(common_genes) == 0:
        return pd.Series(dtype=float)
    
    # Subset the adata to the common genes
    sub_adata = adata[:, common_genes]
    
    # sub_adata.X: shape (n_cells, n_common_genes)
    # Take the mean expression across cells for each gene
    avg_exp = np.array(sub_adata.X.mean(axis=0)).ravel()
    
    return pd.Series(data=avg_exp, index=common_genes)


def pearson_dict(x, y):
    # Get the common keys
    common_keys = set(x.keys()).intersection(y.keys())

    # Extract values corresponding to the common keys
    true_values = [x[key] for key in common_keys]
    calculated_values = [y[key] for key in common_keys]

    # Calculate Pearson correlation
    correlation, _ = pearsonr(true_values, calculated_values)
    
    return correlation

def spearman_dict(x, y):
    # Get the common keys
    common_keys = set(x.keys()).intersection(y.keys())

    # Extract values corresponding to the common keys
    true_values = [x[key] for key in common_keys]
    calculated_values = [y[key] for key in common_keys]

    # Calculate Pearson correlation
    correlation, _ = spearmanr(true_values, calculated_values)
    
    return correlation

def mse_dict(x, y):
    common_keys = set(x.keys()).intersection(y.keys())

    # Extract values corresponding to the common keys
    true_values = np.array([x[key] for key in common_keys])
    calculated_values = np.array([y[key] for key in common_keys])

    # Calculate Mean Squared Error
    mse = np.mean((true_values - calculated_values) ** 2)

    return mse

def compute_pearson(x, y):
    # Ensure both Series share the same index of overlapping genes
    common_genes = x.index.intersection(y.index)
    
    if len(common_genes) == 0:
        return float('nan')
    
    x_vals = x.loc[common_genes].values
    y_vals = y.loc[common_genes].values
    
    pearson_corr, _ = pearsonr(x_vals, y_vals)
    
    return pearson_corr

def compute_spearman(x, y):
    # Ensure both Series share the same index of overlapping genes
    common_genes = x.index.intersection(y.index)
    
    if len(common_genes) == 0:
        return float('nan')
    
    x_vals = x.loc[common_genes].values
    y_vals = y.loc[common_genes].values
    
    spearman_corr, _ = spearmanr(x_vals, y_vals)
    
    return spearman_corr


def pearson_complete(x, y, deg=True, preprocess=False, alpha=0.05, condition_id="condition_ID", method="wilcoxon"):
    """
    Calculates the Pearson correlation between the average expression of genes
    in two AnnData objects. Can use either differentially expressed genes (DEGs)
    or all genes.

    Parameters:
    x (ad.AnnData): First AnnData object.
    y (ad.AnnData): Second AnnData object.
    deg (bool): Whether to use DEGs (True) or all genes (False). Defaults to True.
    preprocess (bool): Whether to apply scanpy preprocessing. Defaults to False.
    alpha (float): Significance level for DEG selection (if deg=True). Defaults to 0.05.
    condition_id (str): Column in .obs representing the condition for DEG calculation (if deg=True). Defaults to "condition_ID".
    method (str): Method for DEG calculation (if deg=True), passed to sc.tl.rank_genes_groups. Defaults to "wilcoxon".

    Returns:
    float: Pearson correlation coefficient. Returns NaN if no common genes are found.
    """
    if preprocess:
        x = scanpy_preprocessing(x)
        y = scanpy_preprocessing(y)

    if deg:
        # Calculate correlation based on DEGs
        x_genes = get_deg_genes(x, groupby=condition_id, method=method, alpha=alpha)
        y_genes = get_deg_genes(y, groupby=condition_id, method=method, alpha=alpha)
    else:
        # Use all genes
        x_genes = set(x.var_names)
        y_genes = set(y.var_names)

    x_avg_exp = get_avg_expression(x, x_genes)
    y_avg_exp = get_avg_expression(y, y_genes)

    return compute_pearson(x_avg_exp, y_avg_exp)


def spearman_complete(x, y, deg=True, preprocess=False, alpha=0.05, condition_id="condition_ID", method="wilcoxon"):
    """
    Calculates the Spearman correlation between the average expression of genes
    in two AnnData objects. Can use either differentially expressed genes (DEGs)
    or all genes.

    Parameters:
    x (ad.AnnData): First AnnData object.
    y (ad.AnnData): Second AnnData object.
    deg (bool): Whether to use DEGs (True) or all genes (False). Defaults to True.
    preprocess (bool): Whether to apply scanpy preprocessing. Defaults to False.
    alpha (float): Significance level for DEG selection (if deg=True). Defaults to 0.05.
    condition_id (str): Column in .obs representing the condition for DEG calculation (if deg=True). Defaults to "condition_ID".
    method (str): Method for DEG calculation (if deg=True), passed to sc.tl.rank_genes_groups. Defaults to "wilcoxon".

    Returns:
    float: Spearman correlation coefficient. Returns NaN if no common genes are found.
    """
    if preprocess:
        x = scanpy_preprocessing(x)
        y = scanpy_preprocessing(y)

    if deg:
      # Calculate correlation based on DEGs
      x_genes = get_deg_genes(x, groupby=condition_id, method=method, alpha=alpha)
      y_genes = get_deg_genes(y, groupby=condition_id, method=method, alpha=alpha)
    else:
      # Use all genes
      x_genes = set(x.var_names)
      y_genes = set(y.var_names)

    x_avg_exp = get_avg_expression(x, x_genes)
    y_avg_exp = get_avg_expression(y, y_genes)

    return compute_spearman(x_avg_exp, y_avg_exp)

def compute_metrics(original_data, generated_data, metric_fn):
    metric_funcs = {
        'w1': SamplesLoss(loss="sinkhorn", p=1, blur=0.01),
        'w2': SamplesLoss(loss="sinkhorn", p=2, blur=0.01),
        'mmd': metric_MMD.iface_compute_MMD,  # SamplesLoss(loss="gaussian", blur=0.5),
        'energy': SamplesLoss(loss="energy", blur=0.5),
    }
    metric_fn = metric_funcs[metric_fn]
    original_data = torch.tensor(original_data)
    generated_data = torch.tensor(generated_data)
    metric = metric_fn(generated_data, original_data)
    return metric.item()
