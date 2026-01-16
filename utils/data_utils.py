

from typing import List
import pickle
import scanpy as sc
import re
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.decomposition import PCA
import scipy
from scipy.sparse import issparse
import anndata


def get_visinfo_subpopulations_pertAndcovars(
        adata:anndata.AnnData,
        str_perturbationkey:str,
        list_covarname:List[str]
    ):
    """
    Given an anndata, the pert column and list of covarnames, considers all case of conditioning on covariates (2 ^ |covariates| cases) and returns the info/crosstab for each case.
    returns
    - a dict mapping each 'covar-selections strID' to the size of the popuation
    - a dict mapping each 'covar-selections strID' to a dataframe with unique rows, and the size of subpopulations as an additional column, ready to be dumped as, e.g., html.
    """
    # generates all binary strings of lenght 'n'
    def getallbin(n):
        if n == 1:
            return ['0','1']

        result_nmin1 = getallbin(n-1)
        toret = ['0'+u for u in result_nmin1] + ['1'+u for u in result_nmin1]
        return toret

    list_str_covarsinclusion = getallbin(len(list_covarname))
    
    dict_conditioningstrID_to_df_subpopulation = {}

    for str_covarsinclusion in list_str_covarsinclusion:  # loops over different cases of inclusion of covariates
        
        # create `list_colnames_to_condition_on`
        list_colnames_to_condition_on = [str_perturbationkey]
        for idx_bit, bit in enumerate(str_covarsinclusion):
            if bit=='1':
                list_colnames_to_condition_on.append(list_covarname[idx_bit])
        
        # 
        df_subpopulation = adata.obs.groupby(list_colnames_to_condition_on, observed=True).size().reset_index().rename(columns={0:'size of subpopulation'})
        df_subpopulation = df_subpopulation[
            df_subpopulation['size of subpopulation'] > 0
        ]  # to drop combinations of columns with 0 frequency, although `observed` is passed as Ture and this is done only to safe-guard.

        assert df_subpopulation['size of subpopulation'].sum() == adata.shape[0], print("Not true for {}. Maybe there are nan-s in that column of the anndata object?".format(list_colnames_to_condition_on))
        
        dict_conditioningstrID_to_df_subpopulation[
            " /\ ".join(list_colnames_to_condition_on)
        ] = df_subpopulation.copy()
    
    return dict_conditioningstrID_to_df_subpopulation

    

def get_embeddings(perturbation_data, column_pert, ctrl, embedding_path, sep="+"):
    sep_checked = "".join(["\\"+s for s in sep])
    pert_col = [re.split(sep_checked, pert) for pert in perturbation_data.obs[column_pert].values]
    perturbations = set([p for pert in pert_col for p in pert if p!=ctrl])
    with open(embedding_path, 'rb') as file:
        data = pickle.load(file)
    perturbation_embedding_dict = {}
    for key in perturbations:
        if key in data:
            perturbation_embedding_dict[key.replace("/", "")] = data[key] if type(data[key])==type(list()) else data[key].reshape(-1).tolist()
        else:
            idx = [key not in pert for pert in pert_col]
            pert_col = [item for item, keep in zip(pert_col, idx) if keep]
            perturbation_data = perturbation_data[idx]
            print(f"{key} not found in dict")
            #raise ValueError(f"{key} not found in dict")
    
    perturbation_data.uns[column_pert] = perturbation_embedding_dict
    return perturbation_data


def adata_preprocessing_ak(adata):

    """
    Assumptions:
    - adata.X and adata.layers['counts'] contain raw counts
    - This function does HVG selection and puts log1p transformed X in anndata.X.
    """
    

    # read the anndata
    assert isinstance(adata, anndata.AnnData)

    # make sure that both adata.X and adata.layers['counts'] contain raw counts.
    assert issparse(adata.X)
    assert np.allclose(adata.X.data, np.floor(adata.X.data))  # i.e. all integers
    assert issparse(adata.layers['counts'])
    assert np.allclose(adata.layers['counts'].data, np.floor(adata.layers['counts'].data))  # i.e. all integers.
    assert np.allclose(
        adata.X.data,
        adata.layers['counts'].data
    )  # i.e. counts are in adata.X.

    # filter cells/genes
    sc.pp.filter_cells(adata, min_genes=200)
    sc.pp.filter_genes(adata, min_cells=3)

    # row normalise, then log tfm
    sc.pp.normalize_total(adata, inplace=True, target_sum=1e4)
    adata_HVG = adata.copy()
    sc.pp.log1p(adata, copy=False)  # adata_HVG.X contains the raw counts.
    
    # compute HVGs
    sc.pp.highly_variable_genes(adata_HVG, flavor='seurat_v3', n_top_genes=2000)
    idx = adata_HVG.var["highly_variable"].copy()

    # filter based on HVGs
    adata = adata[:, idx]


    # # row normalise, then log normalise.
    # sc.pp.normalize_total(adata, inplace=True, target_sum=1e4)
    # sc.pp.log1p(adata, copy=False)  Moved to above, because as if it's the common practice to select HVG
    
    return adata


def adata_preprocessing(adata):
    raise Exception("Usage temporarily disabled.")

    if type(adata) == str:
        original_data = sc.read_h5ad(adata)
    elif type(adata) == sc.AnnData:
        original_data = adata
    else:
        raise ValueError("adata should be a path to an h5ad file or a scanpy AnnData object")
    
    original_data.layers["lognormal"] = original_data.X.copy()  # assumption: adata.X contains lognormalised counts.
    
    # put raw counts in X and do preprocessing 
    original_data.X = original_data.layers["counts"].copy()  # the other assumption: layers['counts'] contains the raw counts.
    sc.pp.filter_cells(original_data, min_genes=200)
    sc.pp.filter_genes(original_data, min_cells=3)

    sc.pp.highly_variable_genes(original_data, flavor='seurat_v3', n_top_genes=2000)
    idx = original_data.var["highly_variable"]
    original_data = original_data[:, idx]

    # put log-normalised values again in adata.X
    original_data.X = original_data.layers["lognormal"].copy()
    return original_data

def plot_data(training_data, original_data, generated_data, perturbation_column_original, initial_samples, plot_modes=False, input_modes=None, model_modes=None, output_modes=None):
    pca = PCA(n_components=2)
    
    X_pca_ref = pca.fit_transform(original_data.X.toarray() if hasattr(original_data.X, "toarray") else original_data.X)
    X_pca_query = pca.transform(generated_data.X.toarray() if hasattr(generated_data.X, "toarray") else generated_data.X)
    X_pca_train = pca.transform(training_data.X.toarray() if hasattr(training_data.X, "toarray") else training_data.X)
    X_pca_initial_samples = pca.transform(initial_samples.X.toarray() if hasattr(initial_samples.X, "toarray") else initial_samples.X)

    if plot_modes:
        row1 = output_modes['mean'][0]
        row2 = output_modes['mean'][1]

        plt.figure(figsize=(8, 5))

        sns.kdeplot(row1, label=f'Distribution Mode PC1', linewidth=5)
        sns.kdeplot(row2, label=f'Distribution Mode PC2', linewidth=3)
    
        plt.legend()
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.title('Distribution of Modes')
        plt.show()

    fig, axes = plt.subplots(4, 1, figsize=(50, 35), sharex=True, sharey=True)

    categories = training_data.obs[perturbation_column_original].astype("category").cat.categories
    palette = dict(zip(categories, sns.color_palette("tab10", len(categories))))

    ax = axes[0]
    for cat in categories:
        mask = training_data.obs[perturbation_column_original] == cat
        ax.scatter(X_pca_train[mask, 0], X_pca_train[mask, 1], c=[palette[cat]], label=cat, alpha=0.7, s=10)
    
    if plot_modes:
        for i in range(len(input_modes["mean"])):
            mean = input_modes["mean"][i][:2]  
            var = input_modes["var"][i][:2]    

            width, height = 2 * np.sqrt(var)  
            ellipse = patches.Ellipse(xy=mean,
                                    width=width, height=height,
                                    angle=0, edgecolor='black',
                                    facecolor='none', linestyle='--', linewidth=3)
            ax.add_patch(ellipse)

        for i in range(len(input_modes["mean"])):
            mean = model_modes['mean'][i][:2]      
            var = model_modes['var'][i][:2]        

            width, height = 2 * np.sqrt(var)
            ellipse = patches.Ellipse(xy=mean,
                                    width=width, height=height,
                                    angle=0, edgecolor='red',
                                    facecolor='none', linestyle='--', linewidth=2)
            ax.add_patch(ellipse)

    ax.set_title("Training Dataset PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    categories = original_data.obs[perturbation_column_original].astype("category").cat.categories
    common_palette = dict(zip(categories, sns.color_palette("tab10", len(categories))))

    # Plot reference dataset
    ax = axes[1]
    for cat in categories:
        mask = original_data.obs[perturbation_column_original] == cat
        ax.scatter(X_pca_ref[mask, 0], X_pca_ref[mask, 1], c=[common_palette[cat]], label=cat, alpha=0.7, s=10)
    
    if plot_modes:
        for i in range(len(input_modes["mean"])):
            mean = input_modes["mean"][i][:2]  
            var = input_modes["var"][i][:2]    

            width, height = 2 * np.sqrt(var)  
            ellipse = patches.Ellipse(xy=mean,
                                    width=width, height=height,
                                    angle=0, edgecolor='black',
                                    facecolor='none', linestyle='--', linewidth=3)
            ax.add_patch(ellipse)

        for i in range(len(input_modes["mean"])):
            mean = model_modes['mean'][i][:2]      
            var = model_modes['var'][i][:2]        

            width, height = 2 * np.sqrt(var)
            ellipse = patches.Ellipse(xy=mean,
                                    width=width, height=height,
                                    angle=0, edgecolor='red',
                                    facecolor='none', linestyle='--', linewidth=2)
            ax.add_patch(ellipse)

    ax.set_title("Reference Dataset PCA")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

    # Plot query dataset
    ax = axes[2]
    for cat in categories:
        mask = generated_data.obs['perturbation'] == cat
        ax.scatter(X_pca_initial_samples[mask, 0], X_pca_initial_samples[mask, 1], c=[common_palette[cat]], label=cat, alpha=0.7, s=10)
    ax.set_title("Initial Samples PCA")
    ax.set_xlabel("PC1")

    if plot_modes:
        group_vars = generated_data.obs.columns.tolist()
        group_keys = generated_data.obs[group_vars].astype(str).agg('_'.join, axis=1)

        unique_groups = group_keys.unique()
        n_groups = len(unique_groups)
        palette = sns.color_palette('husl', n_groups)  # 'tab20', 'Set3', 'Paired'
        colors = {g: palette[i] for i, g in enumerate(unique_groups)}
        
        seen = set()
        for i, g in enumerate(unique_groups):
            if g in seen:
                continue
            seen.add(g)

            mean = output_modes['mean'][i][:2]
            var = output_modes['var'][i][:2]

            width, height = 2 * np.sqrt(var)  
            ellipse = patches.Ellipse(xy=mean,
                                    width=width, height=height,
                                    angle=0,
                                    edgecolor='black',
                                    facecolor=colors[g],
                                    linestyle='--', linewidth=2)
            ax.add_patch(ellipse)

     # Plot query dataset
    ax = axes[3]
    for cat in categories:
        mask = generated_data.obs['perturbation'] == cat
        ax.scatter(X_pca_query[mask, 0], X_pca_query[mask, 1], c=[common_palette[cat]], label=cat, alpha=0.7, s=10)
    ax.set_title("Query Dataset PCA")
    ax.set_xlabel("PC1")

    if plot_modes:
        group_vars = generated_data.obs.columns.tolist()
        group_keys = generated_data.obs[group_vars].astype(str).agg('_'.join, axis=1)

        unique_groups = group_keys.unique()
        n_groups = len(unique_groups)
        palette = sns.color_palette('husl', n_groups)  # 'tab20', 'Set3', 'Paired'
        colors = {g: palette[i] for i, g in enumerate(unique_groups)}
        
        seen = set()
        for i, g in enumerate(unique_groups):
            if g in seen:
                continue
            seen.add(g)

            mean = output_modes['mean'][i][:2]
            var = output_modes['var'][i][:2]

            width, height = 2 * np.sqrt(var)  
            ellipse = patches.Ellipse(xy=mean,
                                    width=width, height=height,
                                    angle=0,
                                    alpha=0.2,
                                    edgecolor='black',
                                    facecolor=colors[g],
                                    linestyle='--', linewidth=2)
            ax.add_patch(ellipse)

    plt.tight_layout()
    plt.show()

    # ---- Create a single legend ----
    fig_legend, ax_legend = plt.subplots(figsize=(10, 1 * len(cat) // 3))  
    ax_legend.axis("off")  
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=common_palette[cat], markersize=8, label=cat) 
            for cat in categories]

    ax_legend.legend(handles=handles, loc="center", fontsize=10, title="Legend", frameon=False, ncol=3) 
    plt.show()
    
def compute_condition_means(adata, condition_name):
    """
    Computes the mean expression value for each unique condition in the AnnData object.

    For each unique condition in adata.obs[condition_name], it calculates a single
    mean expression value averaged over all cells belonging to that condition and all genes.

    Args:
        adata (anndata.AnnData): AnnData object containing gene expression data and condition annotations.
        condition_name (str): Name of the column in adata.obs that contains condition labels.

    Returns:
        dict: A dictionary where keys are unique conditions and values are their corresponding mean expression values.
            Returns an empty dictionary if there are issues or no conditions found.
    """
    condition_means = {}
    if condition_name not in adata.obs.columns:
        print(f"Warning: Condition column '{condition_name}' not found in AnnData object.")
        return {}

    conditions = adata.obs[condition_name].unique()
    if not conditions.size:  # Check if conditions is empty
        print(f"Warning: No unique conditions found in column '{condition_name}'.")
        return {}

    for cond in conditions:
        mask = adata.obs[condition_name] == cond
        expr_data = adata.X[mask]

        if expr_data.shape[0] == 0:  # Handle cases where a condition has no cells
            print(f"Warning: No cells found for condition '{cond}'. Mean set to NaN.")
            condition_means[cond] = np.nan  # Set mean to NaN for conditions with no cells
            continue

        # Convert sparse to dense if needed
        if scipy.sparse.issparse(expr_data):
            expr_data = expr_data.toarray()
        mean_val = np.nanmean(expr_data, axis=0) # Use nanmean to handle potential NaNs in data
        condition_means[cond] = mean_val
    return condition_means

def get_params(model, param_names, detach=False, strict=True, return_dict=False):
    found = {}
    param_dict = dict(model.named_parameters())

    for name in param_names:
        if name in param_dict:
            param = param_dict[name]
            if detach:
                param = param.detach()
            found[name] = param
        else:
            if strict:
                raise ValueError(f"Parameter {name} not found in model.")

    if return_dict:
        return found
    else:
        return list(found.values())
    
    