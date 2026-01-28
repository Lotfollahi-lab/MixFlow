import numpy as np
import anndata
import pickle
import sys
import scanpy as sc
import torch
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader, Dataset
from tensordict import TensorDict
from scipy.sparse import issparse
import re

import cuml
import cupy as cp


class AkPCA:
    """
    PCA transformation, but each component is zero-centered and is divided by variance of expression.
    """

    def __init__(self, n_components, flag_rapidssc_pca):

        # grab args
        assert isinstance(n_components, int)
        assert isinstance(flag_rapidssc_pca, bool)
        assert flag_rapidssc_pca in [True, False]
        
        self.n_components = n_components
        self.flag_rapidssc_pca = flag_rapidssc_pca


        if flag_rapidssc_pca:
            self.pca = cuml.decomposition.PCA(n_components=n_components)
        else:
            self.pca = PCA(n_components=n_components)

        self.EPS_std = 1e-10
    
    def fit(self, X):
        # fit PCA
        if self.flag_rapidssc_pca:
            assert isinstance(X, np.ndarray)
            self.pca.fit(
                cp.array(
                    X + 0.0,
                    dtype=np.float32
                )
            )

        else:
            self.pca.fit(X)

        # update the mean/std of each component
        if self.flag_rapidssc_pca:
            Xenc = self.pca.transform(
                cp.array(
                    X + 0.0,
                    dtype=np.float32
                )
            )  # [N x num_components]
            
            Xenc = cp.asnumpy(Xenc)
            
        else:
            Xenc = self.pca.transform(X)  # [N x num_components]
        
        
        assert Xenc.shape[1] == self.pca.n_components_
        self.mean = np.expand_dims(
            (np.min(Xenc, 0) + np.max(Xenc, 0)) / 2.0,
            0
        )  # [1 x num_components]

        self.std = np.expand_dims(
            np.max(Xenc, 0) - np.min(Xenc, 0),
            0
        )/2.0  # [1 x num_components]
        
    
    def transform(self, X):
        if self.flag_rapidssc_pca:
            output = self.pca.transform(
                cp.array(
                    X + 0.0,
                    dtype=np.float32
                )
            )  # [N x num_components]
            output = cp.asnumpy(output)
            
        else:
            output = self.pca.transform(X)  # [N x num_components]
        
        output = (output - self.mean) / (self.std)  # so it's exactly  between -1.0 and +1.0.
        return output
    
    def inverse_transform(self, encoded_X: np.ndarray | torch.Tensor):
        assert isinstance(encoded_X, np.ndarray) or isinstance(encoded_X, torch.Tensor)
        output = encoded_X if(isinstance(encoded_X, np.ndarray)) else encoded_X.detach().cpu().numpy()  # [N x num_PCs]
        output = output * self.std  # [N x num_PCs]
        output = output + self.mean  # [N x num_PCs]

        if self.flag_rapidssc_pca:
            output = self.pca.inverse_transform(
                cp.array(
                    output + 0.0,
                    dtype=np.float32
                )
            )
            output = cp.asnumpy(output)  # [N x original_D]

        else:
            output = self.pca.inverse_transform(output)  # [N x original_D]
        
        return output

class GeneExpression(Dataset):
    def __init__(
        self,
        logger,
        file_path,
        perturbation_key,
        condition_keys,
        dosage_key,
        flag_rapidssc_pca,
        num_pca_components=None,
        size_embedding=10,
        control="ctrl",
        sep="+",
        split_key=None,
        genetic_perturbation=False,
        delta=False,
        device="cpu"
    ):
        self.condition_keys = condition_keys  # TODO:note:added here in case the custom sampler for the DL needs it.
        self.adata = sc.read_h5ad(file_path)


        assert (
            "counts" in self.adata.layers.keys()
        ), "Counts layer not found in AnnData object. Please provide a counts layer."
        assert (
            perturbation_key in self.adata.obs.keys()
        ), f"Perturbation key {perturbation_key} not found in AnnData object."
        if dosage_key is not None:
            assert (
                dosage_key in self.adata.obs.keys()
            ), f"Dosage key {dosage_key} not found in AnnData object."
        assert (
            control in self.adata.obs[perturbation_key].unique()
        ), f"Control perturbation {control} not found in perturbation key {perturbation_key}."
        if condition_keys is not None:
            assert type(condition_keys) == list, "Condition keys must be a list."
            assert all(
                [key in self.adata.obs.keys() for key in condition_keys]
            ), f"Condition keys {condition_keys} not found in AnnData object."

        self.perturbation_key = perturbation_key
        self.control = control
        self.device = 'cpu' #device  # TODO:note: should I uncomment? It remains 'cpu' in the end as well.
        logger.info(
            f"Loaded dataset with {self.adata.n_obs} cells and {self.adata.n_vars} genes."
        )
        logger.info(f"Extracting and preprocessing gene expression data...")

        # TODO:note: I commented out this part, because the function `adata_preprocessing` already does it and for the combined (i.e. train+test) anndata. Notabley it will collide with HVG removal, etc.
        ''' 
        self.adata.X = self.adata.layers["counts"].copy()
        logger.info("Normalizing the data...")
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        logger.info("Log1p transforming the data...")
        sc.pp.log1p(self.adata)
        '''

        self.assert_no_slashAndbackslash_in_perts_and_covars()

        if split_key is not None:
            logger.info(f"Splitting the dataset using key {split_key}...")
            self.adata_train = self.adata[self.adata.obs[split_key] == "train"].copy()
            self.adata_test = self.adata[self.adata.obs[split_key] == "test"].copy()

        else:
            self.adata_train = self.adata.copy()
            self.adata_test = None
        
        #TODO:note:BUG if delta is False, then `self.expression_data` is never assigned.
        if delta: 
            logger.info(f"Subtracting control mean from the data...")
            if "cell_type" in self.adata.obs.columns:
                cell_types = self.adata.obs.cell_type.unique()
            else:
                cell_types = ["cell_type"]
            
            if issparse(self.adata_train.X):
                self.expression_data = self.adata_train.X.toarray().copy()
            else:
                self.expression_data = self.adata_train.X.copy()
            
            adata_akevaluation = self.adata_test.copy()  # test anndata, but similar 'mean subtraction' is performed on it.
            adata_aktrain4eval = self.adata_train.copy()  # to be used only for evaluation            

            self.dict_celltype_to_ctrlmean_trainingset = {}  
            # TODO:note: so if during evaluation a celltype has been seen during training, the same `ctrl_mean` is added to the generated diff expression vector.
             
            for cell_type in cell_types:
                if len(cell_types) > 1:
                    idx = (self.adata_train.obs.cell_type == cell_type).tolist()  # i.e. subsampling based on cell types really happens.
                    idx_test = (self.adata_test.obs.cell_type == cell_type).tolist()

                    #if np.all(~np.array(idx)) or np.all(~np.array(idx_test)):
                    #    raise Exception(
                    #        "Cell type '{}' is exclusive present in training or testing split. This case (specially the latter) is not supported yet."
                    #    )

                else:
                    idx = np.ones(len(self.adata_train)).astype(bool)  # i.e. all cells are selected (because there is no "cell_type" column in adata.X)
                    idx_test = np.ones(len(self.adata_test)).astype(bool)

                
                ctrl_mean = self.adata_train[idx][self.adata_train[idx].obs[perturbation_key] == control].X.mean(axis=0)  # TODO:note: if reached an error here, one possibility is that in some splits, for some cell types there might be no control population available.
                self.expression_data[idx] = self.expression_data[idx] - ctrl_mean  # TODO:note: since adata.X is log1p-transformed, this is equivalent to division by control. Is this the intended behaviour ???
                

                adata_aktrain4eval.X[idx] = adata_aktrain4eval.X[idx] - ctrl_mean 
                adata_akevaluation.X[idx_test] = adata_akevaluation.X[idx_test] - ctrl_mean

                self.dict_celltype_to_ctrlmean_trainingset[cell_type] = ctrl_mean + 0.0

            
            self.cell_types = cell_types

            self.adata_aktrain4eval = adata_aktrain4eval
            self.adata_akevaluation = adata_akevaluation
        
        else:
            self.expression_data = (
                self.adata_train.X.toarray().copy() if issparse(self.adata_train.X)
                else self.adata_train.X.copy()
            )

            if "cell_type" in self.adata.obs.columns:
                self.cell_types = self.adata.obs.cell_type.unique()
            else:
                self.cell_types = ["cell_type"]

            self.adata_aktrain4eval = self.adata_train.copy()
            self.adata_akevaluation = self.adata_test.copy()

            self.dict_celltype_to_ctrlmean_trainingset = {}
            
        
        if type(self.expression_data) == np.matrix:
            self.expression_data = np.asarray(self.expression_data)

        logger.info("Fitting PCA...")
        if num_pca_components is None:
            
            raise Exception("Temprarily banned this part, to avoid unexpected behaviour.")
            
            self.pca = AkPCA(
                n_components=self.num_pca_components,
                flag_rapidssc_pca=flag_rapidssc_pca
            ) #PCA(n_components=100)
            self.pca.fit(self.expression_data)
            cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
            threshold = 0.99  
            self.num_pca_components = np.argmax(cumulative_variance >= threshold) + 1
            logger.info(f"99% of variance exaplained by {self.num_pca_components} PCs") 
            
        else:
            self.num_pca_components = num_pca_components
            self.pca = AkPCA(
                n_components=self.num_pca_components,
                flag_rapidssc_pca=flag_rapidssc_pca
            )
            #PCA(n_components=self.num_pca_components)
            self.pca.fit(self.expression_data)
        
        self.adata_train.obsm["X_pca"] = self.pca.transform(self.expression_data)  # TODO:note: this is to be used to do FM.
        self.adata_train.obsm['delta'] = self.expression_data.copy()

        perturbations = self.adata_train.obs[perturbation_key]
        perturbations = [re.split(sep, p.replace("/", "")) for p in perturbations]  
        # A list like [['control'], ['TSC22D1'], ['control'], ['ETS2', 'MAP7D1'], ['control'], ['CBL', 'PTPN9'], ...]
        
        
        self.perturbation_names = perturbations
        self.all_perturbations = [control] + sorted(
            set(p for ps in perturbations for p in ps if p != control)
        )  # i.e. unique set of perturbations

        self.shape_adata = self.adata.shape
        self.genetic_perturbation = genetic_perturbation
        self.gene_hot_encoded = None
        self.dim_gene_hot_encoded = None
        if self.genetic_perturbation: # Slightly conusing naming, if set to True doesn't mean it's genetic perturbation (for example in Norman it's set to False). It only means that perts are `gene_hot_encoded`.  
            
            raise Exception("Not sure what `genetic_perturbation` is used for. Banned until further checks.")
            
            gene_names = self.adata.var.gene_name
            self.gene_hot_encoded = np.zeros((self.shape_adata[0], len(gene_names)))
            self.dim_gene_hot_encoded = self.gene_hot_encoded.shape[1]  
            for i, pert in enumerate(perturbations):
                for p in pert:
                    self.gene_hot_encoded[i][gene_names == p] = 1

        try:
            self.embeddings_dict = self.adata.uns[perturbation_key]
        except KeyError:
            logger.error(
                f"Embeddings for perturbations not found in AnnData object. Key {perturbation_key} not found."
            )
            sys.exit(1)
        except Exception as e:
            logger.error(f"An error occurred while loading embeddings: {e}")
            sys.exit(1)

        size_embedding = self.embeddings_dict[self.all_perturbations[1]].shape[0]
        self.embeddings_dict[control] = torch.zeros(size_embedding)  # the embedding for control is all zeros.

        self.max_num_perturbations = max(len(ps) for ps in perturbations)  # i.e. max num combinatorial perts.
        self.perturbation_embeddings = torch.zeros(
            (self.adata_train.X.shape[0], self.max_num_perturbations, size_embedding),
            device=self.device,
        )  # [N x numPerts x embSize]  #TODO:note: maybe not the best efficient way, but OK for now.

        self.mask = torch.zeros(
            (self.adata_train.X.shape[0], self.max_num_perturbations), device=self.device
        )  # [N x numPerts], to contains one-hot encodings of the perturbations for each cell.

        for i, perturbation in enumerate(perturbations):
            for j, p in enumerate(perturbation):
                if p not in self.embeddings_dict:
                    logger.error(f"Embedding for perturbation {p} not found.")
                    sys.exit(1)
                else:
                    self.perturbation_embeddings[i, j] = torch.tensor(
                        self.embeddings_dict[p]
                    )
                    self.mask[i, j] = 1

        self.num_samples = self.expression_data.shape[0]
        self.num_genes = self.expression_data.shape[1]
        self.dim_perturbations = self.perturbation_embeddings.shape[2]

        #TODO:note: Since similar stuff are done for both `self.adata_train` and `self.adata_test`, maybe the better ways is to not do it only on `self.adata_train` but instead on the combined anndata object.

        self.condition_dict = dict()  # will contain set of possible values for each of the speicified covariates.
        self.condition_indexes = dict()  # for each of the specified covariates, it specifies how each possible value is mapped to an index.
        self.condition = dict()  # for each of the specified covariates and for all cells, it contains the indices (and not the one-hot) of the values.
        self.pretrained_embeddings = dict() # for each of the specified covariates, it contains the required dict in adata.uns.
        if condition_keys is not None:  # e.g. cell types or other covariates.
            for key in condition_keys:
                try:
                    self.condition_dict[key] = self.adata_train.obs[key].unique().tolist()  # set of possible values for that condition. TODO:note:check I added `.tolist()`
                    self.condition_indexes[key] = {
                        c: i for i, c in enumerate(sorted(self.condition_dict[key]))
                    }  # how the values are mapped to indices in that column of adata.obs.
                    self.condition[key] = torch.tensor(
                        [self.condition_indexes[key][c] for c in self.adata_train.obs[key].tolist()],  # TODO:note:check I added `.tolist()`
                        dtype=torch.long,
                        device=self.device,
                    )  # [numcells_train]
                except KeyError:
                    logger.error(
                        f"Conditions not found in AnnData object. Key {key} not found."
                    )
                    sys.exit(1)
                

                # TODO:note: I modified this part, for the following consideration
                #    The indexing in `self.pretrained_embeddings[covarname]` must be consistent with `self.condition_indexes`, while it used to add all values of the covar to `self.pretrained_embeddings`, even some covar-values were discarded form the anndata object.
                if key in self.adata_train.uns.keys(): #try:
                    covarname = key  # to improve readability

                    # check that the values in `adata.uns[covarname]` are all lists (and not, e.g., np.ndarrays)
                    for k, v in self.adata_train.uns[covarname].items():
                        if not (isinstance(v, list) or isinstance(v, np.ndarray)):
                            raise Exception(
                                "`adata.uns['{}']['{}']` is of type {}, while PertFlow expects it to be of type list or numpy.ndarray.".format(
                                    covarname,
                                    k,
                                    type(v)
                                )
                            )


                    assert list(self.condition_indexes[covarname].values()) == list(range(len(self.condition_indexes[covarname].values())))  # assert the indices are 0...num_covarvals

                    dict_index_to_covarval = {
                        idx:covarval
                        for covarval, idx in self.condition_indexes[covarname].items()
                    }
                    assert list(dict_index_to_covarval.keys()) == list(range(len(dict_index_to_covarval.keys())))

                    self.pretrained_embeddings[covarname] = np.array([
                        self.adata_train.uns[covarname][
                            dict_index_to_covarval[idx]
                        ] if isinstance(self.adata_train.uns[covarname][dict_index_to_covarval[idx]], list) else self.adata_train.uns[covarname][dict_index_to_covarval[idx]].flatten().tolist()
                        for idx in range(len(dict_index_to_covarval.keys()))
                    ])

                    #TODO:note: I deleted this line and added the lines above. Old code: self.pretrained_embeddings[key] = self.adata_train.uns[key]
                    
                else: #except KeyError:
                    logger.info(
                        f"Condition {key} Embedding not found in AnnData object. It will be computed automatically."
                    )

        
        # check if the following vars are consistent (specially the order of keys and indices)
        """
        condition_dict: will contain set of possible values for each of the speicified covariates.
        self.condition_indexes: for each of the specified covariates, it specifies how each possible value is mapped to an index.
        self.condition: for each of the specified covariates and for all cells, it contains the indices (and not the one-hot) of the values.
        self.pretrained_embeddings: for each of the specified covariates, it contains the required dict in adata.uns.
        """
        for d1 in [self.condition_dict, self.condition_indexes, self.condition]:
            for d2 in [self.condition_dict, self.condition_indexes, self.condition]:
                assert list(d1.keys()) == list(d2.keys()) 
        
        # The following consistency doesn't hold. 
        # for covarname in self.condition_dict.keys():
        #     assert self.condition_dict[covarname] == list(self.condition_indexes[covarname])
           
        for covarname in self.condition_dict.keys():
            assert len(self.condition_dict[covarname]) == len(list(self.condition_indexes[covarname]))
            assert set(list(self.condition_indexes[covarname].values())) == set(range(len(self.condition_dict[covarname])))  # i.e. the indices are 0,1,2...,len(covarvals)

        
        
        if dosage_key is not None:
            raise NotImplementedError(
                "The code with `dosage_key` is not tested yet."
            )
        
        self.dosage = (
            torch.tensor(
                self.adata_train.obs[dosage_key], dtype=torch.float32, device=self.device
            )
            if dosage_key is not None
            else torch.ones(self.num_samples, device=self.device)
        )  # [numcells_train]

    def assert_no_slashAndbackslash_in_perts_and_covars(self):
        """
        Asserts that in the perturbations and covaraite columns of the andata object, there are no slash or back slashes.
        This is done because in some parts of code '/' and '\\' characters are replaced by '', and that may lead to inconsistency.
        """
        for forbidden_char in ['/', '\\']:
            assert forbidden_char not in self.perturbation_key, print("Forbidden character '{}' was found in perturbation column name '{}'".format(forbidden_char, self.perturbation_key))
            
            for str_combpert in set(self.adata.obs[self.perturbation_key].tolist()):
                assert forbidden_char not in str_combpert, print("Forbideden charater '{}' was found in perturbation column.".format(
                    forbidden_char
                ))
            
            if self.condition_keys is not None:
                for covarname in self.condition_keys:
                    assert forbidden_char not in covarname, print("Forbidden character '{}' was found in covariate name '{}'".format(forbidden_char, covarname))

                    for covarval in set(self.adata.obs[covarname].tolist()):
                        assert forbidden_char not in covarval, print("Forbidden character '{}' was found in covariate value '{}' in covariate column '{}'".format(
                            forbidden_char,
                            covarval,
                            covarname
                        ))
                    

        
    
    def __len__(self):
        """Returns the number of samples in the dataset."""
        return self.expression_data.shape[0]  # TODO:note: here it's the training split (and not the testing).

    def custom_collate_fn(self, batch):


        expressions, perturbations, mask, conditions, dosages, one_hot_encoded_genes, _ = zip(*batch)
        expressions = torch.stack(expressions)
        perturbations = torch.stack(perturbations)
        mask = torch.stack(mask)
        dosages = torch.stack(dosages)
        one_hot_encoded_genes = torch.stack(one_hot_encoded_genes) if one_hot_encoded_genes[0] is not None else torch.zeros(dosages.shape[0])
        strIDs_perturbations = [b[6] for b in batch]
        
        # print("type(batch) = {}".format(type(batch)))
        # print("   len(batch) = {}".format(len(batch)))
        # print("      len(strIDs_perturbations) = {}".format(len(strIDs_perturbations)))
        # print("         set(strIDs_perturbations) = {}".format(set(strIDs_perturbations)))
        # print("\n\n")
        """
        prints out (for Norman)
        type(batch) = <class 'list'>
        len(batch) = 5000
            len(strIDs_perturbations) = 5000
                set(strIDs_perturbations) = {'SGK1+TBX3', 'SET', 'PTPN12+ZBTB10', 'UBASH3B+ZBTB25', 'PTPN12+UBASH3B'}
        
        Conclusion: with the custom sampler, batch is of length 5K (i.e. exactly as specified by the custom sampler) 
        """

        # Merge TensorDict conditions
        if isinstance(conditions[0], TensorDict):
            # raise Exception("Check if the below line, then uncomment this line.")

            conditions = TensorDict.cat(conditions, dim=0)
            # `conditions` is a tuple of length, e.g., 50K (50 conditions, and 100 samples from each). 
            # Its each element if a TensorDict that maps each covarname to a single index, where the single index is a tensor of size [1].
            # After the above line, `conditions` becomes a tensor dict that maps each covarname to a tensor of shape [num_cellss] 
        else:
            conditions = None
        
        return expressions, perturbations, mask, conditions, dosages, one_hot_encoded_genes, strIDs_perturbations
    
    def __getitem__(self, idx):
        if issparse(self.expression_data):  # TODO:note: `self.expression_data` seems to be always numpy.ndarray, so not sparse I guess.
            expression = self.expression_data[idx].toarray().ravel()
        else:
            expression = self.expression_data[idx]

        perturbation = self.perturbation_embeddings[idx]  # [numPerts x embSize]
        mask = self.mask[idx]  # [numPerts]
        dosage = self.dosage[idx]  # [numcells_train]

        if self.genetic_perturbation:
            gene_hot_encoded = torch.tensor(
                self.gene_hot_encoded[idx], dtype=torch.float32, device=self.device
                )
        else:
            gene_hot_encoded = None

        expression_tensor = torch.tensor(
            expression, dtype=torch.float32, device=self.device
        )
        perturbation_tensor = torch.tensor(
            perturbation, dtype=torch.float32, device=self.device
        )

        if len(self.condition_dict) > 0:
            dict_condition = dict()  # a dict that maps each covar-name to the index of value, e.g. {'covar_name1':torch.tensor([10]), 'covar_name2':torch.tensor([5])}, where 10 and 5 are the indices of the covariate values.

            for key in self.condition.keys():
                dict_condition[key] = self.condition[key][idx].unsqueeze(0)  # tensor of shape [1]
            
            condition_tensordict = TensorDict(
                dict_condition, batch_size=torch.Size([1]), device=self.device
            )
            # TODO:note: Recall that `self.condition[key]` contains the indices of that condition for all cells (and not the 1-hot-s).
            # TODO:note: shouldn't it be converted to condition embeddings here? It's done later on.

        else:
            condition_tensordict = None

        strIDs_postpertpopulation = self.adata_train.obs[self.perturbation_key].tolist()[idx]  # to be used to identify the population that the instance belongs to, usable for, e.g., mini-batch OT within each subpopulation.

        return (
            expression_tensor,
            perturbation_tensor,
            mask,
            condition_tensordict,
            dosage,
            gene_hot_encoded,
            strIDs_postpertpopulation
        )
