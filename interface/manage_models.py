# interface/manage_models.py
import os
import pickle
import time
import rapids_singlecell
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
import gc
import numpy as np
import pandas as pd
import torch
import torchdiffeq
import scipy.sparse as sp
import scipy.stats as sstats
from sklearn.metrics import mean_squared_error
from tensordict import TensorDict
import anndata as ad
import scanpy as sc
import wandb
import json
from sklearn.cluster import KMeans
from tqdm.autonotebook import tqdm
from enum import Enum

from .manage_adata import PerturbedGeneExpression

from models import GeneExpressionNet, GeneExpressionFlowModel


def _to_dense(a):
    return a.toarray() if sp.issparse(a) else np.asarray(a)


def _literal_sep(sep: Optional[str]) -> str:
    return sep if (sep is not None and sep != "") else "+"


def _default_modes_from_pca(adata_train: ad.AnnData, pca_obj, num_modes: int, D: int):
    """Return {'mean':[M,D], 'var':[M,D]} using PCA ranges."""
    if "X_pca" in adata_train.obsm and adata_train.obsm["X_pca"].shape[1] >= D:
        Xp = adata_train.obsm["X_pca"][:, :D]
    else:
        X = adata_train.X
        if hasattr(X, "toarray"):
            X = X.toarray()
        Xp = pca_obj.transform(X)[:, :D]
    mins = Xp.min(axis=0)
    maxs = Xp.max(axis=0)
    means = np.stack(
        [np.random.uniform(low=mins[d], high=maxs[d], size=(num_modes,)) for d in range(D)],
        axis=1
    ).astype(np.float32)
    vars_ = np.full((num_modes, D), 0.02, dtype=np.float32)
    return {"mean": means, "var": vars_}


class WrappedKmeans:
    """
    Kmeans clustering with some additional functionalities, e.g., providing the variance of each cluster or handling K=1. 
    """
    def __init__(self, kwargs_sklearn_kmeans):
        assert kwargs_sklearn_kmeans['n_clusters'] >= 1
        self.kwargs_sklearn_kmeans = kwargs_sklearn_kmeans
        self.kmeans = KMeans(**kwargs_sklearn_kmeans) if(kwargs_sklearn_kmeans['n_clusters'] > 1) else None  # None means K=1 ==> It's simple mean/variance calculation.

    def _check_input_X(self, x:np.ndarray):
        assert isinstance(x, np.ndarray)
        assert len(x.shape) == 2

    def _fit_Keq1(self, x:np.ndarray):
        """
        Handles the case where K=1.
        Sets `self.means` and `self.vars`, which are of shape [K x D].
        """
        assert self.kwargs_sklearn_kmeans['n_clusters'] == 1
        self._check_input_X(x)

        # compute means/vars
        # x: [N x D]
        self.means = np.expand_dims(np.mean(x, 0), 0)  # [1 x D]
        self.vars = np.expand_dims(np.std(x, 0) * np.std(x, 0), 0)  # [1 x D]
         

    def fit(self, x:np.ndarray):
        """
        Sets `self.means` and `self.vars`, which are of shape [K x D].
        """
        self._check_input_X(x)

        

        if self.kwargs_sklearn_kmeans['n_clusters'] == 1:
            self._fit_Keq1(x)
        else:
            self.kmeans.fit(x)
            cluster_assignments = self.kmeans.predict(x).tolist()  # list of length [num_cells], containing vlaues between 0 and `self.num_modes`.
            assert set(cluster_assignments) == set(range(self.kwargs_sklearn_kmeans['n_clusters']))  # so there are no empty clusters

            means = [
                np.expand_dims(
                    np.mean(
                        x[np.where(np.array(cluster_assignments) == c)[0].tolist()],
                        0
                    ),
                    0
                )
                for c in range(self.kwargs_sklearn_kmeans['n_clusters'])
            ]  # a list of lenght num_modes, whose elements are [1 x D]
            vars = [
                np.expand_dims(
                    np.std(
                        x[np.where(np.array(cluster_assignments) == c)[0].tolist()],
                        0
                    ) ** 2,
                    0
                )
                for c in range(self.kwargs_sklearn_kmeans['n_clusters'])
            ]  # a list of lenght num_modes, whose elements are [1 x D]
            
            self.means = np.concatenate(means,  0)  # [K x D]
            self.vars = np.concatenate(vars, 0)  # [K x D]
            




def _estimate_modes_via_Kmeans(
    adata_train: ad.AnnData,
    num_modes:int,
    float_fixedvariance_GMMs:float
):
    wrapped_kmeans = WrappedKmeans({
        'n_clusters':num_modes
    })

    wrapped_kmeans.fit(
        adata_train.obsm['X_pca']
    )
    modes = {
        'mean':wrapped_kmeans.means,
        'var':wrapped_kmeans.vars * 0.0 + float_fixedvariance_GMMs  
    }

    return modes





def _estimate_modes_via_leiden(adata_train: ad.AnnData, pca_obj, D: int,
                               res_grid=(0.4, 0.6, 0.8, 1.0, 1.2),
                               min_modes=3, max_modes=64) -> tuple[int, dict]:
    """
    Returns (num_modes, modes_dict) where modes_dict has keys 'mean' [M,D] and 'var' [M,D],
    computed by clustering in PCA space using Leiden.
    """
    # Use existing PCA if present; else transform
    if "X_pca" in adata_train.obsm and adata_train.obsm["X_pca"].shape[1] >= D:
        Xp = adata_train.obsm["X_pca"][:, :D]
    else:
        X = adata_train.X
        if hasattr(X, "toarray"): X = X.toarray()
        Xp = pca_obj.transform(X)[:, :D]

    # Build AnnData view with PCA
    tmp = ad.AnnData(X=Xp.copy())
    # neighbors over PCA
    sc.pp.neighbors(tmp, use_rep=None, n_neighbors=15, n_pcs=None)

    best_labels = None
    best_M = None
    for res in res_grid:
        sc.tl.leiden(tmp, resolution=float(res), key_added="leiden_modes", directed=False)
        labels = tmp.obs["leiden_modes"].astype(str).to_numpy()
        uniq = np.unique(labels)
        M = len(uniq)
        if min_modes <= M <= max_modes:
            best_labels, best_M = labels, M
            break
        # keep last as fallback
        best_labels, best_M = labels, M

    # Compute per-cluster mean/var in PCA dims
    uniq = np.unique(best_labels)
    means = []
    vars_ = []
    for u in uniq:
        mask = (best_labels == u)
        if not np.any(mask):
            continue
        Z = Xp[mask]
        means.append(Z.mean(axis=0))
        v = Z.var(axis=0)
        # small floor to keep variances valid
        vars_.append(np.maximum(v, 1e-3))
    means = np.stack(means, axis=0).astype(np.float32)
    vars_ = np.stack(vars_, axis=0).astype(np.float32)
    modes = {"mean": means, "var": vars_}
    return means.shape[0], modes

def _iter_chunks(total: int, chunk: int):
    s = 0
    while s < total:
        e = min(s + chunk, total)
        yield s, e
        s = e

class CalcModeGeodesicLength(Enum):
    MONGE_BASE_AND_TARGET = 1  # like Monge's formulation, E[X - T(X)] will be calculated.
    TRAJ_PIECE_LEN = 2  # sum of lengths of pieces of the trajectory 
    ELL2NORM_FLOW = 3   # Eq. 7 of Tong et. al.
    FMLOSS_ITSELF = 4  # The flow matching loss itself is used to update the base distribution
    ELL2NORM_U = 5     # The Ell-2 norm of Ut
    

class GeodesicLen:
    """
    The computation of geoodesic length during training in 3 different settings. 
    """

    def __init__(self, str_integfromX0_or_integfromX1:str, steps_ODEsolver:int, mode_compute_geodlen:int, coef_FMloss:float):
        """
        Inputs.
        - `str_integfromX0_or_integfromX1`: Whether the generated trajectories start from samples from the base dist or the target dist.
        - `str_mode_computeGoedLen`: 
            - ''
        """
        # check args 
        assert str_integfromX0_or_integfromX1 in ['integfromX0', 'integfromX1']
        assert isinstance(steps_ODEsolver, int)
        assert steps_ODEsolver > 0
        assert mode_compute_geodlen in [
            CalcModeGeodesicLength.MONGE_BASE_AND_TARGET.value,
            CalcModeGeodesicLength.TRAJ_PIECE_LEN.value,
            CalcModeGeodesicLength.ELL2NORM_FLOW.value,
            CalcModeGeodesicLength.FMLOSS_ITSELF.value,
            CalcModeGeodesicLength.ELL2NORM_U.value
        ]
        assert isinstance(coef_FMloss, float)
        assert coef_FMloss >= 0.0
        if mode_compute_geodlen == CalcModeGeodesicLength.FMLOSS_ITSELF.value:
            assert coef_FMloss == 1.0

        # grab args ===
        self.str_integfromX0_or_integfromX1 = str_integfromX0_or_integfromX1
        self.steps_ODEsolver = steps_ODEsolver
        self.mode_compute_geodlen = mode_compute_geodlen
        self.coef_FMloss = coef_FMloss

        # make internals
        self.loss_fn = torch.nn.MSELoss()
        
    
    def get_geod_len(self, vspace:torch.nn.Module, ut, vt, z0, z1, condition_and_perturbation, device):
        """
        This function assumes that mini-batch OT is already performed.
        """
        # get the trajectory
        if self.mode_compute_geodlen in [
            CalcModeGeodesicLength.MONGE_BASE_AND_TARGET.value,
            CalcModeGeodesicLength.TRAJ_PIECE_LEN.value,
            CalcModeGeodesicLength.ELL2NORM_FLOW.value
        ]:
            if self.str_integfromX0_or_integfromX1 == 'integfromX0':
                traj = torchdiffeq.odeint(
                    lambda t, z: vspace.forward(z, t, condition_and_perturbation),
                    z0,
                    torch.linspace(0, 1, steps=self.steps_ODEsolver).to(device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5",
                )  # [steps_ODEsolver x N x numPCs]
            else:
                assert self.str_integfromX0_or_integfromX1 == 'integfromX1'
                traj = torchdiffeq.odeint(
                    lambda t, z: vspace.forward(z, t, condition_and_perturbation),
                    z1,
                    torch.linspace(1, 0, steps=self.steps_ODEsolver).to(device),
                    atol=1e-4,
                    rtol=1e-4,
                    method="dopri5",
                )  # [steps_ODEsolver x N x numPCs]

        # switch based on the calculation mode
        if self.mode_compute_geodlen == CalcModeGeodesicLength.MONGE_BASE_AND_TARGET.value:

            # based on Monge ======
            if self.str_integfromX0_or_integfromX1 == 'integfromX0':
                geod_len = torch.mean(
                    torch.sum(
                        (traj[-1, :, :] - z0) ** 2,
                        1
                    ),
                    0
                )
            else:
                assert self.str_integfromX0_or_integfromX1 == 'integfromX1'
                geod_len = torch.mean(
                    torch.sum(
                        (traj[-1, :, :] - z1) ** 2,
                        1
                    ),
                    0
                )

        elif self.mode_compute_geodlen == CalcModeGeodesicLength.TRAJ_PIECE_LEN.value:

            # baed on sum of lenght of each trajectory piece ========
            traj_pad_beg = torch.cat(
                [torch.zeros_like(traj[0,:,:]).unsqueeze(0), traj],
                0
            )  # [(steps_ODEsolver+1) x N x numPCs]
            traj_pad_end = torch.cat(
                [traj, torch.zeros_like(traj[0,:,:]).unsqueeze(0)],
                0
            )  # [(steps_ODEsolver+1) x N x numPCs]

            traj_len = traj_pad_end - traj_pad_beg  # [(steps_ODEsolver+1) x N x numPCs]
            traj_len = torch.sum(traj_len[1:-1, :,  :] ** 2, 2)  # [(steps_ODEsolver-1) x N]
            traj_len = torch.sqrt(traj_len)  # [(steps_ODEsolver-1) x N]
            geod_len = torch.mean(torch.sum(traj_len, 0))
            
        elif self.mode_compute_geodlen == CalcModeGeodesicLength.ELL2NORM_FLOW.value:
            
            # based on Ell2 norm of the flow, evaluated at trajectory points =====
            assert self.str_integfromX0_or_integfromX1 == 'integfromX0'
            traj_len = 0.0
            for idx_t, t in enumerate(torch.linspace(0, 1, steps=self.steps_ODEsolver).tolist()):
                # skip the last time point
                if t == 1.0:
                    continue
                
                curr_traj_len = vspace.forward(
                    traj[idx_t, :,  :],
                    torch.tensor(condition_and_perturbation.shape[0]*[t]).float().to(device),
                    condition_and_perturbation
                )  # [N x numPCs]
                
                curr_traj_len = torch.mean(
                    torch.sum(curr_traj_len ** 2, 1),
                    0
                )
                traj_len = traj_len + curr_traj_len/(self.steps_ODEsolver - 1.0)
            
            geod_len = traj_len
        
        elif self.mode_compute_geodlen == CalcModeGeodesicLength.FMLOSS_ITSELF.value:
            # based on the FM loss itself ==> there'll be no geodesic lenth in this case
            geod_len = torch.tensor([0.0], device=z0.device).float()
        elif self.mode_compute_geodlen == CalcModeGeodesicLength.ELL2NORM_U.value:
            geod_len = torch.mean(ut ** 2)
        else:
            raise Exception("Unknown geodesic lenth calculation mode: {}".format(self.mode_compute_geodlen))


        # determine the 2nd returned term (i.e. the FMloss term)
        if self.coef_FMloss > 0.0:
            fm_loss = self.coef_FMloss * self.loss_fn(vt, ut)
        else:
            fm_loss = torch.tensor([0.0], device=z0.device).float()
        
        return geod_len, fm_loss


class SchedulerUpdateBaseDist:
    def __init__(
            self,
            total_epochs:int,
            dict_intervalname_to_fracepochs:dict,
            itercycle_update_basedist: int,
            flag_cooldown_settoeval_HthetaHp:bool
        ):
        """
        
        :param total_epochs: total number of epochs.
        :type total_epochs: int
        :param dict_intervalname_to_fracepochs: A dictionary whose values sum up to one, and with the following keys
        - warmup
        - bothVandB
        - cooldown
        :param itercycle_update_basedist: an integer, after this many updates to the flow, a single update is made to the basedist.
        :param flag_cooldown_settoeval_HthetaHp: determines if in the coodldown phase Htheta and Hp are set to eval mode.
        """
        # check args
        assert isinstance(total_epochs, int)
        assert total_epochs > 0
        assert isinstance(dict_intervalname_to_fracepochs, dict)
        assert set(dict_intervalname_to_fracepochs.keys()) == {'warmup', 'bothVandB', 'cooldown'}
        assert sum(list(dict_intervalname_to_fracepochs.values())) == 1.0, print("The values in `dict_intervalname_to_fracepochs` do not sum up to 1.0.")
        assert isinstance(itercycle_update_basedist, int)
        assert isinstance(flag_cooldown_settoeval_HthetaHp, bool)

        # grab args
        self.itercycle_update_basedist = itercycle_update_basedist
        self.flag_cooldown_settoeval_HthetaHp = flag_cooldown_settoeval_HthetaHp

        # split total epochs to the 3 phases
        N_warmup = int(dict_intervalname_to_fracepochs['warmup'] * total_epochs)
        N_bothVandB = int(dict_intervalname_to_fracepochs['bothVandB'] * total_epochs)
        N_cooldown = total_epochs - N_warmup - N_bothVandB

        assert (N_warmup + N_bothVandB + N_cooldown == total_epochs)
        self.N_warmup = N_warmup
        self.N_bothVandB = N_bothVandB
        self.N_cooldown = N_cooldown


    def get_next(self, idx_epoch:int, idx_iter:int):
        """
        Returns 
        - flag_update_basedist
        - flag_setto_eval_mode_Htheta_Hp
        """

        if idx_epoch < self.N_warmup:
            return False, False
        elif idx_epoch < (self.N_warmup + self.N_bothVandB):
            return (idx_iter%self.itercycle_update_basedist == 0), False
        else:
            return False, self.flag_cooldown_settoeval_HthetaHp
        

"""
class SchedulerUpdateBaseDist:
    def __init__(self, numepochs_init_warmup: int, itercycle_update_basedist: int):
        self.numepochs_init_warmup = int(numepochs_init_warmup)
        self.itercycle_update_basedist = int(itercycle_update_basedist)

    def get_flag_updateBaseDist(self, idx_epoch: int, idx_iter: int) -> bool:
        if idx_epoch <= self.numepochs_init_warmup:
            return False
        return (idx_iter % self.itercycle_update_basedist) == 0
"""



class PertFlow:
    def __init__(
        self,
        data: PerturbedGeneExpression,
        kwargs_OT_sampler: dict,
        float_fixedvariance_GMMs:float,
        kwargs_scheduler_updatebasedist: dict,
        prob_dropout_conditionencoder: float,
        prob_dropout_fc1fc2:float,
        kwargs_GeodesicLen:dict,
        hidden_dims:List[int],
        dim_perturbations: int,
        conditional_noise: bool,
        num_modes: int,
        temperature: float,
        kwargs_Htheta:dict,
        dict_kwargs_akDebug: Dict,
        numupdates_each_minibatch: int,
        kwargs_initial_kmeans_for_Htheta:dict,
        gumbel_softmax_flag_hard:bool,
        flag_rapidssc_KMeans:bool,
        MFM: bool = False,
    ):
        self.data = data
        self.device = self.data.device
        self.conditional_noise = bool(conditional_noise)
        self.num_modes = int(num_modes)
        self.temperature = float(temperature)
        self.MFM = bool(MFM)
        self.loss_fn = torch.nn.MSELoss()
        self.dict_kwargs_akDebug = dict_kwargs_akDebug
        self.numupdates_each_minibatch = int(numupdates_each_minibatch)
        self.has_covariates = bool(getattr(self.data.gene_expression_dataset, "condition_keys", []))
        # self.kwargs_GeodesicLen = kwargs_GeodesicLen
        self.float_fixedvariance_GMMs = float_fixedvariance_GMMs
        self.kwargs_initial_kmeans_for_Htheta = kwargs_initial_kmeans_for_Htheta


        # dataset-derived handles
        ds = self.data.gene_expression_dataset
        num_pca_components = int(self.data.num_pca_components or getattr(ds, "num_pca_components", 0) or 2)
        conditions = getattr(ds, "condition_dict", None) or getattr(ds, "condition", {}) or {}
        pretrained_embeddings = getattr(ds, "pretrained_embeddings", {}) or {}
        self.dim_perturbation = int(getattr(ds, "dim_perturbations", dim_perturbations))
        # FIX: if gene-hot isn't used, set 0 (don't force int(None))
        dgene = getattr(ds, "dim_gene_hot_encoded", None)
        self.dim_gene_hot_encoded = int(dgene) if isinstance(dgene, (int, float, np.integer)) else 0

        # Build covariate embedding dims (if any)
        dict_covarname_to_dimembedding = {}
        for covar in (self.data.condition_keys or []):
            if covar in ds.adata_train.uns:
                example = list(ds.adata_train.uns[covar].values())[0]
                dim = len(example) if isinstance(example, list) else np.asarray(example).reshape(-1).shape[0]
            else:
                dim = 10
            dict_covarname_to_dimembedding[covar] = int(dim)

        # v(x, t, cond)
        self.vspace_model = GeneExpressionNet(
            input_dim=num_pca_components,
            dim_embedding=dim_perturbations,
            num_conditions=len(conditions),
            hidden_dims=hidden_dims,
            dict_covarname_to_dimembedding=dict_covarname_to_dimembedding,
        ).to(self.device)

        # Here the GMM `modes` are created and are later fed to `GeneExpressionFlowModel`.
        if conditional_noise: 
            modes = {
                'mean':None,  # the means are produced by a module ==> set to None
                'var':self.float_fixedvariance_GMMs  
            }
        else:
            modes = None
        


        self.modes = modes

        # get the Kmeans modes on combined X, to be used for the initial training of H_theta, or making H_theta additive to these clustre centres.
        assert isinstance(flag_rapidssc_KMeans, bool)
        assert flag_rapidssc_KMeans in [True, False]

        if flag_rapidssc_KMeans:
            TMP_KEY_ADDED = "MIXFLOW_INITIAL_KMEANS"
            assert not (TMP_KEY_ADDED in self.data.gene_expression_dataset.adata_train.obs.columns.tolist())
            rapids_singlecell.tl.kmeans(
                self.data.gene_expression_dataset.adata_train,
                n_clusters=num_modes,
                use_rep='X_pca',
                key_added=TMP_KEY_ADDED,
                n_pcs=self.data.gene_expression_dataset.adata_train.obsm['X_pca'].shape[1]  # i.e. all columns of .obsm['X_pca']
            )
            
            self.data.gene_expression_dataset.adata_train.obs[TMP_KEY_ADDED] = \
                self.data.gene_expression_dataset.adata_train.obs[TMP_KEY_ADDED].astype(int)
            

            assert set(self.data.gene_expression_dataset.adata_train.obs[TMP_KEY_ADDED].tolist())  == set(range(num_modes))

            ten_precopmuted_cluster_modes = []
            for int_cluster_id in range(num_modes):
                ten_precopmuted_cluster_modes.append(
                    np.mean(
                        self.data.gene_expression_dataset.adata_train.obsm['X_pca'][
                            self.data.gene_expression_dataset.adata_train.obs[TMP_KEY_ADDED] == int_cluster_id,
                            :
                        ],
                        0
                    )  # [D]
                )
            ten_precopmuted_cluster_modes = np.stack(ten_precopmuted_cluster_modes, 0)  # [num_modes x D]
            
            self.ten_precopmuted_cluster_modes = torch.tensor(ten_precopmuted_cluster_modes)
            
            
        else:
            kmeans = KMeans(**self.kwargs_initial_kmeans_for_Htheta)
            kmeans.fit(self.data.gene_expression_dataset.adata_train.obsm['X_pca'])
            self.ten_precopmuted_cluster_modes = torch.tensor(kmeans.cluster_centers_)  # [num_modes x D]

            """
            With `num_modes` equal to 100 and 10 PCs,
            `kmeans.cluster_centers_` is a numpy.ndarray of shape [100 x 10]
            """
            

        # Full flow model
        self.model = GeneExpressionFlowModel(
            vspace_model=self.vspace_model,
            dim_perturbation=self.dim_perturbation,
            prob_dropout_conditionencoder=prob_dropout_conditionencoder,
            prob_dropout_fc1fc2=prob_dropout_fc1fc2,
            dim_gene_hot_encoded=self.dim_gene_hot_encoded,
            conditions=conditions,
            pretrained_embeddings=pretrained_embeddings,
            conditional_noise=self.conditional_noise,
            num_modes=self.num_modes,
            modes=self.modes,
            temperature=self.temperature,
            gumbel_softmax_flag_hard=gumbel_softmax_flag_hard,
            MFM=self.MFM,
            kwargs_OT_sampler=kwargs_OT_sampler,
            ten_precopmuted_cluster_modes=self.ten_precopmuted_cluster_modes,
            kwargs_Htheta=kwargs_Htheta,
        ).to(self.device)

        self.sched_flag_update_basedist = SchedulerUpdateBaseDist(**kwargs_scheduler_updatebasedist)

        self.obj_get_geodlen = GeodesicLen(**kwargs_GeodesicLen)

        



    # --------------------------- separate and initial training of H_theta ---------------------------
    def train_sep_Htheta(
        self,
        batch_size: int,
        epochs: int,
        lr: float,
        wandb_runname: str,
        stddiv_noise_fit2_cluster_modes:float,
        min_delta: float = 0.0,
        patience: int = 10,
        board: bool = False,
        hyperparameters: dict = None,
        resume: bool = False,
        conditional_sampler_path: Optional[str] = None,
        vspace_model_path: Optional[str] = None,
    ):

        if not self.conditional_noise:
            print("`conditional_nosie` is set to False, there is no H_theta module, so skipping the initial training of H_theta.")
            return 0

        if hasattr(self.model.module_H_theta, 'upper_bound_additive_amount'):
            if self.model.module_H_theta.upper_bound_additive_amount == 0.0:
                print("H_theta is fixed to cluster centres ==> skipped training H_theta.")
                return 0

        self.model.module_H_theta.train()
        opt_Htheta = torch.optim.AdamW(self.model.module_H_theta.parameters(), lr=lr) 
        

        self.data.init_data_loader(batch_size)
        dl = self.data.data_loader

        """
        Will be done in self.train ===> commented out here.
        if board:
            wandb.init(project="FM-GeneExpression", name=wandb_runname, config=hyperparameters or {})
            wandb.watch(self.model, self.loss_fn, log="all", log_freq=10)
        """

        iters = 0
        with torch.enable_grad():
            for ep in tqdm(range(int(epochs)), desc='The initial training of Htheta'):
                acc = 0.0
                for _, batch in enumerate(dl):
                    x1, pert, mask, cond_td, dosage, gene_hot, _ = batch

                    x1 = x1.to(self.device)
                    pert = pert.to(self.device).to(torch.float32)
                    mask = mask.to(self.device)
                    dosage = dosage.to(self.device)
                    gene_hot = gene_hot.to(self.device) if gene_hot is not None else None

                    # Ensure a non-None, batch-aligned conditions container
                    if cond_td is None:
                        conditions = TensorDict({}, batch_size=[x1.shape[0]]).to(self.device)
                    else:
                        conditions = cond_td.to(self.device)

                    # PCA -> latent
                    z1_np = self.data.gene_expression_dataset.pca.transform(x1.detach().cpu())
                    z1 = torch.as_tensor(z1_np, dtype=torch.float32, device=self.device)

                    update_base = True  # because it's the initial training of basedist. self.sched_flag_update_basedist.get_flag_updateBaseDist(ep, iters)
                    
                    opt_Htheta.zero_grad()

                    loss = self.model.forward_trainsep_Htheta(
                        z1, pert, mask, conditions, dosage, gene_hot,
                        update_base, strID_perturbations=None,
                        ten_precopmuted_cluster_modes=self.ten_precopmuted_cluster_modes.to(self.device),  # [num_modes x D]
                        stddiv_noise_fit2_cluster_modes=stddiv_noise_fit2_cluster_modes
                    )

                    loss.backward()
                    opt_Htheta.step()

                    if board:
                        with torch.no_grad():
                            wandb.log(
                                {"Loss/initital_training_H_theta": loss.detach().cpu().numpy()},
                                step=iters
                            )
                            iters += 1
            
        
        
        pass
        return iters
        


    

    # --------------------------- training ---------------------------
    def train(
        self,
        batch_size: int,
        epochs: int,
        dict_kwargs_optim_flow: float,
        dict_kwargs_optim_basedist:float,
        wandb_runname: str,
        kwargs_trainsep_Htheta:dict,
        min_delta: float = 0.0,
        patience: int = 10,
        board: bool = False,
        hyperparameters: dict = None,
        resume: bool = False,
        conditional_sampler_path: Optional[str] = None,
        vspace_model_path: Optional[str] = None
    ):
        
        if board:
            wandb.init(project="FM-GeneExpression", name=wandb_runname, config=hyperparameters or {})
            wandb.watch(self.model, self.loss_fn, log="all", log_freq=10)

        if resume:
            if vspace_model_path:
                self.vspace_model.load_state_dict(
                    torch.load(vspace_model_path, map_location=self.device, weights_only=True)
                )
            if self.conditional_noise and conditional_sampler_path:
                ckpt = torch.load(conditional_sampler_path, map_location=self.device, weights_only=True)
                self.model.load_state_dict(ckpt, strict=True)
            
            iters = 0
            
        else:
            # the ones added: 
            keys_tobeadded_here = ['wandb_runname', 'conditional_sampler_path', 'vspace_model_path', 'board']
            for k in keys_tobeadded_here:
                assert k not in kwargs_trainsep_Htheta
            
            iters = self.train_sep_Htheta(**{
                **kwargs_trainsep_Htheta,
                **{
                    'wandb_runname':wandb_runname,
                    'conditional_sampler_path':conditional_sampler_path,
                    'vspace_model_path':vspace_model_path,
                    'board':board
                }
            })


        opt_v = torch.optim.Adam(self.model.vspace.parameters(), **dict_kwargs_optim_flow)

        if self.conditional_noise:
            opt_b = torch.optim.Adam(self.model.getparams_base_dist_encoder(), **dict_kwargs_optim_basedist) if self.conditional_noise else None

        self.data.init_data_loader(batch_size)
        dl = self.data.data_loader


        best = float("inf")
        noimp = 0
        
        global_iters = -1

        try:
            with torch.enable_grad():
                for ep in tqdm(range(int(epochs)), desc='Trainig epoch'):
                    acc = 0.0
                    for _, batch in enumerate(dl):

                        global_iters += 1

                        x1, pert, mask, cond_td, dosage, gene_hot, strID_perturbations = batch

                        x1 = x1.to(self.device)
                        pert = pert.to(self.device).to(torch.float32)
                        mask = mask.to(self.device)
                        dosage = dosage.to(self.device)
                        gene_hot = gene_hot.to(self.device) if gene_hot is not None else None

                        # Ensure a non-None, batch-aligned conditions container
                        if cond_td is None:
                            conditions = TensorDict({}, batch_size=[x1.shape[0]]).to(self.device)
                        else:
                            conditions = cond_td.to(self.device)

                        # PCA -> latent
                        z1_np = self.data.gene_expression_dataset.pca.transform(x1.detach().cpu())
                        z1 = torch.as_tensor(z1_np, dtype=torch.float32, device=self.device)

                        update_base, flag_setto_eval_mode_Htheta_Hp = self.sched_flag_update_basedist.get_next(
                            idx_epoch=ep,
                            idx_iter=global_iters
                        )
                        assert update_base in [True, False]
                        assert flag_setto_eval_mode_Htheta_Hp in [True, False]

                        
                        # TODO:is this needed ???
                        if not flag_setto_eval_mode_Htheta_Hp:
                            self.model.vspace.train(); self.model.lower_dim_embedding_perturbation.train()
                            if self.conditional_noise:
                                self.model.module_H_theta.train(); self.model.module_fc1fc2.train()
                        else:
                            self.model.vspace.train(); self.model.lower_dim_embedding_perturbation.eval()
                            if self.conditional_noise:
                                self.model.module_H_theta.eval(); self.model.module_fc1fc2.eval()

                        opt_v.zero_grad()
                        if self.conditional_noise:
                            opt_b.zero_grad()
                        
                        

                        ut, vt, z0_begininteg, condition_and_perturbation_fedtoV = self.model.forward(
                            z1, pert, mask, conditions, dosage, gene_hot,
                            update_base, strID_perturbations=strID_perturbations
                        )

                        if update_base:
                            if not self.conditional_noise:
                                raise Exception(
                                    "`conditional_noise` is set to False (i.e. it's the CFM setting), but the base distribution is asked to get updated.\n"+\
                                    "To avoid this issue please set `kwargs_scheduler_updatebasedist.numiters_init_warmup` to a huge number, e.g., 100000000, and try again."
                                )

                            loss_basedist, loss_FM = self.obj_get_geodlen.get_geod_len(
                                vspace=self.model.vspace,
                                ut=ut,
                                vt=vt,
                                z0=z0_begininteg,
                                z1=z1, 
                                condition_and_perturbation=condition_and_perturbation_fedtoV,
                                device=z1.device
                            )
                            loss = loss_basedist + loss_FM
                        else:
                            loss = self.loss_fn(vt, ut)  # it's MSE loss (the one used also in mintflow)
                        
                        
                        loss.backward()

                        if update_base:
                            assert self.conditional_noise
                            opt_b.step()
                        else:
                            opt_v.step()
                        

                        acc += float(loss.item())

                        if board:
                            # str_name_loss = 'fmloss (shortenning the geodesic)' if update_base else 'fmloss (without l1_loss)'
                            with torch.no_grad():
                                if update_base:
                                    wandb.log(
                                        {"Loss/2Wasserstein (training base dist)": loss_basedist.detach().cpu().numpy()},
                                        step=iters
                                    )
                                    wandb.log(
                                        {"Loss/FMloss (training base dist)\n after mult. by coef {}".format(self.obj_get_geodlen.coef_FMloss): loss_FM.detach().cpu().numpy()},
                                        step=iters
                                    )
                                else:
                                    wandb.log(
                                        {"Loss/FMloss (training V)": loss.detach().cpu().numpy()},
                                        step=iters
                                    )

                                wandb.log(
                                    {"ProbedVals/flag_update_basedist":torch.tensor(update_base + 0.0).detach().cpu().numpy()},
                                    step=iters
                                )
                                wandb.log(
                                    {"ProbedVals/flag_setto_eval_mode_Htheta_Hp":torch.tensor(flag_setto_eval_mode_Htheta_Hp + 0.0).detach().cpu().numpy()},
                                    step=iters
                                )
                        
                        
                        iters += 1

                    avg = acc / max(1, len(dl))
                    if avg < best - min_delta:
                        best, noimp = avg, 0
                    else:
                        noimp += 1
                    # Optional: if noimp >= patience: break

                    if vspace_model_path:
                        torch.save(self.vspace_model.state_dict(), vspace_model_path)
                    if self.conditional_noise and conditional_sampler_path:
                        torch.save(self.model.state_dict(), conditional_sampler_path)
                        meta = {
                            "num_modes": self.num_modes,
                            "conditional_noise": self.conditional_noise,
                            "num_pca_components": int(self.data.num_pca_components or 0),
                        }
                        with open(conditional_sampler_path + ".meta.json", "w") as f:
                            json.dump(meta, f)
        finally:
            if board:
                wandb.finish()
            if vspace_model_path:
                torch.save(self.vspace_model.state_dict(), vspace_model_path)
            if self.conditional_noise and conditional_sampler_path:
                torch.save(self.model.state_dict(), conditional_sampler_path)
                meta = {
                    "num_modes": self.num_modes,
                    "conditional_noise": self.conditional_noise,
                    "num_pca_components": int(self.data.num_pca_components or 0),
                }
                with open(conditional_sampler_path + ".meta.json", "w") as f:
                    json.dump(meta, f)

    
    




    # --------------------------- generation (any CSV) ---------------------------
    @torch.no_grad()
    def generate_anycsv(self, num_samples: int, conditions: str, steps_ODEsolver: int):
        self.model.eval(); self.model.vspace.eval(); self.model.lower_dim_embedding_perturbation.eval()

        if self.conditional_noise:
            self.model.module_H_theta.eval(); self.model.module_fc1fc2.eval()
        
        
        traj, cp = self._generate_anycsv(num_samples, conditions, steps_ODEsolver)
        
        self.model.train(); self.model.vspace.train(); self.model.lower_dim_embedding_perturbation.train()
        
        if self.conditional_noise:
            self.model.module_H_theta.train(); self.model.module_fc1fc2.train()
        
        return traj, cp


    @torch.no_grad()
    def _generate_final_in_chunks(
        self,
        num_samples: int,
        conditions: str,
        steps_ODEsolver: int,
        max_batch_N: int = 32768,
    ):
        """
        Chunked generation that returns ONLY the final latent state (no full trajectory),
        to keep GPU memory bounded.
        """
        ds = self.data.gene_expression_dataset
        perturbation_dict = ds.embeddings_dict
        condition_indexes = getattr(ds, "condition_indexes", {})  # {} if no covars
        pert_key = ds.perturbation_key
        sep = _literal_sep(self.data.sep_literal)
        ctrl = getattr(self.data, "control", None)

        df = pd.read_csv(conditions)
        df.columns = df.columns.map(str).str.strip()
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
        if pert_key not in df.columns:
            raise KeyError(f"CSV must contain '{pert_key}'.")
        if "dosage" not in df.columns:
            raise KeyError("CSV must contain 'dosage'.")

        # covariates used during training (column order must match)
        if condition_indexes:
            train_cov_order = list(getattr(ds, "condition", {}).keys())
            missing = [c for c in train_cov_order if c not in df.columns]
            if missing:
                raise ValueError(f"Missing covariate columns: {missing}")
            cond_cols = train_cov_order
            has_covs = True
        else:
            cond_cols, has_covs = [], False

        perts = df[pert_key].astype(str).tolist()
        dosages = df["dosage"].to_numpy(dtype=np.float32)
        P = len(perts)

        # Prepare pooled embeddings for each CSV row on CPU
        d_emb = int(np.array(next(iter(perturbation_dict.values()))).shape[-1])
        Pmax = ds.max_num_perturbations
        pooled_cpu = []
        for comb in perts:
            singles = [t for t in comb.replace("/", "").split(sep) if t]
            if ctrl is not None:
                singles = [t for t in singles if t != ctrl]
            if len(singles) == 0:
                pooled_cpu.append(np.zeros(d_emb, dtype=np.float32))
                continue
            if len(singles) > Pmax:
                raise ValueError(f"{len(singles)} singles but Pmax={Pmax}")
            vecs = []
            for p in singles:
                if p not in perturbation_dict:
                    raise KeyError(f"Missing embedding for '{p}' (comb='{comb}', sep='{sep}')")
                v = np.asarray(perturbation_dict[p], dtype=np.float32).reshape(-1)
                if v.shape[0] != d_emb:
                    raise ValueError(f"Embedding size mismatch for '{p}': {v.shape[0]} vs {d_emb}")
                vecs.append(v)
            pooled_cpu.append(np.mean(np.stack(vecs, axis=0), axis=0))
        
        pooled_cpu = np.stack(pooled_cpu, axis=0)  # [P, d_emb]
        pooled_cpu = pooled_cpu * dosages[:, None].astype(np.float32)

        # Prebuild covariate index arrays per row
        cov_idx_per_row = {}
        if has_covs:
            for c in cond_cols:
                cov_idx_per_row[c] = np.array([condition_indexes[c][df.iloc[i][c]] for i in range(P)], dtype=np.int64)

        # Return ONLY the final latent states, in CPU memory
        D = int(self.data.num_pca_components or getattr(ds, "num_pca_components", 0) or 2)
        final_latents_cpu = []
        final_basedist_samples_cpu = []
        final_Htheta_output_cpu = []

        # Only keep t0 and t1
        t_eval = torch.tensor([0.0, 1.0], device=self.device, dtype=torch.float32)

        # rows_per_chunk so that (rows_per_chunk * num_samples) ≤ max_batch_N
        rows_per_chunk = max(1, max_batch_N // int(num_samples))

        np_str_pertnames_4vis = []
        for s, e in _iter_chunks(P, rows_per_chunk):
            K = e - s

            # Expand pooled perturbations and project to low-dim expected by vspace
            pert_expanded = torch.as_tensor(
                np.repeat(pooled_cpu[s:e], repeats=int(num_samples), axis=0),
                device=self.device, dtype=torch.float32
            )  # [K*num_samples, d_emb]

            np_str_pertnames_4vis = np_str_pertnames_4vis + np.repeat(np.array(perts[s:e]), repeats=int(num_samples)).tolist() 

            pert_low = self.model.lower_dim_embedding_perturbation(pert_expanded)  # [K*num_samples, dim_perturbations]

            # Build covariate embeddings for this chunk
            if has_covs:
                parts = []
                for c in cond_cols:
                    idx_chunk = np.repeat(cov_idx_per_row[c][s:e], repeats=int(num_samples), axis=0)
                    x = torch.as_tensor(idx_chunk, device=self.device, dtype=torch.long)
                    y = self.model.embedding_layers[c](x)  # [K*num_samples, dim_c]
                    parts.append(y)
                cond_global = torch.cat(parts, dim=1) if parts else None
            else:
                cond_global = None

            cp = torch.cat((cond_global, pert_low), dim=1) if cond_global is not None else pert_low  # [N_chunk, ?]

            # sample initial noise
            N_chunk = cp.shape[0]
            if self.conditional_noise:
                z0, _, _, Htheta_output_4vis = self.model.sample_noise_from_gmm(cp.to(torch.float32))
                
            else:
                z0 = torch.randn(N_chunk, D, device=self.device)

            # safety
            assert isinstance(z0, torch.Tensor), f"Expected Tensor z0, got {type(z0)}"
            assert z0.ndim == 2 and z0.shape[1] == D, f"z0 shape {z0.shape} incompatible with D={D}"

            def vf(tt, zz):
                return self.model.vspace.forward(zz, tt, cp)

            Z = torchdiffeq.odeint(vf, z0, t_eval, atol=1e-4, rtol=1e-4, method="dopri5")  # [2, N_chunk, D]
            Z_final = Z[-1].detach().to("cpu").numpy()  # [N_chunk, D]
            final_latents_cpu.append(Z_final)
            final_basedist_samples_cpu.append(Z[0].detach().cpu().numpy())

            if self.conditional_noise:
                final_Htheta_output_cpu.append(Htheta_output_4vis.detach().cpu().numpy())


            # free GPU
            del pert_expanded, pert_low, cond_global, cp, z0, Z, Z_final
            torch.cuda.empty_cache()

        final_latents_cpu = np.concatenate(final_latents_cpu, axis=0)  # [P*num_samples, D]
        final_basedist_samples_cpu = np.concatenate(final_basedist_samples_cpu, axis=0)  # [P*num_samples, D]
        np_str_pertnames_4vis = np.array(np_str_pertnames_4vis)  # [P*num_samples]

        if self.conditional_noise:
            final_Htheta_output_cpu = np.concatenate(final_Htheta_output_cpu, 0)  
        
        return final_latents_cpu, final_basedist_samples_cpu, np_str_pertnames_4vis, final_Htheta_output_cpu  # TODO:modify `np_str_pertnames_4vis` when covariates are added.


    @torch.no_grad()
    def _generate_anycsv(self, num_samples: int, conditions: str, steps_ODEsolver: int):
        """
        Full-trajectory path (kept for visualization/debug).

        Reads a CSV of conditions (must include the training-time perturbation key and 'dosage'),
        builds the conditioner by concatenating (optional) covariate embeddings with the
        pooled-and-projected perturbation embedding, samples a base point (optionally from a
        conditional GMM), integrates the vector field, and returns the trajectory and the final
        conditioner matrix.

        Returns
        -------
        traj : np.ndarray
            Shape [num_samples, P, steps_ODEsolver, D], with D = num PCA components.
        cp : np.ndarray
            The conditioner used during ODE integration, shape [P*num_samples, C], where
            C = sum(cov_emb_dims) + vspace_model.dim_embedding.
        """
        ds = self.data.gene_expression_dataset
        perturbation_dict = ds.embeddings_dict
        condition_indexes = getattr(ds, "condition_indexes", {})  # {} if no covars
        pert_key = ds.perturbation_key
        sep = _literal_sep(self.data.sep_literal)
        ctrl = getattr(self.data, "control", None)

        # --- read CSV & basic checks
        df = pd.read_csv(conditions)
        df.columns = df.columns.map(str).str.strip()
        df = df.loc[:, ~df.columns.str.match(r"^Unnamed")]
        if pert_key not in df.columns:
            raise KeyError(f"CSV must contain '{pert_key}'.")
        if "dosage" not in df.columns:
            raise KeyError("CSV must contain 'dosage'.")

        # covariates in the same order used during training (if any)
        if condition_indexes:
            train_cov_order = list(getattr(ds, "condition", {}).keys())
            missing = [c for c in train_cov_order if c not in df.columns]
            if missing:
                raise ValueError(f"Missing covariate columns: {missing}")
            cond_cols = train_cov_order
            has_covs = True
        else:
            cond_cols, has_covs = [], False

        perts = df[pert_key].astype(str)
        dosages = df["dosage"].to_numpy(dtype=np.float32)
        P = len(perts)

        # --- build [P, Pmax, d_emb] tensor of raw per-perturbation embeddings
        d_emb = int(np.array(next(iter(perturbation_dict.values()))).shape[-1])
        Pmax = ds.max_num_perturbations
        E = torch.zeros((P, Pmax, d_emb), device=self.device, dtype=torch.float32)
        M = torch.zeros((P, Pmax), device=self.device, dtype=torch.float32)

        for i, comb in enumerate(perts):
            singles = [t for t in comb.replace("/", "").split(sep) if t]
            if ctrl is not None:
                singles = [t for t in singles if t != ctrl]
            if len(singles) == 0:
                E[i, 0].zero_()
                M[i, 0] = 1.0
                continue
            if len(singles) > Pmax:
                raise ValueError(f"{len(singles)} singles but Pmax={Pmax}")
            j = 0
            for p in singles:
                if p not in perturbation_dict:
                    raise KeyError(f"Missing embedding for '{p}' (comb='{comb}', sep='{sep}')")
                v = torch.as_tensor(perturbation_dict[p], device=self.device, dtype=torch.float32).view(-1)
                if v.numel() != d_emb:
                    raise ValueError(f"Embedding size mismatch for '{p}': {v.numel()} vs {d_emb}")
                E[i, j].copy_(v)
                M[i, j] = 1.0
                j += 1

        # --- pooled_low is already in the vspace embedding dim (masked_mean_pooling calls lower_dim_embedding_perturbation)
        pooled_low = self.model.masked_mean_pooling(E, M)  # [P, dim_embedding]
        del E, M
        torch.cuda.empty_cache()

        # dosage modulation at the pooled-low level
        pooled_low = pooled_low * torch.as_tensor(dosages, device=self.device, dtype=pooled_low.dtype).unsqueeze(1)

        # expand to [N, dim_embedding] directly (NO re-projection!)
        N = int(num_samples) * P
        pert_low = pooled_low.repeat_interleave(int(num_samples), dim=0)  # [N, dim_embedding]

        # --- build covariate embeddings aligned to [N, *]
        if has_covs:
            idx_map = {
                c: torch.as_tensor(
                    np.array([condition_indexes[c][df.iloc[i][c]] for i in range(P)], dtype=np.int64)
                ).repeat_interleave(int(num_samples))
                for c in cond_cols
            }
            parts = []
            # chunk to reduce peak memory during embedding lookup
            chunk = max(512, min(4096, N))
            for c in cond_cols:
                outs = []
                idx = idx_map[c]
                for s in range(0, N, chunk):
                    e = min(s + chunk, N)
                    x = idx[s:e].to(torch.long, copy=False).to(self.device, non_blocking=True)
                    y = self.model.embedding_layers[c](x)  # [e-s, emb_dim_c]
                    outs.append(y.detach().to("cpu"))
                    del x, y
                    torch.cuda.empty_cache()
                parts.append(torch.cat(outs, dim=0))
            cond_global = torch.cat(parts, dim=1).to(self.device) if parts else None
        else:
            cond_global = None

        # --- final conditioner [N, sum_c emb_dim_c + dim_embedding]
        cp = torch.cat((cond_global, pert_low), dim=1) if cond_global is not None else pert_low  # [N, C]

        # guard: width should equal sum(cov dims) + vspace_model.dim_embedding
        expected_C = (cond_global.shape[1] if cond_global is not None else 0) + self.vspace_model.dim_embedding
        if cp.shape[1] != expected_C:
            raise RuntimeError(f"Unexpected conditioner width {cp.shape[1]} != {expected_C} "
                            f"(sum(cov_dims) + dim_embedding).")

        # --- sample base point in latent PCA space
        D = int(self.data.num_pca_components or getattr(ds, "num_pca_components", 0) or 2)

        def _std_gauss(n): 
            return torch.randn(n, D, device=self.device)

        if self.conditional_noise:
            z0, _, _, _ = self.model.sample_noise_from_gmm(cp.to(torch.float32))
        else:
            z0 = _std_gauss(cp.shape[0])

        # unwrap tuple if sampler returns (samples, ...)
        # z0 = samp[0] if (isinstance(samp, (list, tuple)) and len(samp) >= 1) else samp
        z0 = z0.to(self.device)

        # --- integrate ODE and reshape to [num_samples, P, steps, D]
        t = torch.linspace(0, 1, steps=int(steps_ODEsolver), device=self.device)

        traj = torchdiffeq.odeint(
            lambda tt, zz: self.model.vspace.forward(zz, tt, cp),
            z0, t, atol=1e-4, rtol=1e-4, method="dopri5"
        )  # [T, N, D]

        traj = traj.permute(1, 0, 2).reshape(int(num_samples), int(P), int(steps_ODEsolver), D)
        return traj.detach().cpu().numpy(), cp.detach().cpu().numpy()


    # --------------------------- full generate (save h5ad) ---------------------------
    @torch.no_grad()
    def generate(
        self,
        chunks_gensample: int,
        num_samples: int,
        PCvisindex_1: int,
        PCvisindex_2: int,
        vspace_model_path: str,
        conditional_sampler_path: Optional[str] = None,
        conditions: str = "",
        akstr_trainortest: str = "test",
        output_path: str = "",
        steps_ODEsolver: int = 10,
        path_dumpfig: str | None = None,
        path_dumpnpy: str | None = None,
    ):
        assert akstr_trainortest in ("train", "test")
        self.model.lower_dim_embedding_perturbation.eval()
        self.model.vspace.eval()
        self.vspace_model.eval()

        if self.conditional_noise:
            self.model.module_H_theta.eval()
            self.model.module_fc1fc2.eval()
        

        self.vspace_model.load_state_dict(
            torch.load(vspace_model_path, map_location=self.device, weights_only=True)
        )
        if self.conditional_noise:
            if not conditional_sampler_path:
                raise ValueError("conditional_noise=True but no conditional_sampler_path was given.")
            ckpt = torch.load(conditional_sampler_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(ckpt, strict=True)

        # memory-safe final-only latent generation
        Z, Z_basedist_samples_allperts, np_pertsandcovars_names, np_all_Htheta_output = self._generate_final_in_chunks(
            num_samples=num_samples,
            conditions=conditions,
            steps_ODEsolver=steps_ODEsolver,
            max_batch_N=int(self.dict_kwargs_akDebug.get("max_gen_batchN", 32768)),
        )  # [N, D] on CPU (numpy)

        # print("np_all_Htheta_output.shape = {}".format(np_all_Htheta_output.shape))  # [N x num_modes x D]

        # dump the embeddings, for visualisation and/or evlauation ====
        if not(path_dumpnpy is None) or not(path_dumpfig is None):
            for pertsandcovars in set(np_pertsandcovars_names.tolist()):  # TODO:modify for the case that covariates are present
                adata_gt_population = None
                if pertsandcovars in self.data.gene_expression_dataset.adata_aktrain4eval.obs[self.data.gene_expression_dataset.perturbation_key].tolist():
                    adata_gt_population = self.data.gene_expression_dataset.adata_aktrain4eval.copy()
                else:
                    adata_gt_population = self.data.gene_expression_dataset.adata_akevaluation.copy()
                
                Z_generated = Z[(np_pertsandcovars_names==pertsandcovars).tolist(), :]
                Z_gt = self.data.gene_expression_dataset.pca.transform(
                    adata_gt_population[
                        adata_gt_population.obs[self.data.gene_expression_dataset.perturbation_key] == pertsandcovars
                    ].X.toarray()  # TODO:modify this whole block, when covariates are added.ss
                )
                Z_basedist_samples = Z_basedist_samples_allperts[(np_pertsandcovars_names==pertsandcovars).tolist(), :]

                if self.conditional_noise:
                    Htheta_output = np_all_Htheta_output[(np_pertsandcovars_names==pertsandcovars).tolist(), :]  # [n x num_modes x D]
                    for n in range(Htheta_output.shape[0]):
                        assert np.allclose(
                            Htheta_output[0,:],
                            Htheta_output[n,:]
                        )
                    
                    Htheta_output = Htheta_output[0,:,:]  # [num_modes x D]

                str_fname_pertsandcovars = pertsandcovars.replace("+", "__").replace("/", "").replace("\\", "").replace("&", "")

                if not (path_dumpnpy is None):
                    os.makedirs(path_dumpnpy, exist_ok=True)
                    with open(
                        os.path.join(
                            path_dumpnpy,
                            "{}.pkl".format(str_fname_pertsandcovars)
                        ),
                        'wb'
                    ) as f:
                        pickle.dump(
                            {'Z_generated':Z_generated, 'Z_gt':Z_gt},
                            f
                        )

                if not (path_dumpfig is None):
                    os.makedirs(path_dumpfig, exist_ok=True)

                    # compute the xlim 
                    xlim_min = min(
                        Z_generated[:, PCvisindex_1].min(),
                        Z_gt[:, PCvisindex_1].min(),
                        np.inf, # base dist samples may have ranges which are off, hence commented out: Z_basedist_samples[:, PCvisindex_1].min(),
                        Htheta_output[:, PCvisindex_1].min() if self.conditional_noise else np.inf
                    )
                    xlim_max = max(
                        Z_generated[:, PCvisindex_1].max(),
                        Z_gt[:, PCvisindex_1].max(),
                        -np.inf, # base dist samples may have ranges which are off, hence commented out: Z_basedist_samples[:, PCvisindex_1].max(),
                        Htheta_output[:, PCvisindex_1].max() if self.conditional_noise else -np.inf
                    )

                    ylim_min = min(
                        Z_generated[:, PCvisindex_2].min(),
                        Z_gt[:, PCvisindex_2].min(),
                        np.inf, # base dist samples may have ranges which are off, hence commented out: Z_basedist_samples[:, PCvisindex_2].min(),
                        Htheta_output[:, PCvisindex_2].min() if self.conditional_noise else np.inf
                    )
                    ylim_max = max(
                        Z_generated[:, PCvisindex_2].max(),
                        Z_gt[:, PCvisindex_2].max(),
                        -np.inf, # base dist samples may have ranges which are off, hence commented out: Z_basedist_samples[:, PCvisindex_2].max(),
                        Htheta_output[:, PCvisindex_2].max() if self.conditional_noise else -np.inf
                    )
                    

                    plt.figure(figsize=[8,8])

                    plt.subplot(2,2,1)
                    if self.conditional_noise:
                        plt.scatter(
                            Htheta_output[:, PCvisindex_1],
                            Htheta_output[:, PCvisindex_2]
                        )
                        plt.xlim([xlim_min, xlim_max])
                        plt.ylim([ylim_min, ylim_max])
                        plt.title("Htheta_output (i.e. modes)")

                    plt.subplot(2,2,2)
                    plt.scatter(
                        Z_basedist_samples[:, PCvisindex_1],
                        Z_basedist_samples[:, PCvisindex_2]
                    )

                    # basedist samples may have ranges which are quite 'off', hence commenting out the following lines
                    # plt.xlim([xlim_min, xlim_max])
                    # plt.ylim([ylim_min, ylim_max])
                    
                    plt.title("Base dist samples, Perts&Covars: \n{} \n(population size:{})".format(pertsandcovars, Z_generated.shape[0]))

                    plt.subplot(2,2,3)
                    plt.scatter(
                        Z_generated[:, PCvisindex_1],
                        Z_generated[:, PCvisindex_2]
                    )
                    plt.xlim([xlim_min, xlim_max])
                    plt.ylim([ylim_min, ylim_max])
                    plt.title("Generated, Perts&Covars: \n{} \n(population size:{})".format(pertsandcovars, Z_generated.shape[0]))

                    plt.subplot(2,2,4)
                    plt.scatter(
                        Z_gt[:, PCvisindex_1],
                        Z_gt[:, PCvisindex_2]
                    )
                    plt.xlim([xlim_min, xlim_max])
                    plt.ylim([ylim_min, ylim_max])
                    plt.title("Groundtruth, Perts&Covars: \n{} \n (population size: {})".format(pertsandcovars, Z_gt.shape[0]))

                    plt.savefig(
                        os.path.join(
                            path_dumpfig,
                            '{}.png'.format(
                                str_fname_pertsandcovars
                            )
                        ),
                        bbox_inches='tight',
                        pad_inches=0
                    )
                
                del adata_gt_population
                gc.collect(); gc.collect(); gc.collect()            

        

        # inverse PCA back to expression
        ds = self.data.gene_expression_dataset
        X = ds.pca.inverse_transform(Z)  # [N, G]
        expr = torch.tensor(X, dtype=torch.float32, device="cpu")

        # rebuild obs for saving
        df = pd.read_csv(conditions)
        perts = df[ds.perturbation_key].astype(str).tolist()
        dosages = (df["dosage"].to_numpy(dtype=np.float32)
                if "dosage" in df.columns else np.ones(len(df), dtype=np.float32))
        obs = {
            "perturbation": np.tile(perts, int(num_samples)),
            "dosages":      np.tile(dosages, int(num_samples)),
        }
        G = ad.AnnData(X=expr.numpy(), obs=obs)
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            G.write(output_path)
        self.output = G

        self.model.lower_dim_embedding_perturbation.train()
        self.model.vspace.train()
        self.vspace_model.train()

        if self.conditional_noise:
            self.model.module_H_theta.train()
            self.model.module_fc1fc2.train()
        
        


    # --------------------------- evaluation ---------------------------

    def _align_varnames_like(self, real_adata, gen_adata):
        rv = pd.Index(real_adata.var_names.astype(str))
        gv = pd.Index(gen_adata.var_names.astype(str))
        inter = rv.intersection(gv)
        if len(inter) > 0:
            real_adata = real_adata[:, inter]
            gen_adata = gen_adata[:, inter]
            gen_adata = gen_adata[:, pd.Index(gen_adata.var_names).get_indexer(real_adata.var_names)]
            return real_adata, gen_adata
        gen_adata.var_names = real_adata.var_names.copy()
        gen_adata.var.index = real_adata.var.index.copy()
        return real_adata, gen_adata

    def _compute_control_means(self, adata, pert_col, control_label, strata_cols=None):
        strata_cols = [c for c in (strata_cols or []) if c in adata.obs.columns]
        idx_ctrl_global = (adata.obs[pert_col] == control_label).to_numpy()
        global_vec = _to_dense(adata[idx_ctrl_global].X).mean(axis=0) if idx_ctrl_global.any() else np.zeros(adata.n_vars, float)
        if not strata_cols:
            return {'__global__': global_vec}
        baselines = {}
        sub = adata.obs[strata_cols].astype(str).drop_duplicates()
        for _, row in sub.iterrows():
            mask = np.ones(adata.n_obs, dtype=bool)
            for c in strata_cols:
                mask &= (adata.obs[c].astype(str).to_numpy() == str(row[c]))
            mask &= (adata.obs[pert_col].to_numpy() == control_label)
            vec = _to_dense(adata[mask].X).mean(axis=0) if mask.any() else global_vec
            baselines[tuple(str(row[c]) for c in strata_cols)] = vec
        return baselines

    def _apply_baseline_per_strata(self, X, obs_df, baselines, strata_cols, mode="subtract"):
        X = _to_dense(X)
        sign = -1 if mode == "subtract" else +1
        if '__global__' in baselines:
            return X + sign * baselines['__global__']
        keys = obs_df[strata_cols].astype(str).apply(lambda r: tuple(r.values.tolist()), axis=1).values
        uniq, inv = np.unique(keys, return_inverse=True)
        for i, k in enumerate(uniq):
            mask = (inv == i)
            vec = baselines.get(k, None)
            if vec is not None:
                X[mask, :] += sign * vec
        return X

    def evaluate(self, delta: bool = False, plot: bool = False, DEG: dict | None = None):
        data = self.data.gene_expression_dataset.adata.copy()
        generated = self.output.copy()
        data, generated = self._align_varnames_like(data, generated)

        pert_col = self.data.perturbation_key
        split_key = self.data.split_key
        control = self.data.control

        order_cols = []
        if "cell_type" in data.obs.columns and "cell_type" in generated.obs.columns:
            order_cols.append("cell_type")
        for c in (self.data.condition_keys or []):
            if c in data.obs.columns and c in generated.obs.columns:
                order_cols.append(c)

        if delta:
            b = self._compute_control_means(data, pert_col, control, strata_cols=order_cols)
            data.X = self._apply_baseline_per_strata(data.X, data.obs, b, strata_cols=order_cols, mode="subtract")
        else:
            b = self._compute_control_means(data, pert_col, control, strata_cols=order_cols)
            generated.X = self._apply_baseline_per_strata(generated.X, generated.obs, b, strata_cols=order_cols, mode="add")

        is_test = (data.obs[split_key].astype(str) == "test").to_numpy()
        test_data = data[is_test].copy()

        if "perturbation" not in generated.obs.columns:
            raise KeyError("'perturbation' column not found in generated data.")
        generated.obs[pert_col] = generated.obs["perturbation"].astype(test_data.obs[pert_col].dtype)

        def _means_masks(adata, cols):
            means, masks = {}, {}
            df = adata.obs[[pert_col] + cols].astype(str)
            for _, row in df.drop_duplicates().iterrows():
                pert = row[pert_col]
                key = "####".join([pert] + [row[c] for c in cols])
                mask = (adata.obs[pert_col].astype(str) == pert).to_numpy()
                for c in cols:
                    mask &= (adata.obs[c].astype(str) == str(row[c])).to_numpy()
                if mask.any():
                    masks[key] = mask
                    means[key] = _to_dense(adata[mask].X).mean(axis=0)
            return means, masks

        real_means, real_masks = _means_masks(test_data, order_cols)
        gen_means, gen_masks = _means_masks(generated, order_cols)
        common = sorted(set(real_means).intersection(gen_means))
        if not common:
            raise ValueError("No common (pert + covariates) between real TEST and generated.")

        from metrics import compute_metrics

        w1 = []; w2 = []; mmd = []; energy = []
        pearson_corr = []; pearson_p = []
        spearman_corr = []; spearman_p = []
        mse_val = []
        vnames = pd.Index(test_data.var_names.astype(str))

        def maybe_filter(om, gm, td, gd, key):
            if DEG is None:
                return om, gm, td, gd
            deg = DEG.get(key) or DEG.get(key.split("####", 1)[0])
            """
            TODO:aknote:BUGBUGBUGBUGBUGBUGBUG
            Although in my run the cell_type covariate is enabled, here `key` is a single perturbation name,
               i.e. without the #### separator. :| 
            """
            if deg is None:
                return om, gm, td, gd
            names = None
            if isinstance(deg, dict):
                names = deg.get("names", None)
            elif hasattr(deg, "columns") and "names" in deg.columns:
                names = deg["names"]
            else:
                names = deg
            if hasattr(names, "tolist"):
                names = names.tolist()
            if not names:
                return om, gm, td, gd
            mask = np.asarray(vnames.isin([str(x) for x in names]), dtype=bool)
            if not mask.any():
                return om, gm, td, gd
            return om[mask], gm[mask], td[:, mask], gd[:, mask]
        
        for key in common:
            td = _to_dense(test_data.X[real_masks[key], :])
            gd = _to_dense(generated.X[gen_masks[key], :])
            om = real_means[key]; gm = gen_means[key]
            om, gm, td, gd = maybe_filter(om, gm, td, gd, key)

            """
            TODO:aknote:BUGBUGBUGBUGBUGBUGBUG
            Although there are 50 precomputed DEGs, here `td` and `gd` are of shape [N x 2000]!
            """
            
            w1.append({key: compute_metrics(td, gd, 'w1')})
            w2.append({key: compute_metrics(td, gd, 'w2')})
            mmd.append({key: compute_metrics(td, gd, 'mmd')})
            energy.append({key: compute_metrics(td, gd, 'energy')})

            pc, pcp = sstats.pearsonr(om, gm)
            sc, scp = sstats.spearmanr(om, gm)
            pearson_corr.append({key: pc}); pearson_p.append({key: pcp})
            spearman_corr.append({key: sc}); spearman_p.append({key: scp})
            mse_val.append({key: mean_squared_error(om, gm)})

        def _m(lst): 
            return float("nan") if not lst else float(np.mean([list(d.values())[0] for d in lst]))

        print(f"Mean Pearson: {_m(pearson_corr):.4f} (p={_m(pearson_p):.4g})")
        print(f"Mean Spearman: {_m(spearman_corr):.4f} (p={_m(spearman_p):.4g})")
        print(f"Mean MSE: {_m(mse_val):.4f}")
        print(f"Wasserstein-1: {_m(w1):.4f}")
        print(f"Wasserstein-2: {_m(w2):.4f}")
        print(f"MMD: {_m(mmd):.4f}")
        print(f"Energy: {_m(energy):.4f}")

        return dict(pearson_corr=pearson_corr, spearman_corr=spearman_corr, mse_val=mse_val,
                    w1=w1, w2=w2, mmd=mmd, energy=energy)
