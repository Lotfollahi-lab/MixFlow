import os, sys, time, argparse, types, random, warnings, pickle
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import json
import torch
import torch.nn as nn

sys.path.extend([p for p in ["../", "../../"] if p not in sys.path])

from interface import PertFlow, PerturbedGeneExpression, CalcModeGeodesicLength
from utils import get_embeddings, adata_preprocessing_ak
from models import ModuleEll2Normalise

warnings.filterwarnings("ignore")


def seed_all(seed: int = 42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_num_modes_from_meta(meta_path: str):
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        nm = meta.get("num_modes", None)
        return int(nm) if nm is not None else None
    except Exception:
        return None

def infer_num_modes_from_state_dict(state_dict: dict, fallback: int | None = None) -> int | None:
    from collections import Counter
    if not isinstance(state_dict, dict):
        return fallback
    keywords = ('mode', 'modes', 'mixt', 'gmm', 'component', 'base', 'pi', 'logit', 'mean', 'var')
    cand, all_leads = [], []
    for k, v in state_dict.items():
        if not isinstance(v, torch.Tensor) or v.dim() < 1: 
            continue
        lead = v.shape[0]
        all_leads.append(lead)
        lk = k.lower()
        if any(w in lk for w in keywords):
            cand.append(lead)
    pool = cand or all_leads
    if not pool:
        return fallback
    return Counter(pool).most_common(1)[0][0]

def parse_covariates_arg(val):
    if val is None: return []
    if isinstance(val, list): return val
    s = str(val).strip()
    if s in ("None", "none", "", "[]"): return []
    try:
        obj = eval(s, {}, {})
        if obj is None: return []
        if isinstance(obj, (list, tuple)): return list(obj)
        return [str(obj)]
    except Exception:
        return [t for t in s.split(",") if t]

def parse_num_modes(val):
    s = str(val).strip().lower()
    if s in ("auto", "", "none"):
        return None
    return int(val)


def _x_is_integer_like(X) -> bool:
    """
    Return True if matrix X (dense or sparse) contains integer-valued entries.
    """
    if sp.issparse(X):
        data = X.data
        if data.size == 0:
            return True
        if np.issubdtype(data.dtype, np.integer):
            return True
        if np.issubdtype(data.dtype, np.floating):
            return np.all(np.mod(data, 1) == 0)
        return False
    else:
        if np.issubdtype(X.dtype, np.integer):
            return True
        if np.issubdtype(X.dtype, np.floating):
            return np.all(np.mod(X, 1) == 0)
        return False


def main():
    t0 = time.time()

    p = argparse.ArgumentParser()
    # Minimal required arguments: a run name + dataset/output paths.
    # Everything else defaults to the values used in our "docs/running.md" example.
    p.add_argument('--str_runname', type=str, required=True)
    p.add_argument('--delta_space', type=str, choices=['True', 'False'], default='False')
    p.add_argument('--path_output_workspace', type=str, required=True)
    p.add_argument('--path_split', type=str, required=True)
    p.add_argument('--mode_DEGs', type=str, choices=['PRECOMPUTED', 'STATTEST', 'SIMPLEDIFF'], default='SIMPLEDIFF')
    p.add_argument('--path_DEGs_precomputed', type=str, default='None')
    p.add_argument('--targetsize_mini_batch', type=str, default='None')
    p.add_argument('--num_repeats_selcondition', type=int, default=10)
    p.add_argument('--maxsize_split_population', type=int, default=100)
    p.add_argument('--num_pca_components', type=int, default=20)
    p.add_argument('--epochs', type=int, default=1000)
    p.add_argument('--dict_kwargs_optim_flow', type=str, default="{'lr':0.001, 'betas':(0.0, 0.99)}")
    p.add_argument('--dict_kwargs_optim_basedist', type=str, default="{'lr':0.001}")
    p.add_argument('--method_OTsampler', type=str, default='exact')
    p.add_argument('--flag_condition_basedist', type=str, choices=['True', 'False'], default='True')
    p.add_argument('--prob_dropout_conditionencoder', type=float, default=0.5)
    p.add_argument('--embedding_path', type=str, required=True)
    p.add_argument('--perturbation_key', type=str, default='condition')
    p.add_argument('--control_key', type=str, default='ctrl')
    p.add_argument('--sep', type=str, default='+')
    p.add_argument('--covariates', type=str, default='[]')
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--skip_if_fitted', action='store_true',
                   help='If set, and pretrained weights already exist at destination, '
                        'resume training from them (continue training) instead of starting fresh.')
    p.add_argument('--skip_if_pretrained', action='store_true',
                   help='If set, and pretrained weights already exist at destination, '
                        'skip training and only run generate + evaluate.')
    p.add_argument(
        '--float_fixedvariance_GMMs',
        type=float,
        help='ddd.',
        default=0.01
    )
    p.add_argument(
        '--num_modes_basedistGMM',
        type=int,
        help='ddd.',
        default=20
    )
    p.add_argument(
        '--str_kwargs_GeodesicLen',
        type=str,
        help='ddd.',
        default="{'str_integfromX0_or_integfromX1':'integfromX0', 'steps_ODEsolver':3, 'mode_compute_geodlen':CalcModeGeodesicLength.ELL2NORM_U.value, 'coef_FMloss':0.0}"
    )
    p.add_argument(
        '--steps_ODEsolver_evaluationphase',
        type=int,
        help='ddd.',
        default=100
    )
    p.add_argument(
        '--kwargs_trainsep_Htheta',  # e.g., {'batch_size': 100, 'epochs': 20, 'lr': 0.001, 'stddiv_noise_fit2_cluster_modes': 0.01}
        type=str,
        help='ddd.',
        default="{'batch_size': 100, 'epochs': 1000, 'lr': 0.001, 'stddiv_noise_fit2_cluster_modes': 0.1}"
    )
    p.add_argument(
        '--prob_dropout_fc1fc2',
        type=float,
        help='ddd.',
        default=0.0
    )

    p.add_argument(
        '--kwargs_scheduler_updatebasedist',
        type=str,
        help='ddd.',
        default="{'itercycle_update_basedist':10, 'flag_cooldown_settoeval_HthetaHp':True, 'dict_intervalname_to_fracepochs':{'warmup':0.1, 'bothVandB':0.8, 'cooldown':0.1}}"
    )
    
    p.add_argument(
        '--moduleV_list_dim_hidden',
        type=str,
        help='ddd.',
        default='[200, 200, 200]'
    )  # e.g., '[200, 200, 200]'
    p.add_argument(
        '--temperature_gumble_softmax',
        type=float,
        help='ddd.',
        default=0.5
    )
    p.add_argument(
        '--kwargs_Htheta',
        type=str,
        help='ddd.',
        default="{'prob_dropout_Htheta':0.0}"
    )
    p.add_argument(
        '--gumbel_softmax_flag_hard',
        type=str,
        help='ddd.',
        default='False'
    )
    p.add_argument(
        '--flag_rapidssc_KMeans',
        type=str,
        help='ddd.',
        default='False'
    )
    p.add_argument(
        '--flag_rapidssc_pca',
        type=str,
        help='ddd.',
        default='False'
    )
    args = p.parse_args()


    print(f"Imports + arg parsing took {time.time() - t0:.2f}s")

    args.flag_condition_basedist = (args.flag_condition_basedist == 'True')
    assert isinstance(args.gumbel_softmax_flag_hard, str)
    assert args.gumbel_softmax_flag_hard in ['True', 'False']

    assert isinstance(args.flag_rapidssc_KMeans, str)
    assert args.flag_rapidssc_KMeans in ['True', 'False']
    args.flag_rapidssc_KMeans = (args.flag_rapidssc_KMeans == 'True')
    assert args.flag_rapidssc_KMeans in [True, False]


    assert isinstance(args.flag_rapidssc_pca, str)
    assert args.flag_rapidssc_pca in ['True', 'False']
    args.flag_rapidssc_pca = (args.flag_rapidssc_pca == 'True')
    assert args.flag_rapidssc_pca in [True, False]
    args.delta_space = (args.delta_space == 'True')
    assert args.delta_space in [True, False]

    # Load split
    adata_path = os.path.join(args.path_split, 'adata.h5ad')
    adata = sc.read_h5ad(adata_path)
    print(f"Loaded dataset with {adata.n_obs} cells and {adata.n_vars} genes.")

    # Ensure integer-valued counts in .X; populate layers['counts'] if needed; ensure CSR
    if not _x_is_integer_like(adata.X):
        if 'counts' in adata.layers and _x_is_integer_like(adata.layers['counts']):
            adata.X = adata.layers['counts'].copy()
        else:
            raise ValueError("The .X matrix is not integer-like, and no integer-like 'counts' layer is found.")
    
    # sparse CSR normalize
    if not sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)
    if 'counts' in adata.layers:
        if not sp.issparse(adata.layers['counts']):
            adata.layers['counts'] = sp.csr_matrix(adata.layers['counts'])
    else:
        adata.layers['counts'] = adata.X.copy()

    # Ensure `.obs['split']` contains only 'train' and 'test' (and not, e.g., 'ood')
    assert 'split' in set(adata.obs.columns.tolist()), print("`adata.obs` must have a column called `split`.")
    # check if only 'train' and 'test' are present
    # if val or ood are present, move to test
    adata.obs['split'] = adata.obs['split'].replace({'val': 'test', 'ood': 'test'})
    assert set(adata.obs['split'].tolist()) == {'train', 'test'}, \
        print("`adata.obs['split'] values must be `train` or `test`, while the values are: {}".format(
            set(adata.obs['split'].tolist())
        ))

    # Ensure there are no nan values in the perturbation column
    if adata.obs[args.perturbation_key].isna().any():
        raise Exception("Found nan value in `adata.obs['{}']`".format(args.perturbation_key))
    
    # Ensure there are no nan values in covariate columns
    for covarname in eval(args.covariates):
        if adata.obs[covarname].isna().any():
            raise Exception("Found nan value in `adata.obs['{}']`".format(covarname))
    
    # Ensure there are no slash or back-slash character in perturbations or covariate columns
    for colname in [args.perturbation_key] + eval(args.covariates):
        for val in set(adata.obs[colname].tolist()):
            for ch in ['/', '\\']:
                if ch in val:
                    raise Exception(
                        "Found invalid character '{}' in value {} in `adata.obs['{}']`.".format(
                            ch,
                            val,
                            colname
                        )
                    )


    # Preprocessing
    adata = adata_preprocessing_ak(adata)

    column_pert = args.perturbation_key  # or "condition"
    if column_pert != "condition":
        adata.obs['condition'] = adata.obs[column_pert].copy()
    ctrl = args.control_key  # or "ctrl"
    # check if ctrl exist as ctrl-sep-ctrl in the data,
    # if yes, change ctrl-sep-ctrl to just ctrl in the data
    ctrl_sep_ctrl = f"{ctrl}{args.sep}{ctrl}"
    if ctrl_sep_ctrl in adata.obs[column_pert].values:
        adata.obs[column_pert] = adata.obs[column_pert].replace({ctrl_sep_ctrl: ctrl})
    sep_literal = args.sep  #  if (args.sep and args.sep != "") else "+"

    # DEGs
    mode = args.mode_DEGs
    print(f"Computing DEGs in mode {mode}")
    if mode == 'STATTEST':
        adata.obs[column_pert] = adata.obs[column_pert].astype('category')
        sc.tl.rank_genes_groups(adata, groupby=column_pert, method='wilcoxon', use_raw=False)
        de_results_per_condition = {
            str(cond): pd.DataFrame({
                'gene': adata.uns['rank_genes_groups']['names'][cond],
                'logfoldchanges': adata.uns['rank_genes_groups']['logfoldchanges'][cond],
                'pvals_adj': adata.uns['rank_genes_groups']['pvals_adj'][cond],
            }).sort_values('pvals_adj')
            for cond in adata.obs[column_pert].cat.categories
        }
    elif mode == 'PRECOMPUTED':
        if not args.path_DEGs_precomputed:
            raise ValueError("PRECOMPUTED requires --path_DEGs_precomputed")
        with open(args.path_DEGs_precomputed, "rb") as f:
            obj = pickle.load(f)
        de_results_per_condition = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                names = None
                if isinstance(v, dict):
                    names = v.get("names", None)
                    if names is None and "names" in v:
                        names = v["names"]
                if names is None and hasattr(v, "columns"):
                    for alt in ("gene", "gene_name", "symbol", "genes", "ENSEMBL", "ensembl_id"):
                        if alt in v.columns:
                            names = v[alt]; break
                if hasattr(names, "tolist"):
                    names = names.tolist()
                if names:
                    de_results_per_condition[str(k)] = {"names": [str(x) for x in names]}
        else:
            de_results_per_condition = {}
    else:
        # SIMPLEDIFF
        de_results_per_condition = {}
    
    # Embeddings    
    adata = get_embeddings(adata, column_pert, ctrl, args.embedding_path, sep_literal)

    # Output dirs
    if '$' in args.path_output_workspace:
        raise Exception(
            "Found invalid character '$' in `args.path_output_workspace.`\n"+\
            "Maybe you have wrapped that argument in single quote rather than double quote?"
        )
    os.makedirs(os.path.join(args.path_output_workspace, "weights/"), exist_ok=True)
    os.makedirs(os.path.join(args.path_output_workspace, "output/"), exist_ok=True)
    os.makedirs(os.path.join(args.path_output_workspace, "weights/FinalResultFigs/"), exist_ok=True)
    os.makedirs(
        os.path.join(
            args.path_output_workspace,
            f"weights/FinalResultFigs/FinalResultFigs_{args.str_runname}"
        ),
        exist_ok=True
    )

    # Persist ready adata for the interface
    fname_ready = os.path.join(args.path_output_workspace, f"{args.str_runname}_4pertflowAPI.h5ad")
    adata.write(fname_ready)

    covariates = parse_covariates_arg(args.covariates)

    # Dataset wrapper
    data = PerturbedGeneExpression(
        file_path=fname_ready,
        perturbation_key=column_pert,
        condition_keys=covariates,           # [] if none
        dosage_key=None,
        control=ctrl,
        flag_rapidssc_pca=args.flag_rapidssc_pca,
        sep=sep_literal,                      # literal; dataset builds regex-safe internally
        split_key='split',
        num_pca_components=int(args.num_pca_components),
        delta=args.delta_space,
        use_cuda=True,
        dict_resample_population_incond=({'targetsize_mini_batch': int(args.targetsize_mini_batch)}
                                         if args.targetsize_mini_batch != "None" else None),
        num_repeats_selcondition=int(args.num_repeats_selcondition),
        maxsize_split_population=int(args.maxsize_split_population),
    )

    # Dump train/test condition CSVs
    def get_conditions(_adata, _pert, _covs):
        _covs = _covs or []
        if "dosage" not in _adata.obs.columns:
            _adata = _adata.copy(); _adata.obs["dosage"] = 1
        cols = [_pert] + _covs + ["dosage"]
        return _adata.obs[cols].drop_duplicates().reset_index(drop=True)

    cond_train = get_conditions(data.gene_expression_dataset.adata_train, column_pert, covariates)
    cond_train_csv = os.path.join(
        args.path_output_workspace,
        f"weights/conditions_{args.str_runname}_akDebug_train.csv"
    )
    cond_train.to_csv(cond_train_csv, index=False)

    cond_test = get_conditions(data.gene_expression_dataset.adata_test, column_pert, covariates)
    cond_test_csv = os.path.join(
        args.path_output_workspace,
        f"weights/conditions_{args.str_runname}_akDebug_test.csv"
    )
    cond_test.to_csv(cond_test_csv, index=False)

    # Paths for weights
    conditional_sampler_path = os.path.join(args.path_output_workspace, "weights", f"conditional_sampler_{args.str_runname}.pth")
    vspace_model_path = os.path.join(
        args.path_output_workspace,
        "weights",
        f"vspace_model_{args.str_runname}.pth"
    )

    def _pretrained_weights_exist(flag_condition_basedist: bool) -> bool:
        v_ok = os.path.isfile(vspace_model_path)
        if not flag_condition_basedist:
            return v_ok
        c_ok = os.path.isfile(conditional_sampler_path)
        return v_ok and c_ok

    have_pretrained = _pretrained_weights_exist(args.flag_condition_basedist)

    # Model
    model = PertFlow(
        float_fixedvariance_GMMs=args.float_fixedvariance_GMMs,
        kwargs_GeodesicLen=eval(args.str_kwargs_GeodesicLen),
        data=data,
        hidden_dims=eval(args.moduleV_list_dim_hidden),
        prob_dropout_conditionencoder=float(args.prob_dropout_conditionencoder),
        kwargs_Htheta=eval(args.kwargs_Htheta),
        prob_dropout_fc1fc2=float(args.prob_dropout_fc1fc2),
        dim_perturbations=128,
        conditional_noise=bool(args.flag_condition_basedist),
        num_modes=args.num_modes_basedistGMM,  
        temperature=args.temperature_gumble_softmax,
        dict_kwargs_akDebug={},
        numupdates_each_minibatch=1,
        kwargs_OT_sampler={'method': args.method_OTsampler},
        kwargs_scheduler_updatebasedist={
            **(eval(args.kwargs_scheduler_updatebasedist)),
            **{'total_epochs':int(args.epochs)}
        },
        kwargs_initial_kmeans_for_Htheta={
            'n_clusters':args.num_modes_basedistGMM
        },
        gumbel_softmax_flag_hard=(args.gumbel_softmax_flag_hard == 'True'),
        flag_rapidssc_KMeans=args.flag_rapidssc_KMeans
    )

    # Optional tiny dry run: skip if reusing weights 
    if not ((args.skip_if_pretrained or args.skip_if_fitted) and have_pretrained):
        _ = model.generate_anycsv(
            num_samples=2,
            conditions=cond_train_csv,
            steps_ODEsolver=3
        )

    if args.skip_if_pretrained and have_pretrained:
        print("[INFO] Pretrained weights found and --skip_if_pretrained set. "
              "Skipping training and proceeding to generation/evaluation.")
        resume_flag = False
    else:
        if args.skip_if_pretrained and not have_pretrained:
            print("[INFO] --skip_if_pretrained set but pretrained weights not found. Proceeding to training from scratch.")

        # If user asked to continue (resume) and weights exist, we resume; else start fresh.
        resume_flag = (args.skip_if_fitted and have_pretrained)
        if args.skip_if_fitted and have_pretrained:
            print("[INFO] Pretrained weights found and --skip_if_fitted set. "
                  "Resuming training from existing weights.")
        elif args.skip_if_fitted and not have_pretrained:
            print("[INFO] --skip_if_fitted set but no pretrained weights found. Training from scratch.")

        # Only call train if we didn't already skip due to --skip_if_pretrained+weights
        if not (args.skip_if_pretrained and have_pretrained):
            model.train(
                batch_size=1500,
                epochs=int(args.epochs),
                dict_kwargs_optim_flow=eval(args.dict_kwargs_optim_flow),
                dict_kwargs_optim_basedist=eval(args.dict_kwargs_optim_basedist),
                min_delta=1e-3,
                patience=10,
                board=True,
                resume=resume_flag,
                conditional_sampler_path=conditional_sampler_path,
                vspace_model_path=vspace_model_path,
                wandb_runname=args.str_runname,
                kwargs_trainsep_Htheta={
                    **{
                        'min_delta': 0.0,
                        'patience': 10,
                        'hyperparameters': None,
                        'resume': False
                    },
                    **eval(args.kwargs_trainsep_Htheta)
                }
            )

    # Generate + save
    def _maybe_conditional_path(use_cond_noise: bool, path: str):
        return path if (use_cond_noise and os.path.isfile(path)) else None

    # Train set generation
    model.generate(
        chunks_gensample=500,
        num_samples=1000,
        PCvisindex_1=0, PCvisindex_2=1,
        vspace_model_path=vspace_model_path,
        conditional_sampler_path=_maybe_conditional_path(args.flag_condition_basedist, conditional_sampler_path),
        conditions=cond_train_csv,
        akstr_trainortest='train',
        output_path=f'./NonGit/output/GeneExpression_{args.str_runname}_train.h5ad',
        steps_ODEsolver=args.steps_ODEsolver_evaluationphase,
        path_dumpfig=os.path.join(
            args.path_output_workspace,
            f"weights/FinalResultFigs/FinalResultFigs_{args.str_runname}/Train/"
        ),
        path_dumpnpy=os.path.join(
            args.path_output_workspace,
            f"weights/FinalResultFigs/FinalResultFigs_{args.str_runname}/Train/"
        )
    )

    # Test set generation
    model.generate(
        chunks_gensample=500,
        num_samples=1000,
        PCvisindex_1=0, PCvisindex_2=1,
        vspace_model_path=vspace_model_path,
        conditional_sampler_path=_maybe_conditional_path(args.flag_condition_basedist, conditional_sampler_path),
        conditions=cond_test_csv,
        akstr_trainortest='test',
        output_path=os.path.join(
            args.path_output_workspace,
            f'output/GeneExpression_{args.str_runname}_test.h5ad'
        ),
        steps_ODEsolver=args.steps_ODEsolver_evaluationphase,
        path_dumpfig=os.path.join(
            args.path_output_workspace,
            f"weights/FinalResultFigs/FinalResultFigs_{args.str_runname}/Test/"
        ),
        path_dumpnpy=os.path.join(
            args.path_output_workspace,
            f"weights/FinalResultFigs/FinalResultFigs_{args.str_runname}/Test/"
        )
    )

    mode = args.mode_DEGs
    if de_results_per_condition:
        metrics_DEGs = model.evaluate(
            plot=False, delta=False,#True,
            DEG=de_results_per_condition if mode == 'PRECOMPUTED' else None
        )
    metrics_allgenes = model.evaluate(plot=False, 
                                      delta=False,#True,
                                      DEG=None)
    with open(
        os.path.join(
            args.path_output_workspace,
            f'weights/FinalResultFigs/FinalResultFigs_{args.str_runname}/metrics_allgenes.pkl'
        ),
        'wb'
    ) as f:
        pickle.dump(metrics_allgenes, f)
    
    if de_results_per_condition:
        with open(
            os.path.join(
                args.path_output_workspace,
                f'weights/FinalResultFigs/FinalResultFigs_{args.str_runname}/metrics_DEGs.pkl'
            ),
            'wb'
        ) as f:
            pickle.dump(metrics_DEGs, f)

    # create a file indicating that the run has finished successfully
    with open(
        os.path.join(
            args.path_output_workspace,
            f'weights/FinalResultFigs/FinalResultFigs_{args.str_runname}/script_finished_successfully.txt'
        ),
        'w'
    ) as f:
        f.write("DONE")
    
    print("DONE")


if __name__ == "__main__":
    main()