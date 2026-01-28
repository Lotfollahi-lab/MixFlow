# Running SP-FM



This page shows a minimal example of how we currently run SP-FM on the **Norman** CRISPR screen dataset.


## Norman dataset example

We release Norman's data splits in `data/Norman.tar.gz`. 

### Minimal command 

```bash
python trainer/perturbations_trainer.py \
  --str_runname "example_norman" \
  --path_output_workspace "./Norman/NonGit/" \
  --path_split "./data/Norman/Split_1/" \
  --embedding_path "./data/GenePT_gene_embeddings.pickle" \
```

#### Required parameters 

- `--str_runname`: A unique name for this run (used to name output folders/files and for bookkeeping).
- `--path_output_workspace`: Output directory where results/checkpoints/logs are written.
- `--path_split`: Path to the dataset split folder/file used for training/evaluation (e.g. a prepared `.h5ad` split).
- `--embedding_path`: Path to the gene embedding file (e.g. a pickled embedding lookup).

#### Default parameters 

All other flags/parameters default to the values shown in the full example command above. Here is what those defaults mean at a high level:

- `--num_modes_basedistGMM` (default: `20`): Number of mixture components in the base distribution GMM.
- `--flag_condition_basedist` (default: `'True'`): Use a condition-dependent base distribution (condition encoder).
- `--prob_dropout_conditionencoder` (default: `0.5`): Dropout probability in the condition encoder network.
- `--method_OTsampler` (default: `'exact'`): OT sampler method used during training.
- `--kwargs_scheduler_updatebasedist` (default: `"{'itercycle_update_basedist':10, 'flag_cooldown_settoeval_HthetaHp':False, 'dict_intervalname_to_fracepochs':{'warmup':0.05, 'bothVandB':0.85, 'cooldown':0.1}}"`): Scheduler controlling when/how the base distribution is updated.
- `--num_pca_components` (default: `20`): PCA dimensionality used for preprocessing / latent representation.
- `--dict_kwargs_optim_flow` (default: `{'lr':0.001, 'betas':(0.0, 0.99)}`): Optimizer hyperparameters for the flow model.
- `--dict_kwargs_optim_basedist` (default: `{'lr':0.001}`): Optimizer hyperparameters for the base distribution model.
- `--epochs` (default: `1000`): Number of training epochs.
- `--num_repeats_selcondition` (default: `10`): Number of repeats when sampling/selecting conditions for evaluation.
- `--kwargs_trainsep_Htheta` (default: the dict in the example): Separate training config for $H_\theta$ (batch size, epochs, lr, and noise level for fitting cluster modes).
- `--prob_dropout_fc1fc2` (default: `0.0`): Dropout probability in the main MLP layers (fc1/fc2).
- `--steps_ODEsolver_evaluationphase` (default: `100`): Number of ODE solver steps used during evaluation.
- `--perturbation_key` (default: `'condition'`): AnnData obs key that stores the perturbation/condition label.
- `--control_key` (default: `'ctrl'`): Label used to identify controls in the perturbation column.
- `--sep` (default: `'+'`): Separator for combinatorial perturbations (e.g. `geneA+geneB`).
- `--maxsize_split_population` (default: `100`): Cap on population size when creating/using splits (helps bound compute/memory).
- `--covariates` (default: `'[]'`): List of covariate keys to condition on (empty by default).
- `--float_fixedvariance_GMMs` (default: `0.01`): Fixed variance for GMM components.
- `--str_kwargs_GeodesicLen` (default: `"{'str_integfromX0_or_integfromX1':'integfromX0', 'steps_ODEsolver':3, 'mode_compute_geodlen':CalcModeGeodesicLength.ELL2NORM_U.value, 'coef_FMloss':0.0}"`): Settings for geodesic length computation (integration direction, steps, mode, etc.).
- `--moduleV_list_dim_hidden` (default: `"[200, 200, 200]"`): Hidden layer widths for the velocity/flow network.
- `--temperature_gumble_softmax` (default: `0.5`): Gumbel-softmax temperature.
- `--gumbel_softmax_flag_hard` (default: `'False'`): Whether to use hard (argmax-like) gumbel-softmax.
- `--kwargs_Htheta` (default: `{'prob_dropout_Htheta':0.0}`): Extra config for $H_\theta$.
- `--flag_rapidssc_KMeans` (default: `'False'`): Whether to enable rapid SSC via KMeans.
- `--flag_rapidssc_pca` (default: `'False'`): Whether to enable rapid SSC via PCA.

If you want to override any of these defaults, just pass the corresponding flag as in the full example.

## Evaluation

The runner will automatically generate and evaluate new cells, results will both be in the Logs and saved in pickle files in the specified workspace.

