# interface/manage_adata.py
import re
import logging
import random
from datetime import datetime
from typing import List, Optional

import anndata
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler

from data import GeneExpression
from tqdm.autonotebook import tqdm



def _regex_safe_sep(sep: Optional[str]) -> str:
    """Return a regex-safe separator. Default to '+' if missing/empty."""
    return re.escape(sep if sep else "+")


def _make_logger(name: str = "PerturbedGeneExpression") -> logging.Logger:
    """
    Create a simple logger if one doesn't already exist.
    Avoid duplicate handlers across multiple imports.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        logger.propagate = False
    return logger


class CustomisedDLSampler(Sampler):
    """
    Yields batches built from (pert[, covars]) populations.
    Splits any over-large population into fixed-size chunks; optional upsampling to target minibatch size.
    """

    def __init__(
        self,
        adata_train: anndata.AnnData,
        maxsize_split_population: int,
        perturbation_key: str,
        num_repeats_selcondition: int,
        dict_resample_population_incond: dict | None,
        condition_keys: List[str] | None,
        size_permutationbank_customsampler: int | None,
    ):
        self.adata_train = adata_train
        self.maxsize_split_population = int(maxsize_split_population)
        self.perturbation_key = perturbation_key
        self.num_repeats_selcondition = int(num_repeats_selcondition)
        self.dict_resample_population_incond = dict_resample_population_incond
        self.condition_keys = list(condition_keys) if condition_keys else []
        self.has_covariates = len(self.condition_keys) > 0
        self.size_permutationbank_customsampler = size_permutationbank_customsampler

        self.list_bank_indices = None
        if self.size_permutationbank_customsampler:
            self.list_bank_indices = [self._get_list_indices()
                                      for _ in tqdm(range(self.size_permutationbank_customsampler), 'Precompute sampling indces for dataloader')]

        # length is stable across iterations
        any_list = self.list_bank_indices[0] if self.list_bank_indices else self._get_list_indices()
        self.precomputed_len = len(any_list)

    def __len__(self):
        return self.precomputed_len

    def __iter__(self):
        random.seed(datetime.now().timestamp())
        if self.list_bank_indices:
            list_indices = random.choice(self.list_bank_indices)
        else:
            list_indices = self._get_list_indices()

        random.shuffle(list_indices)

        pending = []
        for pop_idx_list in list_indices:
            # Optional upsample
            if self.dict_resample_population_incond and \
               int(self.dict_resample_population_incond.get('targetsize_mini_batch', 0)) > len(pop_idx_list):
                tgt = int(self.dict_resample_population_incond['targetsize_mini_batch'])
                up = random.choices(pop_idx_list, k=tgt)
            else:
                up = list(pop_idx_list)

            pending.append(up)
            if len(pending) == self.num_repeats_selcondition:
                yield [i for chunk in pending for i in chunk]
                pending = []

        if pending:
            yield [i for chunk in pending for i in chunk]

    def _get_list_indices(self):
        ad = self.adata_train

        if not self.has_covariates:
            perts = ad.obs[self.perturbation_key].astype(str).tolist()
            u_perts = list(set(perts))
            populations = [
                np.where(ad.obs[self.perturbation_key].astype(str).values == p)[0].tolist()
                for p in u_perts
            ]
        else:
            keys = [
                "@".join([ad.obs.iloc[i][self.perturbation_key]] +
                         [str(ad.obs.iloc[i][c]) for c in self.condition_keys])
                for i in range(ad.n_obs)
            ]
            u_keys = sorted(set(keys))
            populations = [
                np.where(np.array(keys) == k)[0].tolist()
                for k in u_keys
            ]

        # split large populations into chunks
        chunks = []
        for pop in populations:
            random.shuffle(pop)
            if len(pop) <= self.maxsize_split_population:
                chunks.append(pop)
            else:
                for s in range(0, len(pop), self.maxsize_split_population):
                    sub = pop[s:s + self.maxsize_split_population]
                    if len(sub) < self.maxsize_split_population:
                        sub = sub + random.choices(pop, k=self.maxsize_split_population - len(sub))
                    chunks.append(sub)
        return chunks


class PerturbedGeneExpression:
    """
    Thin wrapper that:
      - normalizes separator (regex-safe for the dataset, literal for string splits elsewhere),
      - makes a DataLoader with the population-aware sampler,
      - keeps covariates optional (None/[] = no covars).
    """

    def __init__(
        self,
        file_path: str,
        perturbation_key: str,
        condition_keys: Optional[List[str]],
        dosage_key: Optional[str],
        flag_rapidssc_pca:bool,
        control: str,
        sep: str = "+",       # literal separator used in strings
        split_key: str = "split",
        num_pca_components: Optional[int] = None,
        genetic_perturbation: bool = False,
        delta: bool = False,
        use_cuda: bool = True,
        dict_resample_population_incond: dict | None = None,
        num_repeats_selcondition: int = 10,
        maxsize_split_population: int = 1000,
        size_permutationbank_customsampler: int | None = 5,
    ):
        # logging (FIX): create a real logger for downstream dataset code
        self.logger = _make_logger("PerturbedGeneExpression")

        self.sep_literal = sep if sep else "+"
        self.sep_regex = _regex_safe_sep(self.sep_literal)

        use_cuda = torch.cuda.is_available() and use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        self.perturbation_key = perturbation_key
        self.condition_keys = list(condition_keys) if condition_keys else []
        self.dosage_key = dosage_key
        self.split_key = split_key
        self.control = control
        self.num_pca_components = num_pca_components
        self.delta = delta

        self.dict_resample_population_incond = dict_resample_population_incond
        self.num_repeats_selcondition = int(num_repeats_selcondition)
        self.maxsize_split_population = int(maxsize_split_population)
        self.size_permutationbank_customsampler = size_permutationbank_customsampler

        # FIX: pass a real logger (not None) to GeneExpression
        self.gene_expression_dataset = GeneExpression(
            logger=self.logger,
            file_path=file_path,
            perturbation_key=perturbation_key,
            condition_keys=self.condition_keys,   # [] if none
            dosage_key=dosage_key,
            flag_rapidssc_pca=flag_rapidssc_pca,
            control=control,
            sep=self.sep_regex,                  # regex-safe for any `re.split` inside dataset
            split_key=split_key,
            genetic_perturbation=genetic_perturbation,
            num_pca_components=num_pca_components,
            delta=delta,
            device=self.device,
        )

        self.data_loader = None

    def __len__(self):
        return len(self.gene_expression_dataset)

    def init_data_loader(self, batch_size: int):
        cond_keys = getattr(self.gene_expression_dataset, "condition_keys", []) or []
        cond_keys_for_sampler = cond_keys if len(cond_keys) > 0 else None
        sampler = CustomisedDLSampler(
            adata_train=self.gene_expression_dataset.adata_train,
            maxsize_split_population=self.maxsize_split_population,
            perturbation_key=self.gene_expression_dataset.perturbation_key,
            dict_resample_population_incond=self.dict_resample_population_incond,
            num_repeats_selcondition=self.num_repeats_selcondition,
            condition_keys=cond_keys_for_sampler,
            size_permutationbank_customsampler=self.size_permutationbank_customsampler,
        )

        self.data_loader = DataLoader(
            self.gene_expression_dataset,
            batch_sampler=sampler,
            collate_fn=self.gene_expression_dataset.custom_collate_fn,
            num_workers = 8
        )
