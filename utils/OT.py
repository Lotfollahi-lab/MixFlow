
"""
Note: Sinkhorn OT plan sampler was implemented in this file, but never was integrated into the rest of code base nor was it used in model training.
"""


import torch
import ot
import time
import pandas as pd
from torchcfm.optimal_transport import OTPlanSampler
import numpy as np


class SinkhornOTSampler:

    def __init__(self, kwargs_OT_sinkhorn:dict):
        # ot.backend("torch")  # as if not needed anymore
        self.kwargs_OT_sinkhorn = kwargs_OT_sinkhorn

    @torch.no_grad()
    def get_map(self, x0:torch.Tensor, x1:torch.Tensor):
        assert isinstance(x0, torch.Tensor)
        assert isinstance(x1, torch.Tensor)
        assert len(x0.shape) == 2
        assert len(x1.shape) == 2
        assert x0.shape[0] == x1.shape[0]
        assert x1.shape[1] == x1.shape[1]

        # create M
        M = ot.dist(x0, x1)  # [N x N]
        M = M / M.max()


        # create a, b
        a = torch.tensor(ot.unif(x0.shape[0]), device=x0.device).float()
        b = torch.tensor(ot.unif(x1.shape[0]), device=x0.device).float()

        # compute the OT plan
        P = ot.sinkhorn(a=a, b=b, M=M, **self.kwargs_OT_sinkhorn)  # [N x N]
        P = P / (torch.sum(P, 1).unsqueeze(1))
        
        # TODO:handle if there are NaN values in P
        assert not torch.any(torch.isnan(P)).tolist()

        return P, self.round_P(P)
    
    @torch.no_grad() 
    def round_P(self, P_):
        P = P_ + 0.0
        assert len(P.shape) == 2
        assert P.shape[0] == P.shape[1]
        N = P.shape[0]
        
        # get an ordering of rows, such that the rows with larger max value are treated first 
        list_idx_row = torch.argsort(
            torch.max(P, 1).values,
            descending=True
        ).tolist()

        


        list_idxcol_toret = []
        for idx_row in list_idx_row:
            list_idxcol_toret.append(
                torch.argmax(P[idx_row, :]).tolist()  # an index
            )
            P[:, list_idxcol_toret[-1]] = -torch.inf
        
        assert set(list_idxcol_toret) == set(range(P.shape[0]))

        return list_idxcol_toret
    

    def speedtest_OT_sampler(self, device, num_runs, N, D, func_gensamples):
        otsampler_exact = OTPlanSampler(method='exact')
        data_df = []
        colnames_df = ['run_IDX', 'method', 'mean pairwise distance', 'generation time']

        for idx_run in range(num_runs):
            x0 = func_gensamples(N, D, device=device)
            x1 = func_gensamples(N, D, device=device)

            # sinkhorn =======
            t_tic = time.time()
            _, list_0_ot_1_sinkhorn = self.get_map(x0, x1)
            runtime_sinkhorn = time.time() - t_tic
            t_tic = time.time()
            data_df.append([
                idx_run,
                'sinkhorn',
                ((x1[list_0_ot_1_sinkhorn, :] - x0) ** 2).sum(1).mean(0).detach().cpu().numpy().tolist(),
                time.time() - t_tic
            ])

            # exact ========= 
            t_tic = time.time()
            ot_map_exact = otsampler_exact.get_map(x0, x1)
            list_0_ot_1_exact = [np.where(ot_map_exact[i, :] > 0)[0].tolist()[0] for i in range(ot_map_exact.shape[0])]
            data_df.append([
                idx_run,
                'exact',
                ((x1[list_0_ot_1_exact, :] - x0) ** 2).sum(1).mean(0).detach().cpu().numpy().tolist(),
                time.time() - t_tic
            ])

            # random pairing
            t_tic = time.time()
            ot_map_random_pairing = np.random.permutation(x0.shape[0]).tolist()
            data_df.append([
                idx_run,
                'random',
                ((x1[ot_map_random_pairing, :] - x0) ** 2).sum(1).mean(0).detach().cpu().numpy().tolist(),
                time.time() - t_tic
            ])
        
        
        df_toret = pd.DataFrame(
            data=data_df,
            columns=colnames_df
        )

        return df_toret

            



        
        
            

        

        
        