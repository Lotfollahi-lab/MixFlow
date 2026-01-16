
import os, sys

import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from diffusers import UNet1DModel
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv, global_mean_pool
from torchcfm.optimal_transport import OTPlanSampler
from torchdiffeq import odeint
from torch.special import gammaln
from tqdm import tqdm
from tqdm.auto import tqdm

from FlowMatching.models.FM import (
    compute_conditional_vector_field,
    sample_conditional_pt,
)

class MLP_CondRepeatedlyAdded(nn.Module):
    """
    And MLP where "some" dims of input (i.e. input[:, idx_repeatadd:]) are repeatedly added in intermediate layers.
    """
    def __init__(self, input_dim, hidden_dims, output_dim, idx_repeatadd, activation=nn.ELU):
        super(MLP_CondRepeatedlyAdded, self).__init__()

        self.idx_repeatadd = idx_repeatadd

        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]  # (u + input_dim - idx_repeatadd)
        
        assert len(hidden_dims) > 0  # because the below for loop assumes this.
        for i in range(len(dims) - 1):
            if i == 0:
                layers.append(
                    nn.Sequential(
                        nn.Linear(dims[i], dims[i + 1]),
                        activation()
                    )
                )
            elif i < len(dims) - 2:
                layers.append(
                    nn.Sequential(
                        nn.Linear(dims[i] + input_dim - idx_repeatadd, dims[i + 1]),
                        activation()
                    )
                )
            else:
                layers.append(
                    nn.Linear(dims[i] + input_dim - idx_repeatadd, dims[i + 1])
                )
            

        self.list_layers = nn.ModuleList(layers)
        #self.net = nn.Sequential(*layers)

    def forward(self, x):
        x_part_repeat = x[:, self.idx_repeatadd:] 
        x = self.list_layers[0](x)  # op:0

        for idx_l in range(1, len(self.list_layers)):
            x = self.list_layers[idx_l](
                torch.cat(
                    [x, x_part_repeat],
                    1
                )
            )
        
        return x
        

def init_weights(m):
    '''
    Grabbed from: https://stackoverflow.com/questions/49433936/how-do-i-initialize-weights-in-pytorch
    '''
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform(m.weight)
        m.bias.data.fill_(0.01)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ELU):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GeneExpressionNet(nn.Module):  # i.e. the flow module, v(x,t,conditions)
    def __init__(
        self,
        input_dim,  # dim of x, i.e. num selected PCA components.
        dim_embedding,  # i.e. dimension of perturbation embeddings after they go through dim-reduction and averaging.
        num_conditions,
        dict_covarname_to_dimembedding,  # a dict that maps each covarname to the embedding dim for that covariate.
        hidden_dims,
        activation=nn.ELU
    ):
        super().__init__()
        self.dim_embedding = dim_embedding
        self.num_conditions = num_conditions
        self.input_dim = input_dim

        # self.net = MLP(
        #     input_dim + dim_embedding + 10 * num_conditions + 1,  # note: +1 for 'time'.
        #     hidden_dims,
        #     input_dim,
        #     activation=activation,
        # )



        self.net = MLP_CondRepeatedlyAdded(
            input_dim=input_dim + dim_embedding + sum([dict_covarname_to_dimembedding[covarname] for covarname in dict_covarname_to_dimembedding.keys()]) + 1,  # note: +1 for 'time'.,
            hidden_dims=hidden_dims,
            output_dim=input_dim, 
            idx_repeatadd=input_dim + 1  # aknote: +1 for not repeatedly feeding the time (intuitively, V(.,.,.) has to  smootly vary with time, i.e. not very dependant on time)
        )
        # print `self.net`
        print(">>>>>>>>>>>>>>>>>>")
        for idx_layer, layer in enumerate(self.net.list_layers):
            print("Layer {}: ".format(idx_layer))
            print(layer)
            print("\n\n")
        print("       >>>>>>>>>>>>>>>")
        

    def forward(self, x, t, condition_and_perturbation):

        # inspect shapes in the generation phase, i.e. when `torchdiffeq.odeint` makes calls to this function.
        #   This forward expands `t` ==> works fine in both cases.
        # print("x.shape = {}".format(x.shape))
        # print("t.shape = {}, t={}".format(t.shape, t))
        # print("condition_and_perturbation.shape = {}".format(condition_and_perturbation.shape))
        # print("\n\n\n\n")
        '''
        x.shape = torch.Size([105000, 10])
        t.shape = torch.Size([]), t=0.021857867017388344
        condition_and_perturbation.shape = torch.Size([105000, 50])
        '''

        # Norman: x of shape [N x 50]
        # print("t.shape = {}".format(t.shape)): [N]
        t = t.unsqueeze(0).view(-1, 1).expand(x.shape[0], 1)
        # print(">>>>>> t.shape = {}".format(t.shape)) [N x 1]
        return self.net(torch.cat((x, t, condition_and_perturbation), -1))
        # TODO:aknote: for some reason I've mostly seen t appended at the end, but it shouldn't really matter.

class GeneExpressionGNN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GeneExpressionGNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)
        return x


class ModuleSimplyScale(nn.Module):
    def __init__(self, float_scale:float):
        super(ModuleSimplyScale, self).__init__()

        assert isinstance(float_scale, float)
        self.float_scale = float_scale
    
    def forward(self, x):
        return self.float_scale * x


class ModuleEll2Normalise(nn.Module):
    def __init__(self):
        super(ModuleEll2Normalise, self).__init__()
    
    def forward(self, x):
        rowwise_norm = torch.linalg.norm(x, dim=1).unsqueeze(-1)  # [N x 1]
        return x / rowwise_norm

class ModuleHtheta(nn.Module):
    """
    ddd
    """

    def __init__(self, dim_in:int, dim_out:int, prob_dropout_Htheta:float, ten_precopmuted_cluster_modes:torch.Tensor):
        super(ModuleHtheta, self).__init__()

        # check args
        assert isinstance(ten_precopmuted_cluster_modes, torch.Tensor)
        assert len(ten_precopmuted_cluster_modes.size()) == 1  # i.e. it has to be flatten-ed to [num_modes * D]
        assert isinstance(dim_in, int)
        assert isinstance(dim_out, int)
        
        # make internals
        self.H_theta = nn.Sequential(
            nn.Linear(dim_in, 128),
            nn.Dropout(p=prob_dropout_Htheta),  # TODO:aknote:tune p
            nn.ReLU(),
            nn.Linear(128, dim_out)
        )
        
        self.param_precopmuted_cluster_modes = torch.nn.Parameter(
            torch.tensor(ten_precopmuted_cluster_modes.detach().cpu().numpy()).detach(),
            requires_grad=False
        )
    
    def forward(self, x):
        return self.H_theta(x)





# class HthetaAdditive(nn.Module):
#     """
#     The H_theta module, additive to the cluster centres, with additive amounts bouneded.
#     If the additive amount is 0.0, the module outputs the cluster centres, i.e., no parameter training.
#     """

#     def __init__(self, dim_in:int, dim_out:int, prob_dropout_Htheta:float, upper_bound_additive_amount:float, ten_precopmuted_cluster_modes:torch.Tensor, flag_is_additive:bool, type_ending_module):
#         super(HthetaAdditive, self).__init__()

#         # check args
#         assert isinstance(ten_precopmuted_cluster_modes, torch.Tensor)
#         assert isinstance(upper_bound_additive_amount, float)
#         assert upper_bound_additive_amount >= 0.0
#         assert len(ten_precopmuted_cluster_modes.size()) == 1  # i.e. it has to be flatten-ed to [num_modes * D]
#         assert isinstance(dim_in, int)
#         assert isinstance(dim_out, int)
#         assert isinstance(flag_is_additive, bool)

#         assert type_ending_module in [ModuleEll2Normalise, nn.Tanh]

#         # make internals
#         self.upper_bound_additive_amount = upper_bound_additive_amount
#         if upper_bound_additive_amount > 0:
#             self.H_theta = nn.Sequential(
#                 nn.Linear(dim_in, 128),
#                 nn.Dropout(p=prob_dropout_Htheta),  # TODO:aknote:tune p
#                 nn.ReLU(),
#                 nn.Linear(128, dim_out),
#                 type_ending_module(),  # TODO:revert, nn.Tanh(),  # beacuse of the initial normalisation, the embeddings are in [-1.0, +1.0]
#                 ModuleSimplyScale(float_scale=upper_bound_additive_amount)
#             )  # TODO: is dropout needed here? nn.Dropout(p=self.prob_dropout_conditionencoder),
#         else:
#             assert upper_bound_additive_amount == 0.0
#             self.H_theta = None

#         self.param_precopmuted_cluster_modes = torch.nn.Parameter(
#             torch.tensor(ten_precopmuted_cluster_modes.detach().cpu().numpy()).detach(),
#             requires_grad=False
#         )
#         self.flag_is_additive = flag_is_additive

    
#     def forward(self, x):
#         if self.H_theta is None:
#             assert self.upper_bound_additive_amount == 0.0
#             return torch.stack(x.shape[0]*[self.param_precopmuted_cluster_modes], 0)
#         else:
#             if self.flag_is_additive:
#                 return torch.stack(x.shape[0]*[self.param_precopmuted_cluster_modes], 0) + self.H_theta(x)
#             else:
#                 return self.H_theta(x)
            
            


class GeneExpressionFlowModel(nn.Module):
    def __init__(
        self,
        vspace_model,
        dim_perturbation,
        conditions,
        prob_dropout_conditionencoder:float,
        prob_dropout_fc1fc2:float,
        kwargs_OT_sampler:dict,
        gumbel_softmax_flag_hard:bool,
        ten_precopmuted_cluster_modes:torch.Tensor,
        kwargs_Htheta:dict,
        pretrained_embeddings=None,
        dim_gene_hot_encoded=None,
        conditional_noise=False,
        num_modes=32,
        modes=None,
        temperature=1.0,
        MFM=False
    ):

        super(GeneExpressionFlowModel, self).__init__()

        self.vspace = vspace_model
        self.dim_embedding = vspace_model.dim_embedding
        self.dim_gene_hot_encoded = dim_gene_hot_encoded
        if self.dim_gene_hot_encoded is not None:
            #self.genetic_perturbation = True
            self.genetic_perturbation = False
        else:
            self.genetic_perturbation = False
        self.conditional_noise = conditional_noise
        self.temperature = temperature
        self.ot_sampler = OTPlanSampler(**kwargs_OT_sampler) #  method="exact")
        self.MFM = MFM
        self.num_modes = num_modes

        self.embedding_layers = torch.nn.ModuleDict()
        self.num_conditions = len(conditions)
        self.gumbel_softmax_flag_hard = gumbel_softmax_flag_hard

        

        # conditions: same as `data.gene_expression_dataset.condition_dict`, a dictionary like {'covarname_1':['val11', 'val12'], 'covarname_2':['val21', 'val22']} 
        # pretrained_embeddings: passed as `data.gene_expression_dataset.pretrained_embeddings`, a dictionary like {'covarname_1':{'val11':list_embedding, 'val12':list_embedding}, 'covarname_2':{'val21':list_embds, 'val22':list_embds}}
        for condition, categories in conditions.items():  
            # the purpose of this for loop is to create `self.embedding_layers` that maps each covariate name to its corresponding embedding module. I.e. `self.embedding_layers` maps each covarname to an encoder to embs of shape [num_covariate_values x e.g10].
            # raise Exception("Review the below code, them uncomment this raise Exepction line.")

            num_categories = len(categories)

            if condition in pretrained_embeddings.keys():  # pretrained_embeddings and condition in pretrained_embeddings:
                # Use pre-trained embeddings for this condition
                
                pretrained_emb = torch.tensor(
                    pretrained_embeddings[condition],
                    dtype=torch.float32
                )
                exec(
                    f"embedding_layer_{condition} = torch.nn.Embedding.from_pretrained(pretrained_emb, freeze=True)"
                )

            else:
                # Dynamically initialize a trainable embedding layer
                exec(
                    f"embedding_layer_{condition} = torch.nn.Embedding(num_categories, 10)"
                )

            exec(f"self.embedding_layers[condition] = embedding_layer_{condition}")
        
        self.prob_dropout_conditionencoder = prob_dropout_conditionencoder

        if conditional_noise:
            if modes is not None:  # i.e. when conditioning is done with PertFlow 
                self.variances = modes['var']
            else:  # i.e. when the base dist is the usual N(0,1)
                self.variances = 1.0
            
            self.prob_dropout_fc1fc2 = prob_dropout_fc1fc2
            self.module_fc1fc2 = nn.Sequential(
                nn.Linear(
                    sum([self.embedding_layers[covarname].embedding_dim for covarname in self.embedding_layers.keys()]) + self.dim_embedding,
                    128
                ),
                nn.Dropout(p=self.prob_dropout_fc1fc2),
                nn.ReLU(),
                nn.Linear(128, self.num_modes)
            ) 
            
            """
            self.fc1 = nn.Linear(
                sum([self.embedding_layers[covarname].embedding_dim for covarname in self.embedding_layers.keys()]) + self.dim_embedding,
                128
            )  
            # TODO:aknote:BUGBUGBUGBUGBUGBUG: the above line presumes that the embeddings for each covariate equals 10.
            #   but..... if the embedding is fixed/frozen in adata.uns, its dimension may not be equal to 10.
            # Recall: `self.dim_embedding` is the output dim of the covariate encoder, i.e. the space to which different covars are encoded to and are averaged.
            self.fc2 = nn.Linear(128, self.num_modes)
            """


            # for the GMM means ===
            self.module_H_theta = ModuleHtheta(
                dim_in=10 * self.num_conditions + self.dim_embedding,
                dim_out=self.num_modes*self.vspace.input_dim,
                ten_precopmuted_cluster_modes=ten_precopmuted_cluster_modes.detach().flatten(),
                **kwargs_Htheta
            )
            



            if self.MFM:
                self.gnn = GeneExpressionGNN(
                    input_dim=self.dim_embedding,
                    hidden_dim=512,
                    output_dim=self.dim_embedding,
                )
                self.linear = nn.Linear(self.dim_embedding * 2, self.dim_embedding)

        self.lower_dim_embedding_perturbation = nn.Sequential(
            nn.Linear(dim_perturbation, 128),
            nn.Dropout(p=self.prob_dropout_conditionencoder),  # TODO:aknote:tune p
            nn.ReLU(),
            nn.Linear(128, self.dim_embedding),
        )  
        # TODO:aknote:IMP: since dim_perturbation is, e.g., 3K on norman, then the first layer has 3K x 128 ~ 300K params.
        #    It's usually more than number of cells in a typical dataset.
        # TODO:aknote:IMP what if Dropout is added here ???  

        self.xavierinit_condition_encoder()
        print(" >>>>>>>>> Xavier init was called for the condition encoder.")

        if self.genetic_perturbation:
            self.lower_dim_embedding_hot_encoded_gene = nn.Sequential(
                nn.Linear(self.dim_gene_hot_encoded, 128),
                nn.ReLU(),
                nn.Linear(128, self.dim_embedding),
            )
    

    def xavierinit_condition_encoder(self):
        self.lower_dim_embedding_perturbation.apply(init_weights)
        
        if self.conditional_noise:
            self.module_H_theta.apply(init_weights)
            self.module_fc1fc2.apply(init_weights)

    def getparams_base_dist_encoder(self):
        """
        To be used by the separate optimiser for the base dist, including
            - Basedist means
            - Basedist variances (in case `requires_grad` is set to True later on)
            - self.module_fc1fc2
            - self.gnn and self.linear (if self.MFM)
            - self.lower_dim_embedding_perturbation
            - self.lower_dim_embedding_hot_encoded_gene if self.genetic_perturbation is set to True
        """
        list_toret = []
        list_toret = list_toret + list(self.module_H_theta.parameters())  # [self.means, self.variances]
        list_toret = list_toret + list(self.module_fc1fc2.parameters())  # list(self.fc1.parameters()) + list(self.fc2.parameters())

        if self.MFM:
            list_toret = list_toret + list(self.gnn.parameters()) + list(self.linear.parameters())
        
        list_toret = list_toret +  list(self.lower_dim_embedding_perturbation.parameters())
        if self.genetic_perturbation:
            list_toret = list_toret + list(self.lower_dim_embedding_hot_encoded_gene.parameters())
        
        return list_toret

        

    def masked_mean_pooling(self, perturbations, mask):
            # print("perturbations.shape = {}".format(
            #     perturbations.shape
            # ))
            # print("mask.shape = {}".format(
            #     mask.shape
            # ))
            # perturbations.shape = torch.Size([N, 2, 3072])
            # mask.shape = torch.Size([N, 2])

            perturbations = self.lower_dim_embedding_perturbation(perturbations)  # [N x 2 x 10]
            # `self.lower_dim_embedding_perturbation` is a [Linear, ReLU, Linear(..., self.dim_embedding)] module.
            return (perturbations * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(
                dim=1, keepdim=True
            )  # [N x 10]


    # @torch.no_grad()
    # def _tenperturbations_to_listgroupidx(self, perturbations):
    #     """
    #     Given `perturbations` of shape, e.g., [N x 10] returns a list containing the grouping of the N instances based on their perturbation population.
    #     """
    #     pass


    def forward_trainsep_Htheta(
        self,
        x,
        perturbations,
        mask,
        conditions,                       # may be None or empty TensorDict/dict
        dosage,
        gene_hot_encoded,
        flag_update_basedist_aswell,
        strID_perturbations,              # may be None; we’ll fallback to global OT
        ten_precopmuted_cluster_modes:torch.Tensor,  # [num_modes x D]
        stddiv_noise_fit2_cluster_modes:float, 
    ):
        assert isinstance(ten_precopmuted_cluster_modes, torch.Tensor)
        assert len(list(ten_precopmuted_cluster_modes.shape)) == 2

        # ---- 1) Pool perturbations to a single embedding per sample ----
        perturbations = self.masked_mean_pooling(perturbations, mask)  # [N, dim_emb]

        if self.genetic_perturbation:
            raise Exception("Review the below part, then uncomment this line.")
            gene_hot_encoded = self.lower_dim_embedding_hot_encoded_gene(gene_hot_encoded)
            perturbations = perturbations + gene_hot_encoded

        # ---- 2) Normalise `conditions` and build a global covariate embedding if any ----
        # conditions can be: None, {}, or a TensorDict/dict with keys -> index tensors [N]
        cov_keys = []
        if conditions is not None:
            try:
                cov_keys = list(conditions.keys())
            except Exception:
                cov_keys = []
        has_covs = len(cov_keys) > 0

        global_condition = None
        if has_covs:
            # Embed each covariate index vector using its embedding layer.
            cond_embs = []
            for c in cov_keys:
                idx = conditions[c]
                if idx.dtype != torch.long:
                    idx = idx.long()
                cond_embs.append(self.embedding_layers[c](idx))  # [N, emb_dim_c]
            if len(cond_embs) > 0:
                global_condition = torch.cat(cond_embs, dim=1)   # [N, sum_c emb_dim_c]

        # Final conditioner = [covariate_embs (if any)] + [perturbation_emb]
        if global_condition is None:
            condition_and_perturbation = perturbations
        else:
            condition_and_perturbation = torch.cat((global_condition, perturbations), dim=1)

        # Modulate by dosage
        condition_and_perturbation = condition_and_perturbation * dosage[:, None]

        predicted_means = self.module_H_theta(condition_and_perturbation)  # [N x num_modes * D]
        predicted_means = predicted_means.reshape(
            perturbations.shape[0],
            self.num_modes,
            -1
        )  # [N x num_modes x D]

        loss_toret = \
            ten_precopmuted_cluster_modes.unsqueeze(0) +\
            stddiv_noise_fit2_cluster_modes*torch.randn_like(ten_precopmuted_cluster_modes).unsqueeze(0) -\
            predicted_means  # [N x num_modes x D]

        loss_toret = torch.mean(loss_toret ** 2)
        return loss_toret

    
    def forward(
        self,
        x,
        perturbations,
        mask,
        conditions,                       # may be None or empty TensorDict/dict
        dosage,
        gene_hot_encoded,
        flag_update_basedist_aswell,
        strID_perturbations,              # may be None; we’ll fallback to global OT
    ):
        # ---- 1) Pool perturbations to a single embedding per sample ----
        perturbations = self.masked_mean_pooling(perturbations, mask)  # [N, dim_emb]

        if self.genetic_perturbation:
            raise Exception("Review the below part, then uncomment this line.")
            gene_hot_encoded = self.lower_dim_embedding_hot_encoded_gene(gene_hot_encoded)
            perturbations = perturbations + gene_hot_encoded

        # ---- 2) Normalise `conditions` and build a global covariate embedding if any ----
        # conditions can be: None, {}, or a TensorDict/dict with keys -> index tensors [N]
        cov_keys = []
        if conditions is not None:
            try:
                cov_keys = list(conditions.keys())
            except Exception:
                cov_keys = []
        has_covs = len(cov_keys) > 0

        global_condition = None
        if has_covs:
            # Embed each covariate index vector using its embedding layer.
            cond_embs = []
            for c in cov_keys:
                idx = conditions[c]
                if idx.dtype != torch.long:
                    idx = idx.long()
                cond_embs.append(self.embedding_layers[c](idx))  # [N, emb_dim_c]
            if len(cond_embs) > 0:
                global_condition = torch.cat(cond_embs, dim=1)   # [N, sum_c emb_dim_c]

        # Final conditioner = [covariate_embs (if any)] + [perturbation_emb]
        if global_condition is None:
            condition_and_perturbation = perturbations
        else:
            condition_and_perturbation = torch.cat((global_condition, perturbations), dim=1)

        # Modulate by dosage
        condition_and_perturbation = condition_and_perturbation * dosage[:, None]

        # ---- 3) Conditional base noise (if enabled) ----
        if self.conditional_noise:
            logits = self.module_fc1fc2(condition_and_perturbation)  # self.fc2(F.selu(self.fc1(condition_and_perturbation)))
            if not flag_update_basedist_aswell:
                logits = logits.detach()

            self.gumbel_softmax_output = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=self.gumbel_softmax_flag_hard
            )
            if not flag_update_basedist_aswell:
                self.gumbel_softmax_output = self.gumbel_softmax_output.detach()
            
            predicted_means = self.module_H_theta(condition_and_perturbation)  # [N x num_modes * D]
            predicted_means = predicted_means.reshape(
                logits.shape[0],
                self.num_modes,
                -1
            )  # [N x num_modes x D]

            mean = torch.sum(
                self.gumbel_softmax_output.unsqueeze(-1) * predicted_means,  # [N x num_modes x D]
                1
            )  # [N x D]

            variance = self.variances  # float

            if not flag_update_basedist_aswell:
                mean = mean.detach()

            self.samples = mean + float(np.sqrt(variance)) * torch.randn_like(mean)
            x0 = self.samples
        else:
            x0 = torch.randn_like(x)

        if not flag_update_basedist_aswell:
            x0 = x0.detach()

        # Optional MFM branch
        if self.MFM:
            condition_and_perturbation = self.linear(
                torch.cat(
                    (condition_and_perturbation,
                    self.embed_source(x0, cond=condition_and_perturbation, k=self.k)),
                    dim=1,
                )
            )

        # ---- 4) Mini-batch OT pairing ----
        # If we have a valid list of per-sample IDs, we do per-group OT (pert +/- covars).
        # Otherwise, fallback to a single global OT over the whole batch.
        use_group_ot = isinstance(strID_perturbations, list) and (len(strID_perturbations) == x.shape[0])

        if use_group_ot:
            # Build covariate index lists only if we actually have covariates
            dict_covarname_to_listcovarindices = {}
            if has_covs:
                for c in cov_keys:
                    dict_covarname_to_listcovarindices[c] = conditions[c].detach().cpu().numpy().tolist()

            # Build a per-sample string key
            list_strID_PertAndCovarindices = []
            for n in range(x.shape[0]):
                if has_covs:
                    list_strID_PertAndCovarindices.append(
                        "@".join(
                            [str(strID_perturbations[n])] +
                            [str(dict_covarname_to_listcovarindices[c][n]) for c in cov_keys]
                        )
                    )
                else:
                    # No covariates: key is just the perturbation ID
                    list_strID_PertAndCovarindices.append(str(strID_perturbations[n]))

            new_x0, new_x, new_condition_and_perturbation = [], [], []
            orig_x_shape0 = x.shape[0]

            for key in set(list_strID_PertAndCovarindices):
                with torch.no_grad():
                    mask_sel = (np.array(list_strID_PertAndCovarindices) == key)
                    # if np.sum(mask_sel) < 100:
                    #     raise Exception(
                    #         f"In the training mini-batch there are {np.sum(mask_sel)} < 100 cells with condition {key}. "
                    #         f"Please set `targetsize_mini_batch` to at least 100 and try again."
                    #     )
                    ot_map = self.ot_sampler.get_map(x0[mask_sel, :], x[mask_sel, :])
                    list_0_ot_1 = [np.where(ot_map[i, :] > 0)[0].tolist()[0] for i in range(ot_map.shape[0])]

                new_x0.append(x0[mask_sel, :])
                new_x.append(x[mask_sel, :][list_0_ot_1, :])
                new_condition_and_perturbation.append(condition_and_perturbation[mask_sel, :][list_0_ot_1, :])

            x0 = torch.concat(new_x0, 0)
            x  = torch.concat(new_x, 0)
            condition_and_perturbation = torch.concat(new_condition_and_perturbation, 0)

            assert x0.shape[0] == orig_x_shape0
            assert x.shape[0] == orig_x_shape0
            assert condition_and_perturbation.shape[0] == orig_x_shape0

        else:
            # Fallback: single global OT pairing
            raise Exception("Per-condition mini-batch pairing failed.")
            with torch.no_grad():
                ot_map = self.ot_sampler.get_map(x0, x)
                list_0_ot_1 = [np.where(ot_map[i, :] > 0)[0].tolist()[0] for i in range(ot_map.shape[0])]
            x  = x[list_0_ot_1, :]
            condition_and_perturbation = condition_and_perturbation[list_0_ot_1, :]

        # ---- 5) Sample along conditional path and compute vector field ----
        t  = torch.rand(x0.shape[0]).type_as(x0)
        xt = sample_conditional_pt(x0, x, t, sigma=0.00)   # x1*t + (1-t)*x0
        ut = compute_conditional_vector_field(x0, x)       # x - x0

        vt = self.vspace(xt, t, condition_and_perturbation)
        return ut, vt, x0, condition_and_perturbation

    def sample_noise_from_gmm(self, class_list):  # `class_list` is [N x 10], i.e. the encoding of CombPert from which basetozi vazns are computed.
        if not self.conditional_noise:
            sys.exit("Conditional noise is not enabled.")
        with torch.no_grad():
            logits = self.module_fc1fc2(class_list)  # self.fc2(F.selu(self.fc1(class_list)))  # [N x num_modes]
            gumbel_softmax_output = F.gumbel_softmax(
                logits,
                tau=self.temperature,
                hard=True  # TODO:aknote:IMP during the generation phase no grad is needed --> it can be hard.
            )

            predicted_means = self.module_H_theta(class_list)  # [N x num_modes * D]
            predicted_means = predicted_means.reshape(
                logits.shape[0],
                self.num_modes,
                -1
            )  # [N x num_modes x D]
            mean = torch.sum(
                gumbel_softmax_output.unsqueeze(-1) * predicted_means,  # [N x num_modes x D]
                1
            )  # [N x D]

            variance = self.variances  # float 
            samples = mean + float(np.sqrt(variance)) * torch.randn_like(mean)
            
            x0 = samples
        
        Htheta_output = predicted_means
        return x0, mean, variance, Htheta_output

    def embed_source(self, source_samples, cond=None, k=5):
        b, g = source_samples.shape
        data_list = []

        for i in range(b):
            x = source_samples[i].t()
            edge_index = torch_geometric.nn.pool.knn_graph(x.cpu(), k=k)
            cond_i = cond[i].view(1, -1).expand(x.shape[0], -1)
            data_list.append(
                torch_geometric.data.Data(
                    x=torch.cat((x.cuda(), cond_i), dim=1), edge_index=edge_index.cuda()
                )
            )
        batch_data = torch_geometric.data.Batch.from_data_list(data_list)
        z = batch_data.x
        edge_index = batch_data.edge_index
        z = self.gnn(z, edge_index, batch_data.batch)
        z = z.view(b, cond.shape[1])
        return z
