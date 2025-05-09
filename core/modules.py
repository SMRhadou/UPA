
from core.gnn import PrimalGNN, GNN
from utils import calc_rates

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np

def lagrangian_fn(rates, mu, r_min, p=None, metric='rates', constrained_subnetwork=None):
    assert rates.shape == mu.shape, "Rates and mu must have the same shape"
    if rates.shape != r_min.shape:
        r_min = r_min.view(rates.shape[0], -1)

    sum_rates = torch.sum(rates, dim=1)
    if p is not None:
        sum_power = torch.sum(p, dim=1)

    if constrained_subnetwork is not None:
        rates = rates[:, :int(np.floor(constrained_subnetwork*rates.shape[1]))]
        r_min = r_min[:, :int(np.floor(constrained_subnetwork*r_min.shape[1]))]
        mu = mu[:, :int(np.floor(constrained_subnetwork*mu.shape[1]))]
        if p is not None:
            p = p[:, :int(np.floor(constrained_subnetwork*p.shape[1]))]
    
    if metric == 'rates':
        return -1 * sum_rates + torch.sum((mu * (r_min - rates)), dim=1)    # Posed as a minimization problem
    elif metric == 'power':
        assert p is not None
        return sum_power + torch.sum((mu * (r_min - rates)), dim=1)

class PrimalModel(nn.Module):
    def __init__(self, args, device, unrolled=False):
        super(PrimalModel, self).__init__()
        self.args = args
        self.device = device
        self.normalized_mu = args.mu_max if args.normalize_mu else 1
        self.unrolled = unrolled
        self.primal_num_blocks = args.primal_num_blocks if hasattr(args, 'primal_num_blocks') else args.num_blocks

        self.num_graphs = args.batch_size * args.num_samplers
        if unrolled:
            self.num_features_list = [3] + [args.primal_hidden_size] * args.primal_num_sublayers
        else:
            self.num_features_list = [2] + [args.primal_hidden_size] * args.primal_num_sublayers
        if not self.unrolled:
            if args.architecture == 'GNN':
                self.model = GNN(self.num_features_list, args.P_max).to(device)
            elif args.architecture == 'PrimalGNN':
                norm_layer = getattr(args, 'primal_norm_layer', 'batch')  # Default to 0.0 if not defined
                self.model = PrimalGNN(num_features_list=self.num_features_list, 
                            P_max=args.P_max, k_hops = args.primal_k_hops, norm_layer = norm_layer, dropout_rate=args.dropout_rate,
                            conv_layer_normalize = args.conv_layer_normalize
                            ).to(device)
        else:
            self.blocks = nn.ModuleList()
            for _ in range(self.primal_num_blocks):
                if args.architecture == 'GNN':
                    self.blocks.append(GNN(self.num_features_list, args.P_max).to(device))
                elif args.architecture == 'PrimalGNN':
                    norm_layer = getattr(args, 'primal_norm_layer', 'batch')
                    self.blocks.append(PrimalGNN(num_features_list=self.num_features_list,
                        P_max=args.P_max, k_hops = args.primal_k_hops, norm_layer = norm_layer, dropout_rate=args.dropout_rate,
                        conv_layer_normalize = args.conv_layer_normalize
                        ).to(device))
            
    def sub_forward(self, block_id, input, edge_index_l, edge_weight_l, transmitters_index):
        div_p = self.blocks[block_id](input, edge_index_l, edge_weight_l, transmitters_index, activation=None)
        return div_p

    def forward(self, mu, edge_index_l, edge_weight_l, transmitters_index):
        if not hasattr(self, 'cons_lvl'):
            m = mu.view(-1, self.args.n).shape[0]
            self.cons_lvl = torch.cat([self.args.r_min * torch.ones(m, int(np.floor(self.args.constrained_subnetwork*self.args.n))).to(self.device), 
                                    torch.zeros(m, int(np.ceil((1-self.args.constrained_subnetwork)*self.args.n))).to(self.device)], dim=1).view(-1, 1)
        if not self.unrolled:
            input = torch.cat((mu/self.normalized_mu, self.cons_lvl), dim=1)  # mu is normalized
            p = self.model(input, edge_index_l, edge_weight_l, transmitters_index, activation='sigmoid')
            # if getattr(self.args, 'crop_p', 0) > 0:
            #     p = torch.clamp(p, min=1e-5) # to avoid log(0) in calc_rates
            return p
        else:
            p = torch.zeros_like(mu)
            outputs_list = [p]
            for block_id in range(self.args.primal_num_blocks):
                x = torch.cat((p, mu/self.normalized_mu, self.cons_lvl), dim=1)
                p = p + self.sub_forward(block_id, x, edge_index_l, edge_weight_l, transmitters_index)
                p = self.args.P_max * torch.sigmoid(p)
                # p = torch.clamp(p, min=1e-5)
                outputs_list.append(p)
            return outputs_list
    
    def loss(self, rates, mu, p=None, constraint_eps=None, metric='rates', constrained_subnetwork=None):
        if mu.shape[0] != self.num_graphs * self.args.n:
            num_graphs = mu.shape[0] // self.args.n
        else:
            num_graphs = self.num_graphs
        resilient_weight_deacay = getattr(self.args, 'dual_resilient_decay', 0.0)

        if not isinstance(rates, list):
            rates, mu = rates.view(num_graphs, self.args.n), mu.view(num_graphs, self.args.n)
            p = p.view(num_graphs, self.args.n) if p is not None and not isinstance(p, list) else None
            L = lagrangian_fn(rates, mu, self.cons_lvl, p=p, metric=metric, constrained_subnetwork=constrained_subnetwork).mean()
            if resilient_weight_deacay > 0:
                if constrained_subnetwork is not None:
                    mu = mu[1:, :int(np.floor(constrained_subnetwork*self.args.n))] 
                L = L - (mu.norm(p=1, dim=1)**2/(2*resilient_weight_deacay)).mean()
            constrained_loss = None
        else:
            assert constraint_eps is not None and p is not None
            assert isinstance(p, list) and len(p) == len(rates)
            lagrangian_list = []
            mu = mu.view(num_graphs, self.args.n)
            for r_l, p_l in zip(rates, p):
                r_l, p_l = r_l.view(num_graphs, self.args.n), p_l.view(num_graphs, self.args.n)
                lagrangian_list.append(lagrangian_fn(r_l, mu, self.cons_lvl, p=p_l, metric=metric, constrained_subnetwork=constrained_subnetwork))
            constrained_loss = self.descending_constraints(torch.stack(lagrangian_list), constraint_eps)
            L = lagrangian_list[-1].mean()
            if resilient_weight_deacay > 0:
                if constrained_subnetwork is not None:
                    mu = mu[:, :int(np.floor(constrained_subnetwork*self.args.n))]
                L = L - (mu.norm(p=1, dim=1)**2/(2*resilient_weight_deacay)).mean()
            
        return L, constrained_loss if constrained_loss is not None else torch.zeros(self.primal_num_blocks).to(self.device)
        


    def descending_constraints(self, L_list, constraint_eps):
        return (L_list[1:] - (1-constraint_eps) * L_list[:-1]).mean(-1)
    


    def sanity_check(self, data, mu_test, noise_var, ss_param):
        data = data.to(self.device)
        y, edge_index_l, edge_weight_l, _, \
        _, _, a_l, transmitters_index, num_graphs = \
            data.y, data.edge_index_l, data.edge_weight_l, data.edge_index, data.edge_weight, \
            data.weighted_adjacency, data.weighted_adjacency_l, \
            data.transmitters_index, data.num_graphs
    
        p_test = self(mu_test.detach(), edge_index_l, edge_weight_l, transmitters_index)
        gamma_test = torch.ones(num_graphs * self.args.n, 1).to(self.device) # user selection decisions are always the same (single Rx per Tx)
        if self.unrolled:
            p_test = p_test[-1]
        rates_test = calc_rates(p_test, gamma_test, a_l[:, :, :], noise_var, ss_param)
        rates_test, mu_test = rates_test.view(num_graphs, self.args.n), mu_test.view(num_graphs, self.args.n)
        p_test = p_test.view(num_graphs, self.args.n)
        L = lagrangian_fn(rates_test, mu_test, self.cons_lvl, p=p_test, metric='rates', constrained_subnetwork=None)

        return L.detach().cpu().numpy()






class DualModel(nn.Module):
    def __init__(self, args, device=None, eval_mode='unrolling'):
        super(DualModel, self).__init__()
        self.device = device
        self.eval_mode = eval_mode
        self.num_features_list = [3] + [args.dual_hidden_size] * args.dual_num_sublayers
        self.n = args.n
        self.P_max = args.P_max
        self.constrained_subnetwork = args.constrained_subnetwork
        self.resilient_weight_deacay = getattr(args, 'resilient_weight_decay', 0.0)
        self.normalized_mu = 1# args.mu_max if args.normalize_mu else 1
        self.dual_num_blocks = args.dual_num_blocks if hasattr(args, 'dual_num_blocks') else args.num_blocks
        self.mu_uncons = args.mu_uncons
        self.r_min = args.r_min

        if eval_mode == 'unrolling':
            self.blocks = nn.ModuleList()
            for _ in range(self.dual_num_blocks):
                if args.architecture == 'GNN':
                    self.blocks.append(GNN(self.num_features_list, 1.0, primal=False).to(device))
                else:
                    norm_layer = getattr(args, 'dual_norm_layer', 'batch')
                    self.blocks.append(PrimalGNN(num_features_list=self.num_features_list, 
                        P_max=5.0, k_hops = args.dual_k_hops, norm_layer = norm_layer, dropout_rate = args.dropout_rate,
                        conv_layer_normalize = args.conv_layer_normalize,
                        primal=False
                        ).to(device))


    def forward(self, block_id, mu, p, edge_index_l, edge_weight_l, transmitters_index):
        if not hasattr(self, 'cons_lvl'):
            m = mu.view(-1, self.n).shape[0]
            self.cons_lvl = torch.cat([self.r_min * torch.ones(m, int(np.floor(self.constrained_subnetwork*self.n))).to(self.device), 
                                        torch.zeros(m, int(np.ceil((1-self.constrained_subnetwork)*self.n))).to(self.device)], dim=1).view(-1, 1)
        x = torch.cat((p/self.P_max, mu/self.normalized_mu, self.cons_lvl), dim=1)
        mu = self.blocks[block_id](x, edge_index_l, edge_weight_l, transmitters_index, activation=None)
        if self.constrained_subnetwork < 1:
            mu = mu.view(-1, self.n)
            mu = torch.cat([mu[:, :int(np.floor(self.constrained_subnetwork*self.n))], 
                            self.mu_uncons * torch.ones(mu.shape[0], int(np.ceil((1-self.constrained_subnetwork)*self.n))).to(self.device)], dim=1)           
            mu = mu.view(-1, 1)
        return mu * self.normalized_mu


    def loss(self, outputs_list, r_min, num_graphs, constraint_eps=None, metric='rates', **kwargs):
        Lagrangian_list = []
        violation_list = []
        target = kwargs.get('target', None)
        self.resilient_weight_deacay = kwargs.get('resilient_weight_decay', 0.0)
        supervised = kwargs.get('supervised', False)

        for (mu, p, rates, _) in outputs_list:
            rates, mu, p = rates.view(num_graphs, self.n), mu.view(num_graphs, self.n), p.view(num_graphs, self.n)
            violation = torch.minimum(rates[:, :int(np.floor(self.constrained_subnetwork*self.n))] - r_min, torch.zeros_like(rates[:, :int(np.floor(self.constrained_subnetwork*self.n))]))

            # simple loss
            if supervised:
                assert target is not None
                target = target.view(num_graphs, self.n)
                L = ((mu[:, :int(np.floor(self.constrained_subnetwork*self.n))] - target[:, :int(np.floor(self.constrained_subnetwork*self.n))]) ** 2).mean(1)
            else:
                L = -1 * lagrangian_fn(rates, mu, self.cons_lvl, p=p, metric=metric, constrained_subnetwork=self.constrained_subnetwork)
                
            # resilience loss
            if self.resilient_weight_deacay > 0:
                L = L + mu[:, :int(np.floor(self.constrained_subnetwork*self.n))].norm(p=1, dim=1)**2/(2*self.resilient_weight_deacay)
                violation_list.append(violation.norm(p=1, dim=1) + torch.norm(mu[:, :int(np.floor(self.constrained_subnetwork*self.n))], p=1, dim=1)/self.resilient_weight_deacay)
            else:
                violation_list.append(violation.sum(1).abs())
            Lagrangian_list.append(L)

        # constraine loss
        if constraint_eps is not None:
            constraint_loss = self.descending_constraints(torch.stack(violation_list), constraint_eps)
            return L.mean(), constraint_loss.unsqueeze(1)
        else:
            return L.mean(), torch.zeros_like(L.mean()).to(self.device)



    def descending_constraints(self, L_list, constraint_eps):
        return (L_list[1:] - (1-constraint_eps) * L_list[:-1]).mean(-1)



    def DA(self, primal_model, data, lr_dual, resilient_weight_decay, n, r_min, noise_var, num_iters, ss_param, 
           mu_init, mu_uncons, device, adjust_constraints=True, fix_mu_uncons=True):
        primal_model.eval()
        if hasattr(primal_model, 'cons_lvl'):
            delattr(primal_model, 'cons_lvl')

        data = data.to(self.device)
        edge_index_l, edge_weight_l, a_l, transmitters_index, num_graphs = \
            data.edge_index_l, data.edge_weight_l, data.weighted_adjacency_l, \
            data.transmitters_index, getattr(data, 'num_graphs', 1)

        mu_over_time = []
        L_over_time = []
        all_Ps = []
        all_rates = []

        # initialize
        if self.constrained_subnetwork < 1:
            mu = torch.cat((mu_init * torch.rand(num_graphs, int(np.floor(self.constrained_subnetwork*self.n))).to(self.device), 
                           mu_uncons * torch.ones(num_graphs, int(np.ceil((1-self.constrained_subnetwork)*self.n))).to(self.device)), dim=1)
            m = mu.shape[0]
            r_min = torch.cat([r_min * torch.ones(m, int(np.floor(self.constrained_subnetwork*self.n))).to(self.device), 
                                    torch.zeros(m, int(np.ceil((1-self.constrained_subnetwork)*self.n))).to(self.device)], dim=1).view(-1, 1)
            mu = mu.view(num_graphs * self.n, 1)
            r_min = r_min.view(num_graphs * self.n, 1)

        else:
            mu = 0.1 * torch.rand(num_graphs * n, 1).to(device)

        # DA
        for _ in range(num_iters):
            # pass the instantaneous fading arrays (network states), augmented with dual variables, at each step into the main GNN to get RRM decisions
            p = primal_model(mu.detach(), edge_index_l, edge_weight_l, transmitters_index)
            if isinstance(p, list):
                p = p[-1]
            gamma = torch.ones(num_graphs * n, 1).to(device) # user selection decisions are always the same (single Rx per Tx)
            rates = calc_rates(p.detach(), gamma, a_l[:, :, :], noise_var,ss_param)
            L, _ = primal_model.loss(rates, mu, p, 
                                    metric='power' if hasattr(primal_model.args, 'metric') and primal_model.args.metric == 'power' else 'rates')
                                    # constrained_subnetwork=self.constrained_subnetwork if fix_mu_uncons else None)

            # update the dual variables
            if resilient_weight_decay > 0:
                mu = (1-1/resilient_weight_decay) * mu + lr_dual * (r_min - rates.detach())
            else:
                slack_value = 0
                mu += lr_dual * (r_min - rates.detach())

            if self.constrained_subnetwork and fix_mu_uncons:
                mu = mu.view(-1, self.n)
                mu = torch.cat([mu[:, :int(np.floor(self.constrained_subnetwork*self.n))], 
                                mu_uncons * torch.ones(mu.shape[0], int(np.ceil((1-self.constrained_subnetwork)*self.n))).to(self.device)], dim=1)
                mu = mu.view(-1, 1)
            mu.data.clamp_(0)

            slack_value = - mu.detach()/resilient_weight_decay if adjust_constraints else 0
            new_constraints_value = rates.detach() - r_min - slack_value
            violation = torch.minimum(new_constraints_value, torch.zeros_like(new_constraints_value)).abs()
            violation_rate = (violation>0).sum().float()/violation.numel()
            

            # store results
            mu_over_time.append(mu.detach().cpu().squeeze())
            L_over_time.append(L.detach().cpu().squeeze())
            all_Ps.append(p.detach().cpu().squeeze())
            all_rates.append(rates.detach().cpu().squeeze())

        return mu_over_time, L_over_time, all_Ps, all_rates, \
            {'violation': violation, 'violation_rate': violation_rate.item(), 'slack_value': slack_value,
             'rate_mean': torch.mean(rates.view(num_graphs, n).mean(1).detach().cpu()).tolist(),
             'constrained_rate_mean': torch.mean(rates.view(num_graphs, n)[:, :int(np.floor(self.constrained_subnetwork*self.n))].mean(1).detach().cpu()).tolist(),
             'unconstrained_rate_mean': torch.mean(rates.view(num_graphs, n)[:, int(np.floor(self.constrained_subnetwork*self.n)):].mean(1).detach().cpu()).tolist()}




