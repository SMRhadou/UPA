
from core.gnn import PrimalGNN, GNN
from utils import calc_rates

import torch
import torch.nn as nn
from collections import defaultdict
import numpy as np

def lagrangian_fn(rates, mu, r_min, p=None, metric='rates', constrained_subnetwork=None):
    sum_rates = torch.sum(rates, dim=1)
    if p is not None:
        sum_power = torch.sum(p, dim=1)

    if constrained_subnetwork is not None:
        rates = rates[:, :int(np.floor(constrained_subnetwork*rates.shape[1]))]
        mu = mu[:, :int(np.floor(constrained_subnetwork*mu.shape[1]))]
        if p is not None:
            p = p[:, :int(np.floor(constrained_subnetwork*p.shape[1]))]
    
    if metric == 'rates':
        return -1 * sum_rates + torch.sum((mu * (r_min - rates)), dim=1)    # Posed as a minimization problem
    elif metric == 'power':
        assert p is not None
        return sum_power + torch.sum((mu * (r_min - rates)), dim=1)

class PrimalModel(nn.Module):
    def __init__(self, args, device,  normalize_mu=False, unrolled=False):
        super(PrimalModel, self).__init__()
        self.args = args
        self.device = device
        self.normalized_mu = args.mu_max if normalize_mu else 1

        self.num_graphs = args.batch_size * args.num_samplers
        self.num_features_list = [1] + [args.primal_hidden_size] * args.primal_num_sublayers
        if args.architecture == 'GNN':
            self.model = GNN(self.num_features_list, args.P_max).to(device)
        elif args.architecture == 'PrimalGNN':
            norm_layer = getattr(args, 'primal_norm_layer', 'batch')  # Default to 0.0 if not defined
            self.model = PrimalGNN(num_features_list=self.num_features_list, 
                        P_max=args.P_max, k_hops = args.primal_k_hops, norm_layer = norm_layer, dropout_rate=args.dropout_rate,
                        conv_layer_normalize = args.conv_layer_normalize
                        ).to(device)

    def forward(self, mu, edge_index_l, edge_weight_l, transmitters_index):
        p = self.model(mu/self.normalized_mu, edge_index_l, edge_weight_l, transmitters_index)
        if getattr(self.args, 'crop_p', 0) > 0:
            p = torch.clamp(p, min=1e-5) # to avoid log(0) in calc_rates
        return p
    
    def loss(self, rates, mu, p=None, constrained=False, metric='rates'):
        if mu.shape[0] != self.num_graphs * self.args.n:
            num_graphs = mu.shape[0] // self.args.n
        else:
            num_graphs = self.num_graphs
        rates, mu = rates.view(num_graphs, self.args.n), mu.view(num_graphs, self.args.n)
        p = p.view(num_graphs, self.args.n)
        L = lagrangian_fn(rates, mu, self.args.r_min, p=p, metric=metric, constrained_subnetwork=self.args.constrained_subnetwork).mean()
        if not constrained:
            return L
        else:
            raise NotImplementedError

    def descending_constraints(self, output, target):
        pass

    def sanity_check(self, data, mu_test, noise_var, ss_param):
        data = data.to(self.device)
        y, edge_index_l, edge_weight_l, _, \
        _, _, a_l, transmitters_index, num_graphs = \
            data.y, data.edge_index_l, data.edge_weight_l, data.edge_index, data.edge_weight, \
            data.weighted_adjacency, data.weighted_adjacency_l, \
            data.transmitters_index, data.num_graphs
    
        p_test = self.model(mu_test.detach(), edge_index_l, edge_weight_l, transmitters_index)
        gamma_test = torch.ones(num_graphs * self.args.n, 1).to(self.device) # user selection decisions are always the same (single Rx per Tx)
        rates_test = calc_rates(p_test, gamma_test, a_l[:, :, :], noise_var, ss_param)
        rates_test, mu_test = rates_test.view(num_graphs, self.args.n), mu_test.view(num_graphs, self.args.n)
        p_test = p_test.view(num_graphs, self.args.n)
        L = lagrangian_fn(rates_test, mu_test, self.args.r_min, p=p_test, metric='rates', constrained_subnetwork=self.args.constrained_subnetwork)

        return L.detach().cpu().numpy()



class DualModel(nn.Module):
    def __init__(self, args, device=None, eval_mode='unrolling'):
        super(DualModel, self).__init__()
        self.device = device
        self.eval_mode = eval_mode
        self.num_blocks = args.num_blocks
        self.num_features_list = [2] + [args.dual_hidden_size] * args.dual_num_sublayers
        self.n = args.n
        self.P_max = args.P_max
        self.constrained_subnetwork = args.constrained_subnetwork
        self.resilient_weight_deacay = getattr(args, 'resilient_weight_decay', 0.0)

        if eval_mode == 'unrolling':
            self.blocks = nn.ModuleList()
            for _ in range(self.num_blocks):
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
        x = torch.cat((p/self.P_max, mu), dim=1)
        mu = self.blocks[block_id](x, edge_index_l, edge_weight_l, transmitters_index)
        if self.constrained_subnetwork < 1:
            mu = mu.view(-1, self.n)
            mu = torch.cat([mu[:, :int(np.floor(self.constrained_subnetwork*self.n))], torch.zeros(mu.shape[0], int(np.ceil((1-self.constrained_subnetwork)*self.n))).to(self.device)], dim=1)
            mu = mu.view(-1, 1)
        return mu


    def loss(self, outputs_list, r_min, num_graphs, constraint_eps=None, metric='rates', **kwargs):
        Lagrangian_list = []
        violation_list = []
        target = kwargs.get('target', None)
        self.resilient_weight_deacay = kwargs.get('resilient_weight_decay', 0.0)
        supervised = kwargs.get('supervised', False)

        for (mu, p, rates, _) in outputs_list:
            rates, mu, p = rates.view(num_graphs, self.n), mu.view(num_graphs, self.n), p.view(num_graphs, self.n)
            violation = torch.minimum(rates - r_min, torch.zeros_like(rates))

            # simple loss
            if supervised:
                assert target is not None
                target = target.view(num_graphs, self.n)
                L = ((mu[:, :int(np.floor(self.constrained_subnetwork*self.n))] - target[:, :int(np.floor(self.constrained_subnetwork*self.n))]) ** 2).mean(1)
            else:
                L = -1 * lagrangian_fn(rates, mu, r_min, p=p, metric=metric, constrained_subnetwork=self.args.constrained_subnetwork)
                
            # resilience loss
            if self.resilient_weight_deacay > 0:
                L = L + mu.norm(p=1, dim=1)**2/(2*self.resilient_weight_deacay)
                violation_list.append(violation.norm(p=1, dim=1) + torch.norm(mu, p=1, dim=1)/self.resilient_weight_deacay)
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



    def DA(self, primal_model, data, lr_dual, resilient_weight_decay, n, r_min, noise_var, num_iters, ss_param, device, adjust_constraints=True):
        primal_model.eval()

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
            mu = torch.cat((0.01 * torch.rand(num_graphs, int(np.floor(self.constrained_subnetwork*self.n))).to(self.device), 
                           30*torch.ones(num_graphs, int(np.ceil((1-self.constrained_subnetwork)*self.n))).to(self.device)), dim=1)
            mu = mu.view(num_graphs * self.n, 1)
        else:
            mu = 0.1 * torch.rand(num_graphs * n, 1).to(device)

        # DA
        for _ in range(num_iters):
            # pass the instantaneous fading arrays (network states), augmented with dual variables, at each step into the main GNN to get RRM decisions
            p = primal_model(mu.detach(), edge_index_l, edge_weight_l, transmitters_index)

            gamma = torch.ones(num_graphs * n, 1).to(device) # user selection decisions are always the same (single Rx per Tx)
            rates = calc_rates(p.detach(), gamma, a_l[:, :, :], noise_var,ss_param)
            L = primal_model.loss(rates, mu, p, constrained=False, 
                                    metric='power' if hasattr(primal_model.args, 'metric') and primal_model.args.metric == 'power' else 'rates')

            # update the dual variables
            if resilient_weight_decay > 0:
                mu = (1-1/resilient_weight_decay) * mu + lr_dual * (r_min - rates.detach())
            else:
                slack_value = 0
                mu += lr_dual * (r_min - rates.detach())

            if self.constrained_subnetwork:
                mu = mu.view(-1, self.n)
                mu = torch.cat([mu[:, :int(np.floor(self.constrained_subnetwork*self.n))], 
                                30*torch.ones(mu.shape[0], int(np.ceil((1-self.constrained_subnetwork)*self.n))).to(self.device)], dim=1)
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
             'constrained_rate_mean': torch.mean(rates.view(num_graphs, n)[:, :int(np.floor(self.constrained_subnetwork*self.n))].mean(1).detach().cpu()).tolist()}




