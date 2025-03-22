import numpy as np
import torch
from numpy import linalg as LA
from torch_geometric.data import Data, Dataset

from collections import defaultdict

# calculating rates
def calc_rates(p, gamma, h, noise_var, ss_param=1.0):
    """
    calculate rates for a batch of b networks, each with m transmitters and n recievers
    inputs:
        p: bm x 1 tensor containing transmit power levels
        gamma: bn x 1 tensor containing user scheduling decisions
        h: b x (m+n) x (m+n) weighted adjacency matrix containing instantaneous channel gains
        noise_var: scalar indicating noise variance
        training: boolean variable indicating whether the models are being trained or not; during evaluation, 
        entries of gamma are forced to be integers to satisfy hard user scheudling constraints
        
    output:
        rates: bn x 1 tensor containing user rates
    """
    b = h.shape[0]
    p = p.view(b, -1, 1)
    gamma = gamma.view(b, -1, 1)
    m = p.shape[1]
    
    combined_p_gamma = torch.bmm(p, torch.transpose(gamma, 1, 2))
    signal = torch.sum(combined_p_gamma * h[:, :m, m:], dim=1)
    interference = torch.sum(combined_p_gamma * torch.transpose(h[:, m:, :m], 1, 2), dim=1)
    
    rates = torch.log2(1 + signal / (noise_var + interference/ss_param)).view(-1, 1)
    
    return rates

# baseline ITLinQ method
def ITLinQ(H_raw, Pmax, noise_var, PFs):
    H = H_raw * Pmax / noise_var
    n = np.shape(H)[0]
    prity = np.argsort(PFs)[-1:-n-1:-1]
    flags = np.zeros(n)
    M = 10 ** 2.5
    eta = 0.5
    flags[prity[0]] = 1
    for pair in prity[1:]:
        SNR = H[pair,pair]
        INRs_in = [H[TP,pair] for TP in range(n) if flags[TP]]
        INRs_out = [H[pair,UE] for UE in range(n) if flags[UE]]
        max_INR_in = max(INRs_in)
        max_INR_out = max(INRs_out)
        if max(max_INR_in,max_INR_out) <= M * (SNR ** eta):
            flags[pair] = 1
    return flags * Pmax

def convert_channels(a, snr, conversion=None):
    a_flattened = a[a > 0]
    a_flattened_log = np.log(1 + snr * a_flattened)
    a_norm = LA.norm(a_flattened_log)
    a_log = np.log(1 + snr * a)
    a_log[a == 0] = 0
    
    if conversion == 'spectral':
        return a_log/LA.eig(a_log)[0].max().real
    else:
        return a_log / a_norm

class Data_modTxIndex(Data):
    def __init__(self,
                 y=None,
                 edge_index_l=None,
                 edge_weight_l=None,
                 edge_index=None,
                 edge_weight=None,
                 weighted_adjacency=None,
                 weighted_adjacency_l=None,
                 transmitters_index=None,
                 init_long_term_avg_rates=None,
                 num_nodes=None,
                 m=None,
                 target=None):
        super().__init__()
        self.y = y
        self.edge_index_l = edge_index_l
        self.edge_weight_l = edge_weight_l
        self.edge_index = edge_index
        self.edge_weight = edge_weight
        self.weighted_adjacency = weighted_adjacency
        self.weighted_adjacency_l = weighted_adjacency_l
        self.transmitters_index = transmitters_index
        self.init_long_term_avg_rates = init_long_term_avg_rates
        self.num_nodes = num_nodes
        self.m = m
        self.target=target
                
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'transmitters_index':
            return self.m
        else:
            return super().__inc__(key, value, *args, **kwargs)
        
class WirelessDataset(Dataset):
    def __init__(self, data_list):
        super().__init__(None, None, None)
        self.data_list = data_list

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx], idx
    



  
    # # create optimizer
    # # optimizer = torch.optim.Adam(model.parameters(), lr=lr_main, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    # # optimizer = torch.optim.SGD(model.parameters(), lr=lr_main, momentum=0.9)
    # # optimizer = torch.optim.RMSprop(model.parameters(), lr=lr_main, alpha=0.9)
    # optimizer = torch.optim.AdamW(model.parameters(), lr=lr_main, weight_decay=1e-5)
    # # optimizer = torch.optim.Adamax(model.parameters(), lr=lr_main)
    # # optimizer = torch.optim.Adagrad(model.parameters(), lr=lr_main)
    # # optimizer = torch.optim.Adamax(model.parameters(), lr=lr_main)
    # # optimizer = torch.optim.LBFGS(model.parameters(), lr=lr_main)
    # # optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr_main)
    # # optimizer = torch.optim.ASGD(model.parameters(), lr=lr_main)



def simple_policy(loader, P_max, n, r_min, noise_var, ss_param, device):
    randPolicy_results = defaultdict(list)
    full_power_results = defaultdict(list)
    for data, batch_idx in loader:
        data = data.to(device)
        a_l, num_graphs = data.weighted_adjacency_l,  data.num_graphs

        # random policy
        # p = torch.relu_(P_max * (torch.randn(num_graphs * n).to(device) + 0.5)/2)
        p = P_max * torch.rand(num_graphs * n).to(device)
        gamma = torch.ones(num_graphs * n, 1).to(device)
        rates = calc_rates(p, gamma, a_l[:, :, :], noise_var, ss_param)
        violation = torch.minimum(rates - r_min, torch.zeros_like(rates)).abs()
        
        randPolicy_results['rate_mean'].append(torch.mean(rates.view(num_graphs, n).mean(1).detach().cpu()).tolist())
        randPolicy_results['rate_min'].append(torch.min(rates.detach().cpu()).tolist())
        for percentile in [1, 5, 10, 15, 20, 30, 40, 50]:
                randPolicy_results[f'rate_{percentile}th_percentile'].append(-1*np.percentile(-1 * violation.view(num_graphs, n).detach().cpu().numpy(), percentile, axis=1).mean().tolist())
        randPolicy_results['all_Ps'].append(p.detach().cpu().numpy())
        randPolicy_results['all_rates'].append(rates.detach().cpu().numpy())


        # Full power
        p = P_max * torch.ones(num_graphs * n).to(device)
        rates = calc_rates(p, gamma, a_l[:, :, :], noise_var, ss_param)
        violation = torch.minimum(rates - r_min, torch.zeros_like(rates)).abs() 
        full_power_results['rate_mean'].append(torch.mean(rates.view(num_graphs, n).mean(1).detach().cpu()).tolist())
        full_power_results['rate_min'].append(torch.min(rates.detach().cpu()).tolist())
        for percentile in [1, 5, 10, 15, 20, 30, 40, 50]:
                full_power_results[f'rate_{percentile}th_percentile'].append(-1*np.percentile(-1 * violation.view(num_graphs, n).detach().cpu().numpy(), percentile, axis=1).mean().tolist())
        full_power_results['all_Ps'].append(p.detach().cpu().numpy())
        full_power_results['all_rates'].append(rates.detach().cpu().numpy())
        
    return randPolicy_results, full_power_results