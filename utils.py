import numpy as np
import torch
from numpy import linalg as LA
from torch_geometric.data import Data, Dataset
import pandas as pd  
import os

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
    




def save_results_to_csv(all_epoch_results, args, pathname=None, filename='results.csv'):
    assert pathname is not None, 'Please provide a pathname to save the results'
    
    # Create full filepath
    filepath = f'{pathname}/{filename}'
    
    # Extract the metrics you want to save
    modes = ['SA', 'unrolledPrimal', 'unrolling', 'full_power', 'random']
    available_modes = [mode for mode in modes if (mode, 'rate_mean') in all_epoch_results]
    
    # Dictionary to store results
    data = {
        'timestamp': pd.Timestamp.now(),
        'f_min': args.r_min,
        'P_max': args.P_max,
        'n': args.n,
        'm': args.m,
        'constrained_subnetwork': args.constrained_subnetwork,
        'graph_type': args.graph_type,
        'sparse_graph_thresh': args.sparse_graph_thresh,
        'TxLoc_perturbation_ratio': args.TxLoc_perturbation_ratio,
        'R': args.R,
    }
    
    # Add metrics for each available mode
    for mode in available_modes:
        if (mode, 'rate_mean') in all_epoch_results:
            data[f'{mode}_rate_mean'] = all_epoch_results[mode, 'rate_mean'][-1]
        if (mode, 'violation_rate') in all_epoch_results:
            data[f'{mode}_violation_rate'] = all_epoch_results[mode, 'violation_rate'][-1]
        if (mode, 'mean_violation') in all_epoch_results:
            data[f'{mode}_mean_violation'] = all_epoch_results[mode, 'mean_violation'][-1]
        
        # Add percentiles if available
        for percentile in [5, 10, 15, 20, 30]:
            key = (mode, f'rate_{percentile}th_percentile')
            if key in all_epoch_results:
                data[f'{mode}_rate_{percentile}th_percentile'] = all_epoch_results[key][-1]
    
    # Create DataFrame
    results_df = pd.DataFrame([data])
    
    try:
        # Check if file exists and load it
        existing_df = pd.read_csv(filepath)
        # Append new results
        updated_df = pd.concat([existing_df, results_df], ignore_index=True)
        updated_df.to_csv(filepath, index=False)
        print(f"Results appended to {filepath}")
    except FileNotFoundError:
        # Create new file if it doesn't exist
        results_df.to_csv(filepath, index=False)
        print(f"Results saved to {filepath}")
        
    return results_df



def read_and_analyze_results(experiment_path, filename='results.csv'):
    """
    Read the results CSV file and perform basic analysis
    
    Parameters:
        experiment_path: Path to the experiment directory
        filename: Name of the CSV file
        
    Returns:
        pandas DataFrame with the results or None if error
    """
    import os
    import pandas as pd
    
    # Create full filepath
    filepath = os.path.join(experiment_path, filename)
    
    # Read the CSV file
    try:
        results_df = pd.read_csv(filepath)
        print(f"Successfully loaded {len(results_df)} result entries")
        
        # Display basic statistics
        print("\nBasic statistics for key metrics:")
        numeric_cols = [col for col in results_df.columns if col.endswith('_rate_mean') 
                       or col.endswith('violation_rate') 
                       or col.endswith('_mean_violation')]
        print(results_df[numeric_cols].describe())
        
        return results_df
        
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return None
    except pd.errors.EmptyDataError:
        print(f"Error: No data in file at {filepath}")
        return None
    except Exception as e:
        print(f"Error reading the file: {e}")
        return None


def plot_results_vs_R(results_df, metrics=['rate_mean', 'violation_rate'], save_dir=None):
    """
    Plot metrics vs R from results DataFrame
    
    Parameters:
        results_df: pandas DataFrame with results
        metrics: list of metrics to plot (without method prefix)
        save_dir: directory to save plots (uses current directory if None)
    """
    import matplotlib.pyplot as plt
    import os
    
    if results_df is None or 'R' not in results_df.columns:
        print("Cannot create plots: DataFrame is None or doesn't contain R column")
        return
    
    # Check if we have different n values
    if 'n' in results_df.columns:
        n_values = results_df['n'].unique()
    else:
        n_values = [None]  # If n is not in the dataframe
    
    # Create plots for each n value and metric
    for metric in metrics:
        for n_value in n_values:
            # Filter data for this n if applicable
            if n_value is not None:
                df_n = results_df[results_df['n'] == n_value]
                n_suffix = f"_n{n_value}"
            else:
                df_n = results_df
                n_suffix = ""
            
            # Find columns for this metric
            metric_cols = [col for col in df_n.columns if col.endswith(f'_{metric}')]
            
            if not metric_cols:
                continue
                
            # Create plot
            plt.figure(figsize=(10, 6))
            
            # Plot each method
            for col in metric_cols:
                method_name = col.replace(f'_{metric}', '')
                # Sort by R to ensure line is drawn correctly
                plot_data = df_n.sort_values('R')
                plt.plot(plot_data['R'], plot_data[col], 
                         marker='o', linewidth=2, label=method_name)
            
            # Add labels and styling
            plt.xlabel('R (Environment Size)', fontsize=14)
            plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
            
            title_n = f" (n={n_value})" if n_value is not None else ""
            plt.title(f'{metric.replace("_", " ").title()} vs Environment Size{title_n}', fontsize=16)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend(fontsize=12)
            plt.tight_layout()
            
            # Save the figure
            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f'{metric}_vs_R{n_suffix}.pdf'))
                print(f"Created plot for {metric}{' (n='+str(n_value)+')' if n_value is not None else ''}")
            else:
                plt.show()





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