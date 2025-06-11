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
        'r_min': args.r_min,
        'P_max': args.P_max,
        'n': args.n,
        'constrained_subnetwork': args.constrained_subnetwork,
        'graph_type': args.graph_type,
        'sparse_graph_thresh': args.sparse_graph_thresh,
        'TxLoc_perturbation_ratio': args.TxLoc_perturbation_ratio,
        'R': args.R,
        'best': args.best,
    }
    
    # Add metrics for each available mode
    for mode in available_modes:
        if (mode, 'rate_mean') in all_epoch_results:
            data[f'{mode}_rate_mean'] = all_epoch_results[mode, 'rate_mean'][-1]
        if (mode, 'violation_rate') in all_epoch_results:
            data[f'{mode}_violation_rate'] = all_epoch_results[mode, 'violation_rate'][-1]
        if (mode, 'mean_violation') in all_epoch_results:
            data[f'{mode}_mean_violation'] = all_epoch_results[mode, 'mean_violation'][-1]
        if (mode, 'constrained_mean_rate') in all_epoch_results:
            data[f'{mode}_constrained_mean_rate'] = all_epoch_results[mode, 'constrained_mean_rate'][-1]
        if (mode, 'unconstrained_mean_rate') in all_epoch_results:
            data[f'{mode}_unconstrained_mean_rate'] = all_epoch_results[mode, 'unconstrained_mean_rate'][-1]
        
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



def read_results(experiment_path, filename='results.csv'):
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
        numeric_cols = [col for col in results_df.columns if col.endswith('_mean_violation')]
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


def plot_ood_results(results_df, metrics=['rate_mean', 'mean_violation'], varying_param='r_min', 
                     fixed_params=None, save_dir=None):
    """
    Plot metrics vs a varying parameter while keeping other parameters fixed
    
    Parameters:
        results_df: pandas DataFrame with results
        metrics: list of metrics to plot (without method prefix)
        varying_param: parameter to vary on x-axis
        fixed_params: dictionary of parameters to keep fixed {param_name: value}
        save_dir: directory to save plots (uses current directory if None)
    """
    import matplotlib.pyplot as plt
    import os

    plot_labels = {
        'SA': 'State-augmented GNN',
        'unrolledPrimal': 'Unrolled GNN',
        'unrolling': 'Primal-dual Unrolling (Ours)',
        'full_power': 'Full Power',
        'random': 'Random'
    }
    
    import matplotlib.cm as cm
    cmap = cm.get_cmap('tab10')  # You can try other colormaps like 'viridis', 'plasma', 'Set1', 'tab10'

    # Define consistent colors for modes using the colormap
    colors = {
        'SA': cmap(0),             
        'unrolledPrimal': cmap(1),  
        'unrolling': cmap(2),      
        'random': cmap(3),         
        'full_power': cmap(4)      
    }

    if results_df is None or varying_param not in results_df.columns:
        print(f"Cannot create plots: DataFrame is None or doesn't contain {varying_param} column")
        return
    
    # Filter data based on fixed parameters
    filtered_df = results_df.copy()
    param_suffix = ""
    
    if fixed_params:
        for param, value in fixed_params.items():
            if param in filtered_df.columns:
                filtered_df = filtered_df[filtered_df[param] == value]
                param_suffix += f"_{param}={value}"
        
        if len(filtered_df) == 0:
            print("No data points match the specified fixed parameters")
            return
    
    # Create plots for each metric
    for metric in metrics:
        # Find columns for this metric
        metric_cols = [col for col in filtered_df.columns if col.endswith(f'_{metric}')]
        
        if not metric_cols:
            print(f"No columns found for metric: {metric}")
            continue
            
        # Create plot
        plt.figure(figsize=(10, 6))
        
        # Plot each method
        for col in metric_cols:
            method_name = col.replace(f'_{metric}', '')
            # Sort by varying_param to ensure line is drawn correctly
            plot_data = filtered_df.sort_values(varying_param)
            plt.plot(plot_data[varying_param], plot_data[col], color=colors[method_name],
                     marker='o', linewidth=2, label=plot_labels[method_name])
        
        # Add labels and styling
        plt.xlabel(varying_param, fontsize=14)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=14)
        
        # Create title with fixed parameter information
        if fixed_params:
            fixed_params_str = ", ".join([f"{param}={value}" for param, value in fixed_params.items()])
            title = f'{metric.replace("_", " ").title()} vs {varying_param} (Fixed: {fixed_params_str})'
        else:
            title = f'{metric.replace("_", " ").title()} vs {varying_param}'
            
        plt.title(title, fontsize=16)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(fontsize=12)
        plt.tight_layout()
        
        # Save the figure
        if save_dir:
            os.makedirs(f"{save_dir}/figs", exist_ok=True)
            filename = f'{metric}_vs_{varying_param}{param_suffix}.pdf'
            plt.savefig(os.path.join(save_dir, "figs", filename))
            print(f"Created plot: {filename}")
        else:
            plt.show()