import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import pandas as pd
import pickle

plt.rc('font', family='serif', serif='cm10', size=16)  # Bigger base font size
plt.rc('axes', labelsize=18, titlesize=20)  # Axis labels and titles
plt.rc('xtick', labelsize=16)  # X-axis tick labels
plt.rc('ytick', labelsize=16)  # Y-axis tick labels
plt.rc('legend', fontsize=16)  # Legend font size
plt.rc('figure', titlesize=22)  # Figure title
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def plot_testing(test_results, f_min, P_max, num_curves=15, num_agents=100, num_iters=400, batch_size=32, pathname=None,):
    assert pathname is not None, 'Please provide a pathname to save the figures'
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    multipliers = np.stack(test_results['test_mu_over_time']).squeeze().reshape(num_iters, -1, num_agents)
    # Define distinct colors for different agents
    colors = plt.cm.tab20(np.linspace(0, 1, num_curves))
    for i in range(num_curves):
        ax[0, 0].plot(multipliers[:,0,i]/15, label='agent {}'.format(i), color=colors[i])
        ax[1, 0].plot(multipliers[:,1,i]/15, label='agent {}'.format(i), color=colors[i])
    ax[0, 0].set_xlabel('iterations')
    ax[1, 0].set_xlabel('iterations')
    ax[0, 0].set_title('Dual Variables')
    ax[1, 0].set_title('Dual Variables')

    power_allocated = np.stack(test_results['all_Ps']).squeeze().reshape(num_iters, -1, num_agents)
    for i in range(num_curves):
        ax[0,1].plot(power_allocated[:,0,i]/P_max, label='agent {}'.format(i), color=colors[i])
        ax[1,1].plot(power_allocated[:,1,i]/P_max, label='agent {}'.format(i), color=colors[i]) 
    ax[0,1].set_xlabel('iterations')
    ax[1,1].set_xlabel('iterations')
    # ax[1].legend()
    ax[0,1].set_title('Power allocated')
    ax[1,1].set_title('Power allocated')

    rates = np.stack(test_results['all_rates']).squeeze().reshape(num_iters, -1, num_agents)
    for i in range(num_curves):
        ax[0,2].plot(rates[:,0,i], label='agent {}'.format(i), color=colors[i])
        ax[1,2].plot(rates[:,1,i], label='agent {}'.format(i), color=colors[i])
    ax[0,2].plot(f_min*np.ones_like(rates[0,:,6]), ':k', linewidth=1, label=r'r_min')
    ax[1,2].plot(f_min*np.ones_like(rates[1,:,6]), ':k', linewidth=1, label=r'r_min')
    # ax[2].legend()
    ax[0,2].set_xlabel('iterations')
    ax[1,2].set_xlabel('iterations')
    ax[0,2].set_title('Rates per agent')
    ax[1,2].set_title('Rates per agent')
    fig.tight_layout()
    fig.savefig(f'{pathname}SA_test_results.png')


    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    rate_mean = np.stack(test_results['all_rates']).reshape(num_iters, -1, num_agents).sum(-1)
    for i in range(num_curves):
        ax[0].plot(rate_mean[:,0], label='graph {}'.format(i), color=colors[i])
        ax[1].plot(rate_mean[:,1], label='graph {}'.format(i), color=colors[i])
    ax[0].set_xlabel('iterations')
    ax[1].set_xlabel('iterations')
    # ax[0].legend()
    ax[0].set_title('Sum of rates per graph')
    ax[1].set_title('Sum of rates per graph')
    fig.tight_layout()
    fig.savefig(f'{pathname}rates.png')


    L_over_time = np.stack(test_results['L_over_time'])
    # one plot
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(L_over_time.squeeze(0))
    ax.set_xlabel('iterations')
    ax.set_ylabel('Lagrangian')
    fig.tight_layout()
    fig.savefig(f'{pathname}SA_L_over_time.png')


def plot_constrainedvsUnconsrained_histograms(all_epoch_results, constrained_agents, args, pathname=None, modes_list=None):
    n = args.n
    P_max = args.P_max
    f_min = args.r_min

    # modes_list = ['SA', 'unrolling']
    plot_labels = {
        'SA': 'State-augmented GNN',
        'unrolledPrimal': 'Unrolled GNN',
        'unrolling': 'Primal-dual Unrolling',
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
    

    fig, ax = plt.subplots(1, 4, figsize=(16, 4))

    # colors = {'SA': 'darkorange', 'unrolling': 'green'}
    alpha = 0.5  # Transparency level

    # Define fixed bins for consistent comparison
    power_bins = np.linspace(0, 1, 25)  # From 0 to 1 for normalized power
    rate_bins = np.linspace(0, 12, 12*4)   # Adjust range based on your typical rate values

    for i, mode in enumerate(modes_list):
        if mode not in ['SA', 'unrolledPrimal', 'unrolling']:
            continue
        power_allocated = np.stack(all_epoch_results[mode, 'all_Ps'])[-1,:,-1].reshape(-1, n)
        rates = np.stack(all_epoch_results[mode, 'all_rates'])[-1,:,-1].reshape(-1, n)
        
        # Create histograms with transparent colors
        ax[0].hist((power_allocated[:,:constrained_agents]/P_max).reshape(-1,1), bins=power_bins, 
                     alpha=alpha, color=colors[mode], label=plot_labels[mode])
        ax[1].hist(rates[:,:constrained_agents].reshape(-1,1), bins=rate_bins, 
                     alpha=alpha, color=colors[mode], label=plot_labels[mode])
        ax[2].hist((power_allocated[:,constrained_agents:]/P_max).reshape(-1,1), bins=power_bins, 
                     alpha=alpha, color=colors[mode], label=plot_labels[mode])
        ax[3].hist(rates[:,constrained_agents:].reshape(-1,1), bins=rate_bins, 
                     alpha=alpha, color=colors[mode], label=plot_labels[mode])
    
    # Add a vertical line for r_min in the second subplot
    ax[1].axvline(f_min-0.2, color='red', linestyle='--', linewidth=1.5, label=r'$r_{min}$')

    # add a text under the first two plots saying constrained agents and another text under the last two plots saying unconstrained agents
    fig.text(0.25, 0, '(a) constrained agents', ha='center', va='top', fontsize=16)
    fig.text(0.75, 0, '(b) unconstrained agents', ha='center', va='top', fontsize=16)
    # Add labels and titles
    ax[0].set_xlabel(r'$\mathbf{p}/P_{max}$')
    # ax[0,0].set_ylabel('constrained agents')
    ax[1].set_xlabel('rates')
    ax[2].set_xlabel(r'$\mathbf{p}/P_{max}$')
    # ax[1,0].set_ylabel('unconstrained agents')
    ax[3].set_xlabel('rates')
    # ax[0,0].set_title(r'Power allocated \% P_max')
    # ax[0,1].set_title('Rates')
    
    # Add legend to the figure, not individual subplots
    handles, labels = ax[0].get_legend_handles_labels()
    
    # Remove individual legends from subplots if they exist
    for i in range(4):
        if ax[i].get_legend():
            ax[i].get_legend().remove()
    
    # Adjust figure layout to make space for the legend
    fig.tight_layout(rect=[0, 0, 1, 0.9])  # Leave space at the top for the legend
    
    # Place the legend above the subplots without a box
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95),
              fancybox=False, shadow=False, ncol=len(modes_list), frameon=False)
    
    # Save the figure with bbox_inches='tight' to ensure the legend is included
    fig.savefig(f'{pathname}constrained_agents_comparison.pdf', bbox_inches='tight')

    multiplier_bins = np.linspace(0, 5, 50)
    fig, ax = plt.subplots(figsize=(6, 4))
    for i, mode in enumerate(modes_list):
        if mode not in ['SA', 'unrolling']:
            continue
        multipliers = np.stack(all_epoch_results[mode, 'test_mu_over_time'])[-1,:,-1]
        multipliers = multipliers.reshape(multipliers.shape[0], -1, n)[:,:,:50]

        ax.hist(multipliers.reshape(-1), bins=multiplier_bins, 
                alpha=alpha, color=colors[mode], label=plot_labels[mode])
        
    ax.set_xlabel(r'$\boldsymbol{\lambda}$')
    ax.set_title(r'Dual variables')      
    
    fig.tight_layout()
    fig.savefig(f'{pathname}constrained_agents_comparison_mu.pdf')



def plotting_collective(all_epoch_results, f_min, P_max, num_curves=10, num_agents=100, num_iters=800, unrolling_iters=6, pathname=None, modes_list=None):
    assert pathname is not None, 'Please provide a pathname to save the figures'

    # Use a colormap instead of manually specifying colors
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

    # Define transparency levels for different lines
    alpha_main = 0.85  # For main lines
    alpha_secondary = 0.6  # For constrained/unconstrained lines
    
    # Define line styles for different metrics
    line_styles = ['-', '--', ':']
    
    # Define plot labels
    plot_labels = {
        'SA': 'State-augmented GNN',
        'unrolledPrimal': 'Unrolled GNN',
        'unrolling': 'Primal-dual Unrolling (Ours)',
    }
    
    # if all:
    #     modes_list = ['SA', 'unrolledPrimal', 'unrolling']
    # else:
    #     modes_list = ['SA']

    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    
    # Create legend handles to store only one entry per mode
    legend_handles = []
    
    # Define common x-axis length for visualization
    common_x_length = num_iters
    
    for mode in modes_list:
        if mode=='random' or mode=='full_power':
            continue
        # Get correct number of iterations for this mode
        mode_iters = num_iters if mode == 'SA' or mode == 'unrolledPrimal' else unrolling_iters
        
        power_allocated = np.stack(all_epoch_results[mode, 'all_Ps'])
        power_allocated = power_allocated.reshape(power_allocated.shape[0], power_allocated.shape[1], mode_iters, -1, num_agents)
        power_allocated = torch.tensor(power_allocated).squeeze(0).transpose(1,2).reshape(-1, mode_iters, num_agents)
        
        # Create x coordinates scaled to match common axis length 
        if mode == 'unrolling':
            # Scale x-coordinates to match the common x-axis length
            x_coords = np.linspace(0, common_x_length-1, mode_iters)
        else:
            x_coords = np.arange(mode_iters)
            
        # Plot with different line styles but same color for each mode
        # ax[0].plot(x_coords, power_allocated.mean((0,-1))/P_max, 
        #           color=colors[mode], linestyle=line_styles[0], 
        #           alpha = alpha_main, linewidth=2.0,
        #           label=f'{plot_labels[mode]} - Mean power')
                  
        ax[0].plot(x_coords, power_allocated[:,:,:int(np.floor(num_agents/2))].mean((0,-1))/P_max, 
                  color=colors[mode], linestyle=line_styles[0], 
                  alpha = alpha_secondary, linewidth=2.0,
                  label=f'{plot_labels[mode]} - Mean constrained power')
                  
        ax[0].plot(x_coords, power_allocated[:,:,int(np.floor(num_agents/2)):].mean((0,-1))/P_max, 
                  color=colors[mode], linestyle=line_styles[1], 
                  alpha = alpha_secondary, linewidth=2.0,
                  label=f'{plot_labels[mode]} - Mean unconstrained power')
        
        # Add a legend handle for this mode (solid line)
        from matplotlib.lines import Line2D
        legend_handles.append(Line2D([0], [0], color=colors[mode], lw=2, label=plot_labels[mode]))
        
        # Continue plotting for rates and violation with scaled x-coordinates
        rates = np.stack(all_epoch_results[mode, 'all_rates'])
        rates = rates.reshape(rates.shape[0], rates.shape[1], mode_iters, -1, num_agents)
        rates = torch.tensor(rates).squeeze(0).transpose(1,2).reshape(-1, mode_iters, num_agents)
        
        # ax[1].plot(x_coords, rates.mean((0,-1)),
        #           color=colors[mode], linestyle=line_styles[0], alpha=alpha_main, linewidth=2.0,)
                  
        ax[1].plot(x_coords, rates[:,:,:int(np.floor(num_agents/2))].mean((0,-1)),
                  color=colors[mode], linestyle=line_styles[0], alpha=alpha_secondary, linewidth=2.0,)
                  
        ax[1].plot(x_coords, rates[:,:,int(np.floor(num_agents/2)):].mean((0,-1)),
                  color=colors[mode], linestyle=line_styles[1], alpha=alpha_secondary, linewidth=2.0,)

        violation = torch.minimum(torch.zeros_like(rates[:,:,:int(np.floor(num_agents/2))]), 
                                rates[:,:,:int(np.floor(num_agents/2))]-f_min).abs()
                                
        ax[2].plot(x_coords, violation.mean((0,-1)), 
                  color=colors[mode], linestyle='-', linewidth=2)
        
        print(violation.mean((0,-1))[-1].item())

    # Set x-axis limits consistently for all subplots
    for a in ax:
        a.set_xlim(0, common_x_length-1)
        
    # Set labels for axes with larger font sizes
    ax[0].set_xlabel('iterations', fontsize=16)
    ax[0].set_ylabel(r'$\mathbf{p}/P_{max}$', fontsize=16)
    ax[0].grid(True, linestyle='--', alpha=0.7)
    ax[0].tick_params(axis='both', which='major', labelsize=14)
    
    ax[1].set_xlabel('iterations', fontsize=16)
    ax[1].set_ylabel(r'$\mathbf{r}(\boldsymbol{\theta})$', fontsize=16)
    ax[1].grid(True, linestyle='--', alpha=0.7)
    ax[1].tick_params(axis='both', which='major', labelsize=14)
    
    ax[2].set_xlabel('iterations', fontsize=16)
    ax[2].set_ylabel('mean violation', fontsize=16)
    ax[2].grid(True, linestyle='--', alpha=0.7)
    ax[2].tick_params(axis='both', which='major', labelsize=14)
    
    # Create a common legend for line styles
    from matplotlib.lines import Line2D
    line_style_handles = [
        # Line2D([0], [0], color='black', linestyle=line_styles[0], label='All agents'),
        Line2D([0], [0], color='black', linestyle=line_styles[0], label='Constrained agents'),
        Line2D([0], [0], color='black', linestyle=line_styles[1], label='Unconstrained agents')
    ]
    
    # Add legends with larger fonts
    ax[1].legend(handles=line_style_handles, loc='lower right', fontsize=14)
    ax[2].legend(handles=legend_handles, loc='upper right', fontsize=14)
    
    fig.tight_layout()
    fig.savefig(f'{pathname}collective_results.pdf')    

    print('ok!')



def plot_final_percentiles_comparison(all_epoch_results, f_min=None, pathname=None):
    """
    Plot a bar chart comparing the final values of different percentiles.
    
    Parameters:
        all_epoch_results (dict): Dictionary containing training results with percentile data
        f_min (float, optional): Minimum rate constraint for reference line
        pathname (str, optional): Path to save the figures
        
    Returns:
        fig: The matplotlib figure object
    """
    assert pathname is not None, 'Please provide a pathname to save the figures'
    # Create figure for combined chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Prepare metrics to plot
    metrics = [
        ('rate_mean', 'Obj'),
        ('constrained_mean_rate', 'cons'),
        ('unconstrained_mean_rate', 'uncons'),
        # ('rate_1th_percentile', '1%'),
        ('rate_5th_percentile', '5%'),
        ('rate_10th_percentile', '10%'),
        ('rate_15th_percentile', '15%'),
        ('rate_20th_percentile', '20%'),
        ('rate_30th_percentile', '30%'),
        # ('rate_40th_percentile', '40%'),
        # ('rate_50th_percentile', '50%')
    ]
    
    # Prepare data for all modes
    all_metric_values = []
    # Switched the order of 'random' and 'full_power'
    mode_labels = ['SA', 'unrolling', 'full_power', 'random']
    plot_labels = {
        'SA': 'State-augmented GNN',
        'unrolling': 'Primal-dual Unrolling',
        'full_power': 'Full Power',
        'random': 'Random'
    }
    
    # Define consistent colors for modes
    colors = {'SA': 'darkorange', 'unrolling': 'green', 'random': 'lightblue', 'full_power': 'lightgray'}
    # Added transparency for 'random' and 'full_power'
    alphas = {'SA': 0.6, 'unrolling': 0.6, 'random': 0.7, 'full_power': 0.7}

    for mode in mode_labels:
        # Get values for this mode
        metric_values = []
        for metric_key, _ in metrics:
            key = (mode, metric_key)
            if key in all_epoch_results and len(all_epoch_results[key]) > 0:
                metric_values.append(all_epoch_results[key][-1])
            else:
                metric_values.append(0)  # Handle missing data
        all_metric_values.append(metric_values)
    
    # Set width of bars and positions
    bar_width = 0.25
    r1 = np.arange(0, 1.25*len(metrics), 1.25)
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]  # Adjust for 4th mode
    
    # Create bars
    bars = []
    for i, (values, mode) in enumerate(zip(all_metric_values, mode_labels)):
        pos = [r1, r2, r3, r4][i]
        # Apply transparency (alpha) for 'random' and 'full_power'
        bar = ax.bar(pos, values, width=bar_width, color=colors[mode], alpha=alphas[mode], label=plot_labels[mode])
        bars.append(bar)
    
    # Add value labels on top of each bar
    for i, mode_bars in enumerate(bars):
        for bar in mode_bars:
            height = bar.get_height()
            if height > 0:  # Only label non-zero bars
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=7, rotation=45)
    
    # Add minimum rate constraint line if provided
    if f_min is not None:
        ax.axhline(y=f_min, color='r', linestyle='--', linewidth=1, label=r'$r_{min}$')
    
    # Customize the plot
    ax.set_ylabel('Rate')
    # ax.set_title('Violation Percentiles Comparison')
    ax.set_xticks([r + 2*bar_width for r in r1.tolist()])
    ax.set_xticklabels([label for _, label in metrics])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    fig.tight_layout()
    fig.savefig(f'{pathname}final_percentiles_comparison.png')
    

def hist_power(test_results, P_max, name='random', pathname=None):
    assert pathname is not None, 'Please provide a pathname to save the figures'
    power_allocated = np.stack(test_results['all_Ps'])
    rates = np.stack(test_results['all_rates'])
    
    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    # for i in range(num_curves):
    ax[0].hist(power_allocated[0,:100]/P_max, bins=20)
    ax[1].hist(rates[1,:100], bins=20) 
    ax[0].set_xlabel(r'p/p_{max}')
    ax[1].set_xlabel('rates')
    # ax[1].legend()
    ax[0].set_title(r'Power allocated \% P_max')
    ax[1].set_title('Rates')
    fig.tight_layout()
    fig.savefig(f'{pathname}/results_{name}.png')




def plot_raw_data_histograms(raw_data_files, metrics=['all_Ps', 'all_rates'], mode='unrolling', 
                             experiment_labels=None, save_path=None):
    """
    Load raw data from pickle files and plot histograms for comparison side by side
    
    Parameters:
        raw_data_files: List of paths to raw data pickle files
        metrics: List of metrics to plot ['all_Ps', 'all_rates']
        mode: Model mode to plot ('unrolling', 'SA', etc.)
        experiment_labels: Labels for experiments (defaults to experiment path names)
        save_path: Path to save generated plots
    """
    if len(raw_data_files) < 2:
        print("Need at least two raw data files for comparison")
        return
    
    if isinstance(metrics, str):
        metrics = [metrics]  # Convert single metric to list
        
    # Use default labels if not provided
    if experiment_labels is None:
        experiment_labels = [os.path.basename(os.path.dirname(f)) for f in raw_data_files]
        
    # Create subplot figure
    fig, axes = plt.subplots(1, len(metrics), figsize=(6*len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]  # Make it iterable for single metric case
    
    # Process each metric in its own subplot
    for ax_idx, metric in enumerate(metrics):
        data_key = f"{mode}_{metric}"
        all_data = []
        
        # Load data from each file
        for i, file_path in enumerate(raw_data_files):
            try:
                with open(file_path, 'rb') as f:
                    raw_data = pickle.load(f)
                    
                if data_key not in raw_data['raw_data']:
                    print(f"Key {data_key} not found in {file_path}")
                    continue
                    
                data = raw_data['raw_data'][data_key][0][0][-1]
                data = data.reshape(-1, raw_data['metadata']['n'])[:,:int(np.floor(raw_data['metadata']['n']*raw_data['metadata']['constrained_subnetwork']))]
                
                # Flatten data if it's a list of arrays
                if isinstance(data, list):
                    data = np.concatenate(data).flatten()
                all_data.append(data.flatten())
            except Exception as e:
                print(f"Error loading data from {file_path}: {e}")
        
        if len(all_data) < 2:
            print(f"Failed to load data from at least two files for metric {metric}")
            continue
        
        # different bins for power and rates using linspace
        bins = np.linspace(0, 1, 25) if metric == 'all_Ps' else np.linspace(0, 12, 12*4)

        # Plot each experiment's data on current subplot
        for i, data in enumerate(all_data):
            if metric == 'all_Ps':
                data = data / raw_data['metadata']['P_max']
            axes[ax_idx].hist(data, alpha=0.5, bins=bins, label=experiment_labels[i])
        
        # Format labels and title for this subplot
        pretty_metric = 'Rate' if metric == 'all_rates' else 'Power'
        pretty_mode = {'unrolling': 'Primal-dual Unrolling'}.get(mode, mode)
        
        axes[ax_idx].set_xlabel(pretty_metric, fontsize=14)
        axes[ax_idx].set_ylabel('Frequency', fontsize=14)
        # axes[ax_idx].set_title(f'Histogram of {pretty_metric} for {pretty_mode}', fontsize=16)
        axes[ax_idx].grid(True, linestyle='--', alpha=0.7)
        
        # Only add legend to the first subplot to avoid redundancy
        if ax_idx == 0:
            axes[ax_idx].legend(fontsize=12)

    # Draw a line at r_min
    if 'all_rates' in metrics and 'r_min' in raw_data['metadata']:
        f_min = raw_data['metadata']['r_min']
        rate_idx = metrics.index('all_rates')
        axes[rate_idx].axvline(f_min, color='r', linestyle='--', linewidth=1.5, label=r'$r_{min}$')
        axes[rate_idx].legend(fontsize=12)
    
    plt.tight_layout()
    
    # Save the figure
    if save_path:
        R = raw_data['metadata']['R']
        os.makedirs(save_path, exist_ok=True)
        metrics_str = '_'.join([m.replace('all_', '') for m in metrics])
        filename = f'histogram_{metrics_str}_{f_min}_{R}.pdf'
        plt.savefig(os.path.join(save_path, filename))
        print(f"Created histogram: {filename}")
    else:
        plt.show()