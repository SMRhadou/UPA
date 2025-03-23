import numpy as np
import matplotlib.pyplot as plt
plt.rc('font', family='serif', serif='cm10')
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')


def plot_testing(test_results, f_min, num_curves=20, pathname=None, num_agents=100, num_iters=400, batch_size=32):
    assert pathname is not None, 'Please provide a pathname to save the figures'
    fig, ax = plt.subplots(2, 3, figsize=(12, 6))
    multipliers = np.stack(test_results['test_mu_over_time'])
    for i in range(num_curves):
        ax[0, 0].plot(multipliers[0,:,i], label='agent {}'.format(i))
        ax[1, 0].plot(multipliers[1,:,i], label='agent {}'.format(i))
    ax[0, 0].set_xlabel('iterations')
    ax[1, 0].set_xlabel('iterations')
    ax[0, 0].set_title('Dual Variables')
    ax[1, 0].set_title('Dual Variables')
    
    power_allocated = np.stack(test_results['all_Ps'])
    for i in range(num_curves):
        ax[0,1].plot(power_allocated[0,:,i], label='agent {}'.format(i))
        ax[1,1].plot(power_allocated[1,:,i], label='agent {}'.format(i)) 
    ax[0,1].set_xlabel('iterations')
    ax[1,1].set_xlabel('iterations')
    # ax[1].legend()
    ax[0,1].set_title('Power allocated')
    ax[1,1].set_title('Power allocated')

    rates = np.stack(test_results['all_rates'])
    for i in range(num_curves):
        ax[0,2].plot(rates[0,:,i], label='agent {}'.format(i))
        ax[1,2].plot(rates[1,:,i], label='agent {}'.format(i))
    ax[0,2].plot(f_min*np.ones_like(rates[0,:,6]), ':k', linewidth=1, label=r'r_min')
    ax[1,2].plot(f_min*np.ones_like(rates[1,:,6]), ':k', linewidth=1, label=r'r_min')
    # ax[2].legend()
    ax[0,2].set_xlabel('iterations')
    ax[1,2].set_xlabel('iterations')
    ax[0,2].set_title('Rates per agent')
    ax[1,2].set_title('Rates per agent')
    fig.tight_layout()
    fig.savefig(f'{pathname}/SA_test_results.png')


    fig, ax = plt.subplots(1, 2, figsize=(8, 3))
    rate_mean = np.stack(test_results['all_rates']).reshape(-1, num_iters, batch_size, num_agents).sum(-1)
    for i in range(num_curves):
        ax[0].plot(rate_mean[0,:,i], label='graph {}'.format(i))
        ax[1].plot(rate_mean[1,:,i], label='graph {}'.format(i))
    ax[0].set_xlabel('iterations')
    ax[1].set_xlabel('iterations')
    # ax[0].legend()
    ax[0].set_title('Sum of rates per graph')
    ax[1].set_title('Sum of rates per graph')
    fig.tight_layout()
    fig.savefig(f'{pathname}/rates.png')
    

def plot_subnetworks(all_epoch_results, constrained_agents, P_max, n, pathname=None):
    modes_list = ['SA', 'unrolling']
    fig, ax = plt.subplots(2, 2, figsize=(8, 4))

    colors = {'SA': 'darkorange', 'unrolling': 'green'}
    alpha = 0.5  # Transparency level

    for i, mode in enumerate(modes_list):
        power_allocated = np.stack(all_epoch_results[mode, 'all_Ps'])[-1,:,-1].reshape(-1, n)
        rates = np.stack(all_epoch_results[mode, 'all_rates'])[-1,:,-1].reshape(-1, n)
        
        # Create histograms with transparent colors
        ax[0,0].hist((power_allocated[:,:constrained_agents]/P_max).reshape(-1,1), bins=20, 
                     alpha=alpha, color=colors[mode], label=mode)
        ax[0,1].hist(rates[:,:constrained_agents].reshape(-1,1), bins=20, 
                     alpha=alpha, color=colors[mode], label=mode)
        ax[1,0].hist((power_allocated[:,constrained_agents:]/P_max).reshape(-1,1), bins=20, 
                     alpha=alpha, color=colors[mode], label=mode)
        ax[1,1].hist(rates[:,constrained_agents:].reshape(-1,1), bins=20, 
                     alpha=alpha, color=colors[mode], label=mode)
    
    # Add labels and titles
    ax[0,0].set_xlabel(r'p/p_{max}')
    ax[0,0].set_ylabel('constrained agents')
    ax[0,1].set_xlabel('rates')
    ax[1,0].set_xlabel(r'p/p_{max}')
    ax[1,0].set_ylabel('unconstrained agents')
    ax[1,1].set_xlabel('rates')
    ax[0,0].set_title(r'Power allocated \% P_max')
    ax[0,1].set_title('Rates')
    
    # Add legend to each subplot
    for i in range(2):
        for j in range(2):
            ax[i,j].legend()

    fig.tight_layout()
    fig.savefig(f'{pathname}/constrained_agents_comparison.png')



def plotting_SA(all_epoch_results, f_min, P_max, num_curves=15, num_agents=100, num_iters=800, unrolling_iters=6, pathname=None):
    assert pathname is not None, 'Please provide a pathname to save the figures'

    # Define consistent colors for modes
    colors = {'SA': 'darkorange', 'unrolling': 'green', 'random': 'lightblue', 'full_power': 'lightgray'}
    
    modes_list = ['SA', 'unrolling']
    layer = -1
    for mode in modes_list:
        for graph in range(3):
            fig, ax = plt.subplots(2, 3, figsize=(12, 6))
            multipliers = np.stack(all_epoch_results[mode, 'test_mu_over_time'])
            for i in range(num_curves):
                ax[0,0].plot(multipliers[-1,0,:,i+graph*num_agents], label='agent {}'.format(i))
                ax[1,0].plot(multipliers[-1,1,:,i+graph*num_agents], label='agent {}'.format(i))
            ax[0,0].set_xlabel('iterations')
            ax[1,0].set_xlabel('iterations')
            ax[0,0].set_title('Dual variables')
            ax[1,0].set_title('Dual variables')
            
            power_allocated = np.stack(all_epoch_results[mode, 'all_Ps'])
            ax[0,1].hist(power_allocated[-1,0,layer,graph*num_agents:(1+graph)*num_agents]/P_max, 
                         bins=20, color=colors[mode], alpha=0.7)
            ax[1,1].hist(power_allocated[-1,1,layer,+graph*num_agents:(1+graph)*num_agents]/P_max, 
                         bins=20, color=colors[mode], alpha=0.7) 
            ax[0,1].set_xlabel('p')
            ax[1,1].set_xlabel('p')
            ax[0,1].set_title('Power allocated % P_max')
            ax[1,1].set_title('Power allocated % P_max')

            rates = np.stack(all_epoch_results[mode, 'all_rates'])
            iters = unrolling_iters if mode == 'unrolling' else num_iters
            start = max(0, num_iters-1000)
            for i in range(num_curves):
                ax[0,2].plot(np.arange(start, iters), rates[-1,0,start:,i+graph*num_agents], label='agent {}'.format(i))
                ax[1,2].plot(np.arange(start, iters), rates[-1,1,start:,i+graph*num_agents], label='agent {}'.format(i))
            ax[0,2].plot(np.arange(start, iters), f_min*np.ones_like(rates[-1,0,start:,6]), ':k', linewidth=1, label=r'r_min')
            ax[1,2].plot(np.arange(start, iters), f_min*np.ones_like(rates[-1,1,start:,6]), ':k', linewidth=1, label=r'r_min')
            ax[0,2].set_xlim(start, iters)
            ax[1,2].set_xlim(start, iters)
            ax[0,2].set_xlabel('rate')
            ax[1,2].set_xlabel('rate')
            ax[0,2].set_title('Rates')
            ax[1,2].set_title('Rates')
            fig.tight_layout()
            fig.savefig(f'{pathname}/SA_results_{mode}_{graph}.png')


        fig, ax = plt.subplots(2, 3, figsize=(15, 9))
        num_epochs = len(all_epoch_results[mode, 'all_rates'])
        iters = num_iters if mode == 'SA' else unrolling_iters
        rate_mean = np.stack(all_epoch_results[mode, 'all_rates']).reshape(num_epochs, 4, iters, -1, num_agents).sum(-1)
        for i in range(1):
            for j in range(num_curves):
                ax[i//3, i%3].plot(rate_mean[i,0,:,j], label='graph {}'.format(j))
            ax[i//3, i%3].set_xlabel('iterations')
            ax[i//3, i%3].set_ylabel('mean rate')
            ax[i//3, i%3].set_title(f'training epoch {200*(i+1)}')
            ax[i//3, i%3].legend()
        fig.tight_layout()
        fig.savefig(f'{pathname}/SA_mean_rates_{mode}.png')

    fig, ax = plt.subplots(1, 2, figsize=(6, 3))
    for i, mode in enumerate(modes_list):
        iters = num_iters if mode == 'SA' else unrolling_iters
        temp_rates = np.stack(all_epoch_results[mode, 'all_rates'])[:,:,0]
        violation = [np.abs(np.minimum(np.stack(all_epoch_results[mode, 'all_rates'])[:,:,i] - f_min, np.zeros_like(temp_rates))).mean().item() for i in range(iters)]
        ax[i].plot(violation, color=colors[mode], linewidth=2)
        ax[i].set_xlabel('layers')
        ax[i].set_ylabel('mean violation')
        ax[i].set_title(f'{mode}')
    fig.tight_layout()
    fig.savefig(f'{pathname}/SA_mean_violation.png')



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
    fig, ax = plt.subplots(figsize=(6, 4))
    
    # Prepare metrics to plot
    metrics = [
        ('rate_mean', 'Obj'),
        ('constrained_mean_rate', 'cons'),
        # ('rate_1th_percentile', '1%'),
        ('rate_5th_percentile', '5%'),
        ('rate_10th_percentile', '10%'),
        ('rate_15th_percentile', '15%'),
        ('rate_20th_percentile', '20%'),
        ('rate_30th_percentile', '30%'),
        ('rate_40th_percentile', '40%'),
        ('rate_50th_percentile', '50%')
    ]
    
    # Prepare data for all modes
    all_metric_values = []
    # Switched the order of 'random' and 'full_power'
    mode_labels = ['SA', 'unrolling', 'full_power', 'random']
    
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
        bar = ax.bar(pos, values, width=bar_width, color=colors[mode], alpha=alphas[mode], label=mode)
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
    ax.set_ylabel('Rate (bits/s/Hz/agent)')
    ax.set_title('Violation Percentiles Comparison')
    ax.set_xticks([r + 2*bar_width for r in r1.tolist()])
    ax.set_xticklabels([label for _, label in metrics])
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout and save
    fig.tight_layout()
    fig.savefig(f'{pathname}/final_percentiles_comparison.png')
    

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


