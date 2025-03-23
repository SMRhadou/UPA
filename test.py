import os
import numpy as np
import torch
from collections import defaultdict
from types import SimpleNamespace
from tqdm import tqdm
import pickle
import json
import random

from torch_geometric.loader import DataLoader

from core.data_gen import create_data
from core.gnn import GNN
from core.trainer import Trainer
from core.modules import PrimalModel, DualModel
from utils import calc_rates, WirelessDataset, simple_policy
from utils_plots import plot_testing, plot_final_percentiles_comparison, plotting_SA, plot_subnetworks

NUM_EPOCHS = 200
max_D_TxRx = 60

RANDOM_SEED = 1357531

# set the random seed
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def main(R):
    experiment_path = './results/subnetwork_m_100_R_{}_Pmax_0_ss_1.0_resilience_0.0_depth_3_MUmax_10.0_rMin_2.0_lr_1e-06/'.format(R)
    experiment_path += '0ebee525' if R == 2500 else 'c8e72391' # 1000 or 2000
    # c820fc50 (2), 7841d161 (3)

    all_epoch_results = defaultdict(list)
    # with open('{}/results_dict.pkl'.format(experiment_path), 'rb') as f:
    #     all_epoch_results.update(pickle.load(f))
    # plotting_SA(all_epoch_results, f_min=1.5, pathname=experiment_path+'/figs')

    # read args file as adictionary
    with open('{}/args.json'.format(experiment_path), 'r') as f:
        args = json.load(f)
    args = SimpleNamespace(**args)
    args.lr_DA_dual = 1                 #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    args.dual_resilient_decay = 10.0  #<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    args.use_wandb = False
    args.adjust_constraints = False
 
    # args.constrained_subnetwork = 0.5
    args.normalize_mu = getattr(args, 'normalize_mu', False)
    # args.primal_hidden_size = 256
    # args.primal_num_sublayers = 3
    # args.primal_k_hops = 2
    # args.dual_hidden_size = 256
    # args.dual_num_sublayers = 3
    # args.dual_k_hops = 2

    # load data
    # data_path = './data/m_100_R_2500_Pmax_0_60.json'
    data_path = './data/{}_{}_train_{}_target.json'.format(experiment_path.split('/')[-2][:30], max_D_TxRx, args.num_samples_train)
    data_list = torch.load(data_path, map_location='cpu')
    loader = DataLoader(WirelessDataset(data_list['test']), batch_size=32, shuffle=False)
    del data_list

    # load model from checkpoint
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    primal_model = PrimalModel(args, device)
    dual_model = DualModel(args, device)
    if args.training_modes[0] == 'dual':
        if args. normalize_mu:
            primal_experiment_path = './results/subnetwork_m_100_R_2500_Pmax_0_ss_1.0_resilience_0.0_depth_3_MUmax_10.0_rMin_2.0_lr_1e-06/bb3c2c94' #7fe6ab7b
        else:
            primal_experiment_path = './results/m_100_R_2500_Pmax_0_ss_1.0_resilience_100.0_depth_3_MUmax_2.0_rMin_2.0_lr_1e-06/7fe6ab7b' #7fe6ab7b
        # primal_experiment_path = './results/subnetwork_m_100_R_2500_Pmax_0_ss_1.0_resilience_0.0_depth_3_MUmax_10.0_rMin_2.0_lr_1e-06/bb3c2c94'
        checkpoint = torch.load('{}/best_primal_model.pt'.format(primal_experiment_path))
        primal_model.load_state_dict(checkpoint['model_state_dict'])

        checkpoint = torch.load('{}/best_dual_model.pt'.format(experiment_path), map_location='cpu')
        dual_model.load_state_dict(checkpoint['model_state_dict'])
        dual_trained = True
    else:
        checkpoint = torch.load('{}/best_primal_model.pt'.format(experiment_path), map_location='cpu')
        primal_model.load_state_dict(checkpoint['model_state_dict'])
        dual_trained = False
   

    executer = Trainer(primal_model=primal_model.to(device), dual_model=dual_model.to(device), args=args, device=device, dual_trained=dual_trained)
    test_results, unrolling_results, random_results, full_power_results = executer.eval(loader, num_iters=NUM_EPOCHS, adjust_constraints=args.adjust_constraints)
    modes_list = ['SA', 'random', 'full_power']
    if args.training_modes[0] == 'dual':
        assert unrolling_results is not None, 'unrolling results are None'
        modes_list.append('unrolling')


    for mode in modes_list:
        if mode == 'SA':
            results = test_results 
        elif mode == 'random':
            results = random_results
        elif mode == 'full_power':
            results = full_power_results
        elif mode == 'unrolling':
            results = unrolling_results
        

        all_epoch_results[mode, 'rate_mean'].append(np.stack(results['rate_mean']).mean())
        for percentile in [5, 10, 15, 20, 30, 40, 50]:
            all_epoch_results[mode, f'rate_{percentile}th_percentile'].append(np.stack(results[f'rate_{percentile}th_percentile']).mean())
        all_epoch_results[mode, 'all_rates'].append(results['all_rates'])
        all_epoch_results[mode, 'all_Ps'].append(results['all_Ps'])
        all_epoch_results[mode, 'violation_rate'].append(np.stack(results['violation_rate']).mean())
        all_epoch_results[mode, 'mean_violation'].append(np.stack(results['mean_violation']).mean())
        all_epoch_results[mode, 'constrained_mean_rate'].append(np.stack(results['constrained_mean_rate']).mean())

    all_epoch_results['SA', 'test_mu_over_time'].append(test_results['test_mu_over_time'])
    if args.training_modes[0] == 'dual':
        all_epoch_results['unrolling', 'test_mu_over_time'].append(unrolling_results['test_mu_over_time'])

    plotting_SA(all_epoch_results, args.r_min, args.P_max, num_agents=args.n, num_iters=NUM_EPOCHS, unrolling_iters=args.num_blocks+1, pathname='{}/figs'.format(experiment_path))
    plot_final_percentiles_comparison(all_epoch_results, pathname='{}/figs'.format(experiment_path))
    if args.constrained_subnetwork < 1:
        plot_subnetworks(all_epoch_results, int(np.floor(args.constrained_subnetwork*args.n)) , args.P_max, args.n, pathname='{}/figs'.format(experiment_path))


    print('ok!')



if __name__ == '__main__':
    # Rs = [1000, 2000, 3000, 4000, 5000]
    # for R in Rs:
    #     main(int(R))
    main(int(2500))
    print('ok!')

