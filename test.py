import os
import numpy as np
import torch
from collections import defaultdict
from types import SimpleNamespace
from tqdm import tqdm
import argparse
import json
import random

from torch_geometric.loader import DataLoader

from core.data_gen import create_data
from core.gnn import GNN
from core.trainer import Trainer
from core.modules import PrimalModel, DualModel
from utils import calc_rates, WirelessDataset, simple_policy
from utils_plots import plot_testing, plot_final_percentiles_comparison, plotting_collective, plot_constrainedvsUnconsrained_histograms

NUM_EPOCHS = 800
max_D_TxRx = 60

RANDOM_SEED = 1357531

# set the random seed
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def main(experiment_path, sa_path=None, best=True, perturbation_ratio=None, R=None, r_min=None, n=None):

    all_epoch_results = defaultdict(list)
    # read args file as a dictionary
    with open('{}/args.json'.format(experiment_path), 'r') as f:
        args = json.load(f)
    args = SimpleNamespace(**args)

    if sa_path is not None:
        with open('{}/args.json'.format(sa_path), 'r') as f:
            sa_args = json.load(f)
        sa_args = SimpleNamespace(**sa_args)

        assert sa_args.unrolled_primal is False, 'SA args should not be unrolled'
        assert sa_args.training_modes == ['primal'], 'SA args should only have primal training mode'
        assert args.training_modes == ['primal', 'dual'], 'args should have primal and dual training modes'
        assert args.unrolled_primal is True, 'args should be unrolled primal'
        assert args.primal_num_sublayers* args.primal_num_blocks == sa_args.primal_num_sublayers, 'two models should share the same depth'
        # assert args.TxLoc_perturbation_ratio == sa_args.TxLoc_perturbation_ratio, 'two models should share the same perturbation ratio'
        assert args.graph_type == sa_args.graph_type, 'two models should share the same graph type'
        # assert args.sparse_graph_thresh == sa_args.sparse_graph_thresh, 'two models should share the same sparse graph threshold'
        assert args.R == sa_args.R, 'two models should share the same R'
        assert args.r_min == sa_args.r_min, 'two models should share the same r_min'
        assert args.constrained_subnetwork == sa_args.constrained_subnetwork, 'two models should share the same constrained subnetwork'
        assert args.noisy_training is True, 'args should have noisy training enabled'
        assert sa_args.noisy_training is False, 'SA args should not have noisy training enabled'
        assert sa_args.lambda_sampler == 'DA', 'SA args should use DA sampler'
        assert args.lambda_sampler == 'hybrid', 'args should use hybrid sampler'

    fix_mu_uncons = True
    args.use_wandb = False
    args.adjust_constraints = False
    args.sparse_graph_thresh = getattr(args, 'sparse_graph_thresh', 6e-2)
    args.TxLoc_perturbation_ratio = getattr(args, 'TxLoc_perturbation_ratio', 20)

    # load data
    num_samples = {'test': args.num_samples_test}
    if perturbation_ratio is not None:
        variable = perturbation_ratio
        data_path = './data/{}_{}_{}_perturbation_ratio_{}.json'.format(experiment_path.split('/')[-2][:30], args.graph_type, max_D_TxRx, perturbation_ratio)
        data_list = create_data(args.m, args.n, args.T, args.R, data_path, num_samples, args.P_max, args.noise_var, args.graph_type, 
                                args.sparse_graph_thresh, perturbation_ratio)
    elif R is not None:
        variable = R
        data_path = './data/{}_{}_{}_R_{}.json'.format(experiment_path.split('/')[-2][:30], args.graph_type, max_D_TxRx, R)
        data_list = create_data(args.m, args.n, args.T, R, data_path, num_samples, args.P_max, args.noise_var, args.graph_type, 
                                args.sparse_graph_thresh, perturbation_ratio=args.TxLoc_perturbation_ratio)
        
    elif n is not None:
        variable = n
        data_path = './data/{}_{}_{}_n_{}.json'.format(experiment_path.split('/')[-2][:30], args.graph_type, max_D_TxRx, n)
        data_list = create_data(n, n, args.T, args.R, data_path, num_samples, args.P_max, args.noise_var, args.graph_type, 
                                args.sparse_graph_thresh, perturbation_ratio=args.TxLoc_perturbation_ratio)
    else:
        variable = 0.0
        data_path = './data/{}_{}_{}_train_{}.json'.format(experiment_path.split('/')[-2][:30], args.graph_type, max_D_TxRx, args.num_samples_train)
        data_list = torch.load(data_path, map_location='cpu')
    loader = DataLoader(WirelessDataset(data_list['test']), batch_size=args.num_samples_test, shuffle=False)
    del data_list

    if r_min is not None:
        variable = r_min
        args.r_min = r_min
        if sa_path is not None:
            sa_args.r_min = r_min
    

    # load model from checkpoint
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    primal_model = PrimalModel(args, device, unrolled=args.unrolled_primal)
    dual_model = DualModel(args, device)
    if sa_path is not None:
        gnn_sa = PrimalModel(sa_args, device, unrolled=False)

    if sa_path is not None:         # SA model (benchmark)
        # if best:
        #     checkpoint = torch.load('{}/best_primal_model.pt'.format(sa_path), map_location='cpu')
        # else:
        checkpoint = torch.load('{}/best_primal_model.pt'.format(sa_path), map_location='cpu')
        gnn_sa.load_state_dict(checkpoint['model_state_dict'])
        executer1 = Trainer(primal_model=gnn_sa.to(device), dual_model=dual_model.to(device), args=args, device=device, dual_trained=False)
        SA_results, _, _, _ = executer1.eval(loader, num_iters=NUM_EPOCHS, adjust_constraints=args.adjust_constraints, fix_mu_uncons=fix_mu_uncons)
        del executer1

    if best:
        checkpoint = torch.load('{}/best_primal_model.pt'.format(experiment_path), map_location='cpu')
    else:
        checkpoint = torch.load('{}/primal_model.pt'.format(experiment_path), map_location='cpu')
    primal_model.load_state_dict(checkpoint['model_state_dict'])

    if best:
        checkpoint = torch.load('{}/best_dual_model.pt'.format(experiment_path), map_location='cpu')
    else:
        checkpoint = torch.load('{}/dual_model.pt'.format(experiment_path), map_location='cpu')
    dual_model.load_state_dict(checkpoint['model_state_dict'])
    dual_trained = True

    executer = Trainer(primal_model=primal_model.to(device), dual_model=dual_model.to(device), args=args, device=device, dual_trained=dual_trained)
    test_results, unrolling_results, random_results, full_power_results = executer.eval(loader, num_iters=NUM_EPOCHS, 
                                                                                        adjust_constraints=args.adjust_constraints, fix_mu_uncons=fix_mu_uncons)
    
    modes_list = ['random', 'full_power']
    if 'dual' in args.training_modes:
        assert unrolling_results is not None, 'unrolling results are None'
        assert test_results is not None, 'test results are None'
        modes_list.append('unrolling')
        modes_list.append('unrolledPrimal')
    if sa_path is not None:
        modes_list.append('SA')


    for mode in modes_list:
        if mode == 'unrolledPrimal':
            results = test_results 
        elif mode == 'random':
            results = random_results
        elif mode == 'full_power':
            results = full_power_results
        elif mode == 'unrolling':
            results = unrolling_results
        elif mode == 'SA':
            results = SA_results
        
        all_epoch_results[mode, 'rate_mean'].append(np.stack(results['rate_mean']).mean())
        for percentile in [5, 10, 15, 20, 30, 40, 50]:
            all_epoch_results[mode, f'rate_{percentile}th_percentile'].append(np.stack(results[f'rate_{percentile}th_percentile']).mean())
        all_epoch_results[mode, 'all_rates'].append(results['all_rates'])
        all_epoch_results[mode, 'all_Ps'].append(results['all_Ps'])
        all_epoch_results[mode, 'violation_rate'].append(np.stack(results['violation_rate']).mean())
        all_epoch_results[mode, 'mean_violation'].append(np.stack(results['mean_violation']).mean())
        all_epoch_results[mode, 'constrained_mean_rate'].append(np.stack(results['constrained_mean_rate']).mean())
        all_epoch_results[mode, 'unconstrained_mean_rate'].append(np.stack(results['unconstrained_mean_rate']).mean())
        
    if sa_path is not None:
        all_epoch_results['SA', 'test_mu_over_time'].append(test_results['test_mu_over_time'])
        all_epoch_results['SA', 'L_over_time'].append(test_results['L_over_time'])
    if 'dual' in args.training_modes:
        all_epoch_results['unrolledPrimal', 'test_mu_over_time'].append(test_results['test_mu_over_time'])
        all_epoch_results['unrolledPrimal', 'L_over_time'].append(test_results['L_over_time'])
        all_epoch_results['unrolling', 'test_mu_over_time'].append(unrolling_results['test_mu_over_time'])
        all_epoch_results['unrolling', 'dual_fn'].append(np.stack(unrolling_results['dual_fn']))

    plotting_collective(all_epoch_results, args.r_min, args.P_max, num_agents=args.n, num_iters=NUM_EPOCHS, unrolling_iters=args.dual_num_blocks+1, 
                pathname='{}/figs/{}_{}_'.format(experiment_path, variable, best), modes_list=modes_list)
    plot_final_percentiles_comparison(all_epoch_results, pathname='{}/figs/{}_{}_'.format(experiment_path, variable, best))
    if args.constrained_subnetwork < 1:
        plot_constrainedvsUnconsrained_histograms(all_epoch_results, int(np.floor(args.constrained_subnetwork*args.n)), args, 
                        pathname='{}/figs/{}_{}_'.format(experiment_path, variable, best), modes_list=modes_list)


    print('ok!')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DA Unrolling - Wireless Allocation Test')
    parser.add_argument('--experiment_path', type=str, default="subnetwork_m_100_R_2000_Pmax_0_regular_ss_1.0_resilience_0.0_depth_3_MUmax_1.0_rMin_1.5_lr_0.0001/703acd18")
    parser.add_argument('--sa_path', type=str, default=None) #"subnetwork_m_100_R_2000_Pmax_0_regular_ss_1.0_resilience_0.0_depth_3_MUmax_1.0_rMin_1.5_lr_0.0001/d4de8720")
    parser.add_argument('--best', action='store_true', default=False, help='use best model')
    parser.add_argument('--R', type=int, default=None, help='Size of the map')
    parser.add_argument('--r_min', type=float, default=None, help='Minimum-rate constraint')
    # parser.add_argument('--m', type=int, default=None, help='Number of transmitters')
    parser.add_argument('--n', type=int, default=None, help='Number of receivers')
    # parser.add_argument('--constrained_subnetwork', type=float, default=0.5, help='impose constraints on part of the agents, 1 <==> full network')
    # parser.add_argument('--graph_type', type=str, default='regular', choices=['CR', 'regular'], help='Type of graph to generate')
    # parser.add_argument('--sparse_graph_thresh', type=float, default=6e-2, help='Threshold for sparse graph generation')
    parser.add_argument('--TxLoc_perturbation_ratio', type=float, default=None, help='Perturbation ratio for transmitter locations')
    test_args = parser.parse_args()


    main('results/{}'.format(test_args.experiment_path), 'results/{}'.format(test_args.sa_path) if test_args.sa_path is not None else None,
            best=test_args.best, perturbation_ratio=test_args.TxLoc_perturbation_ratio,
            R=test_args.R, r_min=test_args.r_min, n=test_args.n)
    print('ok!')

