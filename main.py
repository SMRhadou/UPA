import os
import random
import numpy as np
import json
import pickle
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm
import wandb
import uuid

import torch
from torch_geometric.loader import DataLoader

from core.data_gen import create_data
from core.channel import max_D_TxRx
from utils import WirelessDataset, simple_policy
from utils_plots import plotting_SA, plot_final_percentiles_comparison

from core.trainer import Trainer
from core.modules import PrimalModel, DualModel

RANDOM_SEED = 1357531

# set the random seed
os.environ['PYTHONHASHSEED']=str(RANDOM_SEED)
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def make_parser():
    # Argument parser
    parser = argparse.ArgumentParser(description='DA Unrolling')
    # parser.add_argument('--random_seed', type=int, default=1357531, help='Random seed')

    # experiment setup
    parser.add_argument('--m', type=int, default=100, help='Number of transmitters')
    parser.add_argument('--n', type=int, default=100, help='Number of receivers')
    parser.add_argument('--T', type=int, default=1, help='Number of time slots for each configuration')
    parser.add_argument('--R', type=int, default=2500, help='Size of the map')
    parser.add_argument('--density_mode', type=str, default='var_density', choices=['var_density', 'fixed_density'], help='Density mode')
    parser.add_argument('--BW', type=float, default=20e6, help='Bandwidth (Hz)')
    parser.add_argument('--P_maxdBm', type=float, default=0, help='Maximum transmit power (dBm)')
    parser.add_argument('--r_min', type=float, default=1.5, help='Minimum-rate constraint')
    parser.add_argument('--ss_param', type=float, default=1.0, help='Spread Spectrum parameter')
    parser.add_argument('--T_0', type=int, default=1, help='Size of the iteration window for averaging recent rates for dual variable updates')
    parser.add_argument('--metric', type=str, default='rates', choices=['rates', 'power'], help='Metric for rate calculation')

    # training parameters
    parser.add_argument('--training_modes', type=list, default=['primal'], help='Training modes for the model')
    parser.add_argument('--supervised', action='store_true', default=False, help='Supervised training')
    parser.add_argument('--num_samples_train', type=int, default=2048, help='Number of training samples')
    parser.add_argument('--num_samples_test', type=int, default=128, help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size/No. of graphs in a batch')
    parser.add_argument('--num_samplers', type=int, default=32, help='Number of samplers for the data loader')
    parser.add_argument('--num_epochs_primal', type=int, default=5000, help='Number of training epochs')
    parser.add_argument('--num_epochs_dual', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--num_iters', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--num_cycles', type=int, default=1, help='Number of training cycles')
    parser.add_argument('--lr_main', type=float, default=1e-8, help='Learning rate for primal model parameters')
    parser.add_argument('--lr_primal_multiplier', type=float, default=1e-5, help='Learning rate for Lagrangian multipliers in trainnig primal model')
    parser.add_argument('--lr_dual_main', type=float, default=1e-4, help='Learning rate for dual networks')
    parser.add_argument('--lr_dual_multiplier', type=float, default=1e-5, help='Learning rate for Lagrangian multipliers ion trainnig dual networks')
    parser.add_argument('--dual_resilient_decay', type=float, default=0.0, help='Resilient dual variables')
    parser.add_argument('--lr_DA_dual', type=float, default=1, help='Learning rate for dual variables in the DA algorithm')
    parser.add_argument('--training_resilient_decay', type=float, default=0.0, help='Learning rate for resilient dual variables')
    parser.add_argument('--thresh_resilient', type=float, default=2.5, help='Threshold for resilient dual variables')
    parser.add_argument('--evaluation_interval', type=int, default=500, help='Interval for evaluating the model')
    
    # architecture parameters
    parser.add_argument('--primal_k_hops', type=int, default=2, help='Number of hops in the GNN')
    parser.add_argument('--primal_hidden_size', type=int, default=256, help='Number of GNN features in different layers')
    parser.add_argument('--primal_num_sublayers', type=int, default=3, help='Number of primal sub-layers')
    parser.add_argument('--unrolled_primal', action='store_true', default=True, help='Unrolled primal model')
    parser.add_argument('--primal_num_blocks', type=int, default=4, help='Number of blocks in the primal model')


    parser.add_argument('--dual_k_hops', type=int, default=2, help='Number of hops in the GNN')
    parser.add_argument('--dual_hidden_size', type=list, default=256, help='Number of GNN features in different layers')
    parser.add_argument('--dual_num_sublayers', type=int, default=3, help='Number of dual sub-layers')
    parser.add_argument('--dual_num_blocks', type=int, default=4, help='Number of blocks in the dual model')

    parser.add_argument('--primal_norm_layer', type=str, default='layer', choices=['batch', 'layer', 'graph'], help='Normalization layer for the GNN')
    parser.add_argument('--dual_norm_layer', type=str, default='batch', choices=['batch', 'layer', 'graph'], help='Normalization layer for the dual model')
    parser.add_argument('--dropout_rate', type=float, default=0.2, help='Dropout rate')
    parser.add_argument('--conv_layer_normalize', type=bool, default=False, help='Convolutional layer normalization')
    # dual distribution for primal training
    parser.add_argument('--normalize_mu', action='store_true', default=False, help='Normalize the dual variables while training the primal model')
    parser.add_argument('--mu_max', type=int, default=5.0, help='maximum value of the dual variables in the training set of the primal model')
    parser.add_argument('--mu_distribution', type=str, default='uniform', choices=['uniform', 'exponential'], help='Distribution of the dual variables')
    # dual variable values for unrolling
    parser.add_argument('--mu_init', type=float, default=0.0, help='initial value of the dual variables in the training set')
    parser.add_argument('--mu_uncons', type=float, default=0.0, help='value of lambda bar of the unconstrained users')

    parser.add_argument('--zero_probability', type=float, default=0.2, help='Probability of zeroing out the dual variables')
    parser.add_argument('--all_zeros', action='store_true', default=True, help='Use all zeros for the dual variables')
    parser.add_argument('--constrained_subnetwork', type=float, default=0.5, help='impose constraints on part of the agents, 1 <==> full network')
    parser.add_argument('--architecture', type=str, default='PrimalGNN', help='Architecture of the model')
    parser.add_argument('--crop_p', type=float, default=1e-5, help='lowest primal power to bepredicted')

    # wandb parameters
    parser.add_argument('--use_wandb', action='store_true', default=True, help='Use Weights & Biases for logging')
    parser.add_argument('--wandb_project', type=str, default='PowerAllocation_PDUnrolling', help='W&B project name')
    parser.add_argument('--wandb_entity', type=str, default=None, help='W&B entity name')

    args = parser.parse_args()
    return args


def main(args):
    assert args.evaluation_interval <= args.num_epochs_primal, 'Evaluation interval must be less than the number of epochs for primal model'
    assert args.evaluation_interval <= args.num_epochs_dual, 'Evaluation interval must be less than the number of epochs for dual model'
    assert args.num_samplers == 1 or args.batch_size == 1, 'Batch size must be 1 for multiple samplers'

    num_samples = {'train': args.num_samples_train, 'test': args.num_samples_test}
    N = -174 - 30 + 10 * np.log10(args.BW)
    args.noise_var = np.power(10, N / 10)
    args.P_max = np.power(10, (args.P_maxdBm - 30) / 10)

    # set network area side length based on the density mode
    if args.density_mode == 'var_density':
        R = args.R
    elif args.density_mode == 'fixed_density':
        R = 1000 * np.sqrt(args.m / 20)
    else:
        raise Exception

    # create a string indicating the main experiment (hyper)parameters
    if args.constrained_subnetwork:
        experiment_name = 'subnetwork_m_{}_R_{}_Pmax_{}'.format(args.m, args.R, args.P_maxdBm)
    else:
        experiment_name = 'm_{}_R_{}_Pmax_{}'.format(args.m, args.R, args.P_maxdBm)

    
    # create folders to save the data, results, and final model
    os.makedirs('./data', exist_ok=True)

    # set the computation device and create the model using a GNN parameterization
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    primal_model = PrimalModel(args, device, unrolled=args.unrolled_primal) #normalize_mu)
    dual_model = DualModel(args, device)

    if args.training_modes[0] == 'dual':
        if args.unrolled_primal:
            experiment_path = './results/subnetwork_m_100_R_5000_Pmax_0_ss_1.0_resilience_0.0_depth_3_MUmax_60.0_rMin_1.5_lr_0.0001/7c3a5193' 
        else:
            if args. normalize_mu:
                experiment_path = './results/subnetwork_m_100_R_2500_Pmax_0_ss_1.0_resilience_0.0_depth_3_MUmax_10.0_rMin_2.0_lr_1e-06/bb3c2c94' #7fe6ab7b
            else:
                experiment_path = './results/m_100_R_2500_Pmax_0_ss_1.0_resilience_100.0_depth_3_MUmax_2.0_rMin_2.0_lr_1e-06/7fe6ab7b' #7fe6ab7b
                assert args.normalize_mu == False, 'Normalization of mu is not supported for dual training'
        checkpoint = torch.load('{}/best_primal_model.pt'.format(experiment_path), map_location='cpu')
        primal_model.load_state_dict(checkpoint['model_state_dict'])

    # create PyTorch Geometric datasets and dataloaders
    print('Generating the training and evaluation data ...')
    path = './data/{}_{}_train_{}'.format(experiment_name, max_D_TxRx, args.num_samples_train)   
    data_list = create_data(args.m, args.n, args.T, R, path, num_samples, args.P_max, args.noise_var)
    
    batch_size = {'train': args.batch_size, 'test': args.batch_size * args.num_samplers}
    loader = {}

    if len(args.training_modes) == 1 and args.training_modes[0] == 'primal':
        num_samples =  512
    else:
        num_samples = args.num_samples_train

    if not args.supervised:
        for phase in data_list:
            loader[phase] = DataLoader(WirelessDataset(data_list[phase][:num_samples]), batch_size=batch_size[phase], shuffle=(phase == 'train'))
    else:
        if os.path.exists('{}_target.json'.format(path)):
            data_list_supervised = torch.load('{}_target.json'.format(path), map_location='cpu')
        else: 
            data_list_supervised = defaultdict(list)
            for phase in batch_size.keys():
                for data in tqdm(data_list[phase]):
                    data = data.to(device)
                    target = dual_model.DA(primal_model, data, args.lr_DA_dual, 10.0, #args.dual_resilient_decay, 
                                            args.n, args.r_min, args.noise_var, 200, args.ss_param, device,
                                            adjust_constraints=False)
                    
                    data.target = torch.stack(target[0])[-1].unsqueeze(1)       # Mu*
                    data_list_supervised[phase].append(data)
            path = '{}_target.json'.format(path)
            torch.save(data_list_supervised, path)

        for phase in data_list_supervised:
            loader[phase] = DataLoader(WirelessDataset(data_list_supervised[phase][:num_samples]), batch_size=batch_size[phase], shuffle=(phase == 'train'))

    # create a string indicating the main experiment (hyper)parameters
    experiment_name += '_ss_{}_resilience_{}_depth_{}_MUmax_{}_rMin_{}_lr_{}'.format(args.ss_param,
                                                                                    args.dual_resilient_decay, 
                                                                                    args.dual_num_sublayers, 
                                                                                    args.mu_max,
                                                                                    args.r_min,
                                                                                    args.lr_main)

    # Create a unique run ID for wandb
    run_id = str(uuid.uuid4())[:8]
    wandb_run_name = f"{run_id}"
    
    # Initialize wandb if enabled
    if args.use_wandb:
        # Create a group name based on training modes
        group_name = f"{'_'.join(args.training_modes)}_training"
        
        # Configure wandb run with unique ID, tags, and group
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=vars(args),
            name=wandb_run_name,
            id=run_id,
            group=group_name,
        )
        wandb.run.log_code(".", include_fn=lambda path: path.endswith(".py"))
        
        # Log system info
        wandb.run.summary.update({
            "device": str(device),
            "experiment_name": experiment_name,
            "training_modes": args.training_modes,
        })

    # create folders for results and models
    os.makedirs('./results/{}/{}'.format(experiment_name, run_id), exist_ok=True)
    os.makedirs('./results/{}/{}/figs'.format(experiment_name, run_id), exist_ok=True)
    experiment_name += f'/{run_id}'

    #save args 
    with open('./results/{}/args.json'.format(experiment_name), 'w') as f:
        json.dump(vars(args), f, indent = 6)




    ############################ Modules, Optimizers and Trainer ############################

    
    optimizers = {'primal': torch.optim.Adam(primal_model.parameters(), lr=args.lr_main), #, weight_decay=1e-5), 
                  'dual': torch.optim.Adam(dual_model.parameters(), lr=args.lr_dual_main)}

    trainer = Trainer(primal_model, dual_model, loader['train'], optimizers, device, args)




    ############################# start training and evaluation #############################
    all_epoch_results = defaultdict(list)
    mu_test = args.mu_max * torch.rand(args.num_samplers * args.batch_size * args.m, 1).to(device)
    mu_test[-1] = torch.zeros_like(mu_test[-1])
    L_test = []
    best_loss = {mode: np.inf for mode in args.training_modes}

    num_epochs = {'primal': args.num_epochs_primal, 'dual': args.num_epochs_dual}
    print('Starting the training and evaluation process ...')
    for cycle in tqdm(range(args.num_cycles)):
        if cycle > 0:
            checkpoint = torch.load('./results/{}/best_primal_model.pt'.format(experiment_name), map_location='cpu')
            primal_model.load_state_dict(checkpoint['model_state_dict'])
            checkpoint = torch.load('./results/{}/best_dual_model.pt'.format(experiment_name), map_location='cpu')
            dual_model.load_state_dict(checkpoint['model_state_dict'])

        for mode in args.training_modes:
            num_blocks = args.primal_num_blocks if mode == 'primal' else args.dual_num_blocks
            training_multipliers = torch.zeros(num_blocks, 1).to(device)

            for epoch in tqdm(range(num_epochs[mode])):
                for phase in loader:

                    if phase == 'train':
                        train_loss, training_multipliers = trainer.train(epoch, training_multipliers, mode=mode)

                        #save best model
                        if train_loss < best_loss[mode]:
                            best_loss[mode] = train_loss
                            if mode == 'primal':
                                # Save best model with metadata for better tracking
                                checkpoint = {
                                    'model_state_dict': trainer.primal_model.state_dict(),
                                    'optimizer_state_dict': trainer.primal_optimizer.state_dict(),
                                    'loss': best_loss,
                                    'epoch': epoch,
                                }
                                torch.save(checkpoint, './results/{}/best_primal_model.pt'.format(experiment_name))
                                if args.use_wandb:
                                    wandb.run.summary.update({"best_primal_loss": best_loss, "best_primal_epoch": epoch})
                                    wandb.log({'best primal training loss': train_loss})
                            if mode == 'dual':
                                # Save best dual model with metadata
                                checkpoint = {
                                    'model_state_dict': trainer.dual_model.state_dict(),
                                    'optimizer_state_dict': trainer.dual_optimizer.state_dict(),
                                    'loss': best_loss,
                                    'epoch': epoch,
                                }
                                torch.save(checkpoint, './results/{}/best_dual_model.pt'.format(experiment_name))
                                if args.use_wandb:
                                    wandb.run.summary.update({"best_dual_loss": best_loss, "best_dual_epoch": epoch})
                                    wandb.log({'best dual training loss': train_loss})
                    
                    else:

                        if mode == 'primal':
                            L = trainer.primal_model.sanity_check(next(iter(loader[phase]))[0], mu_test, args.noise_var, args.ss_param)
                            L_test.append(L)

                        if mode == 'dual' and (epoch+1)%args.evaluation_interval == 0:
                            SA_results, unrolling_results, random_results, full_power_results = trainer.eval(loader[phase], num_iters=args.num_iters)

                            modes_list = ['SA', 'random', 'full_power', 'unrolling']
                            for test_mode in modes_list:
                                if mode == 'full_power':
                                    test_results = full_power_results
                                elif test_mode == 'random':
                                    test_results = random_results
                                elif test_mode == 'unrolling':
                                    test_results = unrolling_results
                                else:
                                    test_results = SA_results

                                all_epoch_results[test_mode, 'rate_mean'].append(np.stack(test_results['rate_mean']).mean())
                                for percentile in [5, 10, 15, 20, 30, 40, 50]:
                                    all_epoch_results[mode, f'rate_{percentile}th_percentile'].append(np.stack(test_results[f'rate_{percentile}th_percentile']).mean())
                                all_epoch_results[test_mode, 'all_rates'].append(test_results['all_rates'])
                                all_epoch_results[test_mode, 'all_Ps'].append(test_results['all_Ps'])
                                all_epoch_results[test_mode, 'violation_rate'].append(np.stack(test_results['violation_rate']).mean())
                                all_epoch_results[test_mode, 'mean_violation'].append(np.stack(test_results['mean_violation']).mean())
                            
                            all_epoch_results['SA', 'test_mu_over_time'].append(SA_results['test_mu_over_time'])
                            if mode == 'dual':
                                all_epoch_results['unrolling', 'test_mu_over_time'].append(unrolling_results['test_mu_over_time'])

                        

                            

        ############################# save the results and model ############################
        # save the results and overwrite the saved model with the current model 
        with open('./results/{}/results_dict.pkl'.format(experiment_name), 'wb') as f:
            pickle.dump(all_epoch_results, f)
        torch.save(trainer.primal_model.state_dict(), './results/{}/primal_model.pt'.format(experiment_name))
        torch.save(trainer.dual_model.state_dict(), './results/{}/dual_model.pt'.format(experiment_name))

    fig = plt.figure(figsize=(4, 5))
    plt.plot(np.stack(L_test, axis=1)[:10].T)
    plt.xlabel('epochs')
    plt.ylabel('Lagrangian')
    fig.savefig('./results/{}/figs/L_test.png'.format(experiment_name))


    # Plotting results
    plotting_SA(all_epoch_results, args.r_min, args.P_max, num_agents=args.n, num_iters=args.num_iters, unrolling_iters=args.dual_num_blocks+1, pathname='./results/{}/figs'.format(experiment_name))
    plot_final_percentiles_comparison(all_epoch_results, f_min=args.r_min, pathname='./results/{}/figs'.format(experiment_name))
    print('Training complete!')
    

if __name__ == '__main__':
    args = make_parser()
    main(args)

    