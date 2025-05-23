import torch
import numpy as np
from utils import calc_rates
import wandb

from functorch import make_functional, vmap
from torch.func import functional_call, jacrev

from collections import defaultdict
from torch_geometric.transforms import GDC



class Trainer():
    def __init__(self, primal_model, dual_model, optimizers=None, device=None, args=None, **kwargs):
        self.primal_model = primal_model
        self.dual_model = dual_model
        self.primal_optimizer = optimizers['primal'] if optimizers is not None else None
        self.dual_optimizer = optimizers['dual'] if optimizers is not None else None
        self.device = device
        # N = -174 - 30 + 10 * np.log10(args.BW)
        self.noise_var = args.noise_var #np.power(10, N / 10)
        self.args = args
        self.dual_trained = kwargs.get('dual_trained', False)
        self.multipliers_table = []
        self.mu_uncons = args.mu_uncons
        


    def multiplier_sampler(self, num_samplers, mu_max, *dist_param, dist='uniform', all_zeros=True):
        if dist == 'uniform':
            assert len(dist_param) == 1
            mu_initial = mu_max * torch.rand(num_samplers, 1)
        elif dist == 'exponential':
            rate = -torch.log(1 - torch.tensor(0.96)) / (1.5*mu_max)
            dist = torch.distributions.exponential.Exponential(rate=rate)
            mu_initial = dist.sample(sample_shape=(num_samplers, 1))
        else:
            raise NotImplementedError
        
        zero_probability = dist_param[0] if len(dist_param) > 0 else 0.2
        if all_zeros:
            mu_initial = mu_initial.view(-1, self.args.n)  # Ensure mu_initial is 2D
            zero_ids = torch.rand(mu_initial.shape[0], 1) < zero_probability
            zero_ids = zero_ids.repeat(1, mu_initial.shape[1])
            mu = torch.where(zero_ids, torch.zeros_like(mu_initial), mu_initial)
        else:
            zero_ids = torch.rand_like(mu_initial) < zero_probability
            mu = torch.where(zero_ids, torch.zeros_like(mu_initial), mu_initial)

        if self.multipliers_table != []:
            mu = mu.view(-1, self.args.n)
            indices = torch.randint(0, torch.stack(self.multipliers_table).view(-1, self.args.n).shape[0], (mu.shape[0]//2,))
            mu = torch.cat((
                mu[:mu.shape[0]//2].to(self.device),
                torch.stack(self.multipliers_table).view(-1, self.args.n)[indices].to(self.device)
            ), dim=0)

        if self.args.constrained_subnetwork < 1:
            mu = mu.view(-1, self.args.n)
            mu = torch.cat((mu[:, :int(np.floor(self.args.constrained_subnetwork*self.args.m))], 
                            self.mu_uncons * torch.ones(mu.shape[0], int(np.ceil((1-self.args.constrained_subnetwork)*self.args.m)))), dim=1)
        
        return mu.view(-1, 1)


    def unroll_DA(self, data=None, edge_index_l=None, edge_weight_l=None, a_l=None, transmitters_index=None, num_graphs=None, 
                  mode='dual', noisy_training=False):
        if data is not None:
            data = data.to(self.device)
            edge_index_l, edge_weight_l, a_l, transmitters_index, num_graphs = \
                data.edge_index_l, data.edge_weight_l, data.weighted_adjacency_l, \
                data.transmitters_index, data.num_graphs
            
            edge_index_l, edge_weight_l = GDC().sparsify_sparse(edge_index=edge_index_l, edge_weight=edge_weight_l,
                                                                    num_nodes=self.args.n, method="threshold", eps=2e-2)
            
        if hasattr(self.primal_model, 'cons_lvl'):
            delattr(self.primal_model, 'cons_lvl')
        if hasattr(self.dual_model, 'cons_lvl'):
            delattr(self.dual_model, 'cons_lvl')

        mu = torch.cat((self.args.mu_init * torch.rand(num_graphs, int(np.floor(self.args.constrained_subnetwork*self.args.m))).to(self.device), 
                        self.mu_uncons * torch.ones(num_graphs, int(np.ceil((1-self.args.constrained_subnetwork)*self.args.m))).to(self.device)), dim=1)
        mu = mu.view(num_graphs * self.args.n, 1)

        gamma = torch.ones(num_graphs * self.args.n, 1).to(self.device)
        outputs_list = []
        for block_id in range(self.dual_model.dual_num_blocks):
            p = self.primal_model(mu, edge_index_l, edge_weight_l, transmitters_index)
            if isinstance(p, list):
                p = p[-1]  # Use the last primal output if p is a list
            rates = calc_rates(p, gamma, a_l[:, :, :], self.noise_var, self.args.ss_param)
            L, _ = self.primal_model.loss(rates, mu, p, metric='power' if self.primal_model.args.metric == 'power' else 'rates',
                                       constrained_subnetwork=None)         # -1 * dual_function
            outputs_list.append((mu, p, rates, -1*L/ self.args.n))
            mu = mu + self.dual_model(block_id, mu, rates, edge_index_l, edge_weight_l, transmitters_index)
            if noisy_training and block_id < self.dual_model.dual_num_blocks - 1:
                noise_var = mu.detach().max() * np.exp(-1*(block_id+1))
                rand_noise = torch.cat((noise_var * torch.randn(num_graphs, int(np.floor(self.args.constrained_subnetwork*self.args.m))).to(self.device), 
                        torch.zeros(num_graphs, int(np.ceil((1-self.args.constrained_subnetwork)*self.args.m))).to(self.device)), dim=1)
                mu += rand_noise.view(-1,1).detach()
            mu = torch.relu(mu)
        
        # Primal recovery
        p = self.primal_model(mu, edge_index_l, edge_weight_l, transmitters_index)
        if isinstance(p, list):
            p = p[-1]
        rates = calc_rates(p, gamma, a_l[:, :, :], self.noise_var, self.args.ss_param)
        L, _ = self.primal_model.loss(rates, mu, p, metric='power' if self.primal_model.args.metric == 'power' else 'rates',
                                   constrained_subnetwork=None) 
        outputs_list.append((mu, p, rates, -1*L/self.args.n))

        if self.args.use_wandb and mode == 'dual':
            wandb.log({'dual/dual lagrangian loss': L.item()/self.args.n})
            wandb.log({'dual/dual function': -1*L.item()/self.args.n})
            wandb.log({'dual/max_lambda': torch.max(mu.detach().view(-1, self.args.n)[:,:int(np.floor(self.args.constrained_subnetwork*self.args.n)) ].cpu()).item()})
            wandb.log({'dual/70th_percentile_lambda': torch.quantile(mu.detach().view(-1, self.args.n)[:,:int(np.floor(self.args.constrained_subnetwork*self.args.n)) ], 0.7).item()})
            wandb.log({'dual/30th_percentile_lambda': torch.quantile(mu.detach().view(-1, self.args.n)[:,:int(np.floor(self.args.constrained_subnetwork*self.args.n)) ], 0.3).item()})
            wandb.log({'dual/mean rate': torch.mean(rates.view(num_graphs, self.args.n).mean(1).detach().cpu()).item()})
            wandb.log({'dual/min rate': torch.min(rates.detach().cpu()).item()})
            wandb.log({'dual/mean constrained rate': torch.mean(rates.view(num_graphs, self.args.n)[:, :int(np.floor(self.args.constrained_subnetwork*self.args.n))].mean(1).detach().cpu()).item()})
            wandb.log({'dual/mean unconstrained rate': torch.mean(rates.view(num_graphs, self.args.n)[:, int(np.floor(self.args.constrained_subnetwork*self.args.n)):].mean(1).detach().cpu()).item()})
            wandb.log({'dual/30th_percentile_rate': torch.quantile(rates.detach().cpu(), 0.3).item()})
            wandb.log({'dual/70th_percentile_rate': torch.quantile(rates.detach().cpu(), 0.7).item()})
            wandb.log({'dual/max rate': torch.max(rates.detach().cpu()).item()})
            wandb.log({'dual/min P': torch.min(p.detach().cpu()).item()})
            wandb.log({'dual/30th_percentile_P': torch.quantile(p.detach().cpu(), 0.3).item()})
            wandb.log({'dual/70th_percentile_P': torch.quantile(p.detach().cpu(), 0.7).item()})
            wandb.log({'dual/max P': torch.max(p.detach().cpu()).item()})

        return outputs_list



    def train(self, epoch, loader, training_multipliers, mode='primal'):
        assert self.primal_optimizer is not None, 'Primal optimizer is not defined'
        assert self.dual_optimizer is not None, 'Dual optimizer is not defined'
        assert self.primal_model is not None, 'Primal model is not defined'
        assert self.dual_model is not None, 'Dual model is not defined'
        assert self.device is not None, 'Device is not defined'
        assert self.args is not None, 'Args are not defined'

        num_samplers = self.args.num_samplers if mode=='primal' else 1
        # initialize the training multipliers
        
        for data, batch_idx in loader:
            self.primal_model.zero_grad()
            data = data.to(self.device)
            y, edge_index_l, edge_weight_l, _, \
            _, _, a_l, transmitters_index, num_graphs = \
                data.y, data.edge_index_l, data.edge_weight_l, data.edge_index, data.edge_weight, \
                data.weighted_adjacency, data.weighted_adjacency_l, \
                data.transmitters_index, data.num_graphs
            
            edge_index_l, edge_weight_l = GDC().sparsify_sparse(edge_index=edge_index_l, edge_weight=edge_weight_l,
                                                                    num_nodes=self.args.n, method="threshold", eps=2e-2)
            
            loss_list = []
            if num_samplers > 1:
                # Create num_samplers copies of the graph
                edge_index_expanded = []
                edge_weight_expanded = []
                for s in range(num_samplers):
                    offset = s * self.args.n
                    curr_edge_index = edge_index_l.clone()
                    curr_edge_index += offset
                    edge_index_expanded.append(curr_edge_index)
                    edge_weight_expanded.append(edge_weight_l)

                edge_index_l = torch.cat(edge_index_expanded, dim=1)
                edge_weight_l = torch.cat(edge_weight_expanded)
                a_l = a_l.repeat(num_samplers, 1, 1)
                transmitters_index = torch.arange(num_samplers*self.args.n).to(self.device)

            if mode == 'primal':
                self.primal_model.train()
                self.dual_model.eval()
                
                # Sample the multipliers
                if hasattr(self.primal_model, 'cons_lvl'):
                    delattr(self.primal_model, 'cons_lvl')
                if self.args.lambda_sampler == 'DA':
                    mu_over_time, _, _, _, _ = self.dual_model.DA(self.primal_model, data, self.args.lr_DA_dual, self.args.dual_resilient_decay, 
                                                                    self.args.n, self.args.r_min, self.noise_var, self.args.num_iters, self.args.ss_param, 
                                                                    self.args.mu_init, self.mu_uncons, self.device, False, True, num_samplers=10)
                    mu_over_time = torch.stack(mu_over_time).reshape(-1, self.args.n)
                    # Choose 32 random rows from mu_over_time
                    selected_indices = torch.randperm(mu_over_time.shape[0])[:num_samplers]
                    mu = mu_over_time[selected_indices].view(-1,1).to(self.device)

                elif self.args.lambda_sampler == 'hybrid':
                    mu_over_time, _, _, _, _ = self.dual_model.DA(self.primal_model, data, self.args.lr_DA_dual, self.args.dual_resilient_decay, 
                                                                    self.args.n, self.args.r_min, self.noise_var, self.args.num_iters, self.args.ss_param, 
                                                                    self.args.mu_init, self.mu_uncons, self.device, False, True, num_samplers=10)
                    mu_over_time = torch.stack(mu_over_time).reshape(-1, self.args.n)
                    # Choose 32 random rows from mu_over_time
                    selected_indices = torch.randperm(mu_over_time.shape[0])[:num_samplers//4]
                    mu = mu_over_time[selected_indices].view(-1,1).to(self.device)
                
                    assert len(self.args.training_modes) == 2
                    outputs_list = self.unroll_DA(edge_index_l=edge_index_l, edge_weight_l=edge_weight_l, a_l=a_l, transmitters_index=transmitters_index, 
                                                  num_graphs=num_graphs*num_samplers, mode=mode)
                    mu_over_time = torch.cat([outputs_list[i][0].view(num_graphs*num_samplers, self.args.n).detach()
                                    for i in range(len(outputs_list))], dim=0)
                    selected_indices = torch.randperm(mu_over_time.shape[0])[:num_samplers//2]
                    mu = torch.cat((mu, mu_over_time[selected_indices].view(-1,1).to(self.device)), dim=0)
                    del outputs_list

                    random_elements = num_graphs*num_samplers*self.args.n - mu.shape[0]
                    mu_random = self.multiplier_sampler(random_elements, self.args.mu_max, self.args.zero_probability, 
                             dist=self.args.mu_distribution, all_zeros=self.args.all_zeros).to(self.device)
                    mu = torch.cat((mu, mu_random.view(-1, 1)), dim=0)

                elif self.args.lambda_sampler == 'random':
                    mu = self.multiplier_sampler(num_graphs*num_samplers*self.args.n, self.args.mu_max, self.args.zero_probability, 
                             dist=self.args.mu_distribution, all_zeros=self.args.all_zeros).to(self.device)

                mu = mu.detach()
                rand_noise = torch.cat((1 + 0.1 * (torch.rand(num_samplers, int(np.floor(self.args.constrained_subnetwork*self.args.m)))-0.5).to(self.device), 
                        torch.zeros(num_samplers, int(np.ceil((1-self.args.constrained_subnetwork)*self.args.m))).to(self.device)), dim=1)
                mu *= rand_noise.view(-1, 1)
                

                if hasattr(self.primal_model, 'cons_lvl'):
                    delattr(self.primal_model, 'cons_lvl')

                
                
                # Forward pass and Losses
                p = self.primal_model(mu, edge_index_l, edge_weight_l, transmitters_index, noisy_training=self.args.noisy_training)    # MU is normalized to [0, 1]
                gamma = torch.ones(num_samplers*num_graphs * self.args.n, 1).to(self.device) 
                # if p is not a list
                if not isinstance(p, list):
                    rates = calc_rates(p, gamma, a_l[:, :, :], self.noise_var, self.args.ss_param)
                else:
                    rates = []
                    for i in range(len(p)):
                        rates.append(calc_rates(p[i], gamma, a_l[:, :, :], self.noise_var, self.args.ss_param))

                loss, constraints_loss = self.primal_model.loss(rates, mu, p, constraint_eps=self.args.primal_constraint_eps, 
                                                                metric=self.args.metric, constrained_subnetwork=None)
                L = loss + torch.sum(training_multipliers * constraints_loss)

                # primal update
                self.primal_optimizer.zero_grad()
                L.backward()
                self.primal_optimizer.step()

                # dual update
                if self.args.training_resilient_decay > 0:
                    training_multipliers = (1-1/self.args.training_resilient_decay) * training_multipliers + self.args.lr_primal_multiplier * constraints_loss.detach()
                    multiplier_update = True
                else:
                    training_multipliers = training_multipliers + self.args.lr_primal_multiplier * constraints_loss.detach()
                    multiplier_update = True
                training_multipliers = torch.maximum(training_multipliers, torch.zeros_like(training_multipliers))

                loss_list.append((loss+torch.maximum(constraints_loss, torch.zeros_like(constraints_loss)).sum()).item())

                if isinstance(p, list):
                    for i in range(len(rates)):
                        wandb.log({'primal/mean rate at primal layer {}'.format(i): torch.mean(rates[i].view(num_graphs*num_samplers, self.args.n).mean(1).detach().cpu()).item()})
                        if self.args.constrained_subnetwork < 1:
                            wandb.log({'primal/mean constrained rate at primal layer {}'.format(i): torch.mean(rates[i].view(num_graphs*num_samplers, self.args.n)[:, :int(np.floor(self.args.constrained_subnetwork*self.args.n))].mean(1).detach().cpu()).item()})
                            wandb.log({'primal/mean unconstrained rate at primal layer {}'.format(i): torch.mean(rates[i].view(num_graphs*num_samplers, self.args.n)[:, int(np.floor(self.args.constrained_subnetwork*self.args.n)):].mean(1).detach().cpu()).item()})
                    rates, p = rates[-1], p[-1]

                if self.args.use_wandb:
                    wandb.log({'primal/primal lagrangian loss': -1*loss.item()/self.args.n})
                    wandb.log({'primal/primal training loss': L.item()})
                    wandb.log({'primal/mean rate': torch.mean(rates.view(num_graphs*num_samplers, self.args.n).mean(1).detach().cpu()).item()})
                    wandb.log({'primal/min rate': torch.min(rates.detach().cpu()).item()})
                    wandb.log({'primal/mean constrained rate': torch.mean(rates.view(num_graphs*num_samplers, self.args.n)[:, :int(np.floor(self.args.constrained_subnetwork*self.args.n))].mean(1).detach().cpu()).item()})
                    wandb.log({'primal/mean unconstrained rate': torch.mean(rates.view(num_graphs*num_samplers, self.args.n)[:, int(np.floor(self.args.constrained_subnetwork*self.args.n)):].mean(1).detach().cpu()).item()})
                    wandb.log({'primal/30th_percentile_rate': torch.quantile(rates.detach().cpu(), 0.3).item()})
                    wandb.log({'primal/70th_percentile_rate': torch.quantile(rates.detach().cpu(), 0.7).item()})
                    wandb.log({'primal/max rate': torch.max(rates.detach().cpu()).item()})
                    wandb.log({'primal/min P': torch.min(p.detach().cpu()).item()})
                    wandb.log({'primal/30th_percentile_P': torch.quantile(p.detach().cpu(), 0.3).item()})
                    wandb.log({'primal/70th_percentile_P': torch.quantile(p.detach().cpu(), 0.7).item()})
                    wandb.log({'primal/max P': torch.max(p.detach().cpu()).item()})
                    if self.primal_model.unrolled:
                        wandb.log({'primal/primal constraint loss': torch.mean(constraints_loss.detach().cpu()).item()})
                
                del mu

            else:
                self.dual_model.train()
                self.primal_model.eval()

                # if epoch == 0:
                #     self.multipliers_table = []
                
                # Forward pass
                outputs_list = self.unroll_DA(edge_index_l=edge_index_l, edge_weight_l=edge_weight_l, a_l=a_l, transmitters_index=transmitters_index, 
                                              num_graphs=num_graphs*num_samplers, noisy_training=self.args.noisy_training)
                # self.multipliers_table.append(torch.from_numpy(np.stack([
                #                         outputs_list[i][0].view(num_graphs,self.args.n).detach().cpu().numpy() for i in range(len(outputs_list))
                #                         ])).view(-1, self.args.n))
                
                if self.args.supervised:
                    assert 'target' in data, 'Supervised training requires target in data'
                #     mu_over_time, _, _, _, _ = self.dual_model.DA(self.primal_model, data, 0.1, 100, 
                #                                                     self.args.n, self.args.r_min, self.noise_var, 200, self.args.ss_param, 
                #                                                     self.args.mu_init, self.mu_uncons, self.device, False, True)
                #     target = mu_over_time[-1].unsqueeze(-1).to(self.device)
                #     del mu_over_time
                    

                # calculate the loss
                loss, constraints_loss, dual_gap = self.dual_model.loss(outputs_list, self.args.r_min, num_graphs*num_samplers, 
                                                                        constraint_eps=self.args.dual_constraint_eps, metric=self.args.metric,
                                                                        resilient_weight_decay=100, #self.args.dual_resilient_decay, #Add regularization to the multipliers
                                                                        dual_training_loss=self.args.dual_training_loss,
                                                                        rates_prop_grads=self.args.rates_prop_grads,
                                                                        supervised=self.args.supervised, target=data.target if self.args.supervised else None)
                L = loss + torch.sum(training_multipliers * constraints_loss)

                self.dual_optimizer.zero_grad()
                torch.autograd.set_detect_anomaly(True)
                L.backward()                        # Minimize the loss
                self.dual_optimizer.step()


                # update the training multipliers with resilience
                if self.args.training_resilient_decay > 0:
                    training_multipliers = (1-1/self.args.training_resilient_decay) * training_multipliers + self.args.lr_dual_multiplier * constraints_loss.detach()
                    multiplier_update = True
                else:
                    training_multipliers = training_multipliers + self.args.lr_dual_multiplier * constraints_loss.detach()
                    multiplier_update = True
                training_multipliers = torch.maximum(training_multipliers, torch.zeros_like(training_multipliers))

                loss_list.append((loss + torch.maximum(constraints_loss, torch.zeros_like(constraints_loss)).sum()).item())
                self.dual_trained = True

                if self.args.use_wandb:
                    wandb.log({'dual/dual pure loss': loss})
                    wandb.log({'dual/dualality gap': dual_gap})
                    wandb.log({'dual/dual training loss': L.item()})
                    wandb.log({'dual/constraint loss': torch.mean(constraints_loss.detach().cpu()).item()})
                    
            # clear the cuda memory from data
            del data, edge_index_l, edge_weight_l, a_l, transmitters_index
            torch.cuda.empty_cache()
        
        return np.stack(loss_list).mean(), training_multipliers if multiplier_update else None

    def validate(self, loader):
        assert self.primal_model is not None, 'Primal model is not defined'
        assert self.dual_model is not None, 'Dual model is not defined'
        assert self.device is not None, 'Device is not defined'
        assert self.args is not None, 'Args are not defined'

        L = []
        
        for data, batch_idx in loader:
            self.primal_model.zero_grad()
            data = data.to(self.device)
            y, edge_index_l, edge_weight_l, _, \
            _, _, a_l, transmitters_index, num_graphs = \
                data.y, data.edge_index_l, data.edge_weight_l, data.edge_index, data.edge_weight, \
                data.weighted_adjacency, data.weighted_adjacency_l, \
                data.transmitters_index, data.num_graphs
            
            edge_index_l, edge_weight_l = GDC().sparsify_sparse(edge_index=edge_index_l, edge_weight=edge_weight_l,
                                                                    num_nodes=self.args.n, method="threshold", eps=2e-2)
            
            self.dual_model.eval()
            self.primal_model.eval()
            
            # Forward pass
            outputs_list = self.unroll_DA(edge_index_l=edge_index_l, edge_weight_l=edge_weight_l, a_l=a_l, transmitters_index=transmitters_index, num_graphs=num_graphs)
            
            # calculate the loss
            loss, constraints_loss, dual_gap = self.dual_model.loss(outputs_list, self.args.r_min, num_graphs, constraint_eps=0.0, metric=self.args.metric,
                                                        resilient_weight_decay=self.args.dual_resilient_decay, 
                                                        dual_training_loss="complementary_slackness",
                                                        rates_prop_grads=self.args.rates_prop_grads,
                                                        supervised=self.args.supervised, target=data.target if self.args.supervised else None)
            
            L.append(loss + 0.5*torch.maximum(constraints_loss, 
                                           torch.zeros_like(constraints_loss)).sum() + dual_gap.abs())
            
        return torch.stack(L).mean().item()


    def eval_primal(self, loader, num_iters=None, **kwargs):
        adjust_constraints = kwargs.get('adjust_constraints', True)
        fix_mu_uncons = kwargs.get('fix_mu_uncons', True)

        test_results = defaultdict(list)

        for data, _ in loader:
            # DA
            mu_over_time, L_over_time, all_Ps, all_rates, violation_dict = \
                self.dual_model.DA(self.primal_model, data, self.args.lr_DA_dual, self.args.dual_resilient_decay,  #self.args.lr_DA_dual, self.args.dual_resilient_decay, 
                                        self.args.n, self.args.r_min, self.noise_var, num_iters, self.args.ss_param, 
                                        self.args.mu_init, self.mu_uncons, self.device,
                                        adjust_constraints, fix_mu_uncons)
        violation = violation_dict['violation']

        constrained_agents = int(np.floor(self.args.constrained_subnetwork*self.args.n)) 
        percentile_list = [5, 10, 15, 20, 30, 40, 50]

        test_results['rate_mean'].append(violation_dict['rate_mean'])
        test_results['all_Ps'].append(torch.stack(all_Ps).detach().cpu())
        test_results['all_rates'].append(torch.stack(all_rates).detach().cpu())
        test_results['test_mu_over_time'].append(torch.stack(mu_over_time).detach().cpu())
        test_results['test_L_over_time'].append(torch.stack(L_over_time).detach().cpu())
        for percentile in percentile_list:
            test_results[f'rate_{percentile}th_percentile'].append(-1*np.percentile(-1 * violation.view(data.num_graphs, self.args.n)[:, :constrained_agents].detach().cpu().numpy(), percentile, axis=1).mean().tolist())
        test_results['mean_violation'].append(violation.mean().item())
        test_results['violation_rate'].append(violation_dict['violation_rate'])
        test_results['constrained_mean_rate'].append(violation_dict['constrained_rate_mean'])
        test_results['unconstrained_mean_rate'].append(violation_dict['unconstrained_rate_mean'])
        test_results['L_over_time'].append(torch.stack(L_over_time).detach().cpu().numpy())

        return test_results
            

    def eval(self, loader, num_iters=None, **kwargs):
        adjust_constraints = kwargs.get('adjust_constraints', True)
        fix_mu_uncons = kwargs.get('fix_mu_uncons', True)

        test_results = defaultdict(list)
        unrolling_results = defaultdict(list)
        randPolicy_results = defaultdict(list)
        full_power_results = defaultdict(list)
        if hasattr(self.primal_model, 'cons_lvl'):
            delattr(self.primal_model, 'cons_lvl')
        for data, _ in loader:
            # DA
            mu_over_time, L_over_time, all_Ps, all_rates, violation_dict = \
                self.dual_model.DA(self.primal_model, data, self.args.lr_DA_dual, self.args.dual_resilient_decay, 
                                        self.args.n, self.args.r_min, self.noise_var, num_iters, self.args.ss_param, 
                                        self.args.mu_init, self.mu_uncons, self.device,
                                        adjust_constraints, fix_mu_uncons)
            violation = violation_dict['violation']

            constrained_agents = int(np.floor(self.args.constrained_subnetwork*self.args.n)) 
            percentile_list = [5, 10, 15, 20, 30, 40, 50]

            test_results['rate_mean'].append(violation_dict['rate_mean'])
            test_results['all_Ps'].append(torch.stack(all_Ps).detach().cpu())
            test_results['all_rates'].append(torch.stack(all_rates).detach().cpu())
            test_results['test_mu_over_time'].append(torch.stack(mu_over_time).detach().cpu())
            test_results['test_L_over_time'].append(torch.stack(L_over_time).detach().cpu())
            for percentile in percentile_list:
                test_results[f'rate_{percentile}th_percentile'].append(-1*np.percentile(-1 * violation.view(-1, self.args.n)[1:, :constrained_agents].detach().cpu().numpy(), percentile, axis=1).mean().tolist())
            test_results['mean_violation'].append(violation.mean().item())
            test_results['violation_rate'].append(violation_dict['violation_rate'])
            test_results['constrained_mean_rate'].append(violation_dict['constrained_rate_mean'])
            test_results['unconstrained_mean_rate'].append(violation_dict['unconstrained_rate_mean'])
            test_results['L_over_time'].append(torch.stack(L_over_time).detach().cpu().numpy())


            slack_value = violation_dict['slack_value'] if adjust_constraints else 0

            # Unrolling 
            if self.dual_trained:
                self.primal_model.eval()
                self.dual_model.eval()
                
                outputs_list = self.unroll_DA(data=data)
                _, _, rates, dual_fn = outputs_list[-1]
                # rates = rates.squeeze(-1)
                for i in range(len(outputs_list)):
                    violation = torch.minimum(outputs_list[i][2].detach() - self.args.r_min - slack_value, torch.zeros_like(rates)).abs()
                    violation_rate = (violation>0).sum().float()/violation.numel()
                    unrolling_results['violation_rate'].append(violation_rate.item())
                
                unrolling_results['dual_fn'].append([outputs_list[i][-1].item() for i in range(len(outputs_list))]) 
                unrolling_results['test_mu_over_time'].append(np.stack([outputs_list[i][0].squeeze(-1).detach().cpu().numpy() for i in range(len(outputs_list))]))
                unrolling_results['all_Ps'].append(np.stack([outputs_list[i][1].squeeze(-1).detach().cpu().numpy() for i in range(len(outputs_list))]))
                unrolling_results['all_rates'].append(np.stack([outputs_list[i][2].squeeze(-1).detach().cpu().numpy() for i in range(len(outputs_list))]))
                unrolling_results['rate_mean'].append(torch.mean(rates.view(data.num_graphs, self.args.n).mean(1).detach().cpu()).tolist())
                unrolling_results['rate_min'].append(torch.min(rates.detach().cpu()).tolist())
                for percentile in percentile_list:
                    unrolling_results[f'rate_{percentile}th_percentile'].append(-1*np.percentile(-1 * violation.view(data.num_graphs, self.args.n)[:, :constrained_agents].detach().cpu().numpy(), percentile, axis=1).mean().tolist())
                # unrolling_results['violation_rate'].append(violation_rate.item())
                unrolling_results['mean_violation'].append(violation.mean().item())
                unrolling_results['constrained_mean_rate'].append(torch.mean(rates.view(data.num_graphs, self.args.n)[:, :constrained_agents].mean(1).detach().cpu()).tolist())
                unrolling_results['unconstrained_mean_rate'].append(torch.mean(rates.view(data.num_graphs, self.args.n)[:, constrained_agents:].mean(1).detach().cpu()).tolist())
            
            # random policy
            # p = torch.relu_(P_max * (torch.randn(num_graphs * n).to(device) + 0.5)/2)
            p = self.args.P_max * torch.rand(data.num_graphs * self.args.n).to(self.device)
            gamma = torch.ones(data.num_graphs * self.args.n, 1).to(self.device)
            rates = calc_rates(p, gamma, data.weighted_adjacency_l[:, :, :], self.noise_var, self.args.ss_param)
            violation = torch.minimum(rates - self.args.r_min - slack_value, torch.zeros_like(rates)).abs()
            violation_rate = (violation>0).sum().float()/violation.numel()
            
            randPolicy_results['rate_mean'].append(torch.mean(rates.view(data.num_graphs, self.args.n).mean(1).detach().cpu()).tolist())
            randPolicy_results['rate_min'].append(torch.min(rates.detach().cpu()).tolist())
            for percentile in percentile_list:
                    randPolicy_results[f'rate_{percentile}th_percentile'].append(-1*np.percentile(-1 * violation.view(-1, self.args.n)[1:, :constrained_agents].detach().cpu().numpy(), percentile, axis=1).mean().tolist())
            randPolicy_results['all_Ps'].append(p.detach().cpu().numpy())
            randPolicy_results['all_rates'].append(rates.detach().cpu().numpy())
            randPolicy_results['violation_rate'].append(violation_rate.item())
            randPolicy_results['mean_violation'].append(violation.mean().item())
            randPolicy_results['constrained_mean_rate'].append(torch.mean(rates.view(data.num_graphs, self.args.n)[:, :constrained_agents].mean(1).detach().cpu()).tolist())
            randPolicy_results['unconstrained_mean_rate'].append(torch.mean(rates.view(data.num_graphs, self.args.n)[:, constrained_agents:].mean(1).detach().cpu()).tolist())


            # Full power
            p = self.args.P_max * torch.ones(data.num_graphs * self.args.n).to(self.device)
            rates = calc_rates(p, gamma, data.weighted_adjacency_l[:, :, :], self.noise_var, self.args.ss_param)
            violation = torch.minimum(rates - self.args.r_min - slack_value, torch.zeros_like(rates)).abs() 
            violation_rate = (violation>0).sum().float()/violation.numel()

            full_power_results['rate_mean'].append(torch.mean(rates.view(data.num_graphs, self.args.n).mean(1).detach().cpu()).tolist())
            full_power_results['rate_min'].append(torch.min(rates.detach().cpu()).tolist())
            for percentile in percentile_list:
                    full_power_results[f'rate_{percentile}th_percentile'].append(-1*np.percentile(-1 * violation.view(-1, self.args.n)[1:, :constrained_agents].detach().cpu().numpy(), percentile, axis=1).mean().tolist())
            full_power_results['all_Ps'].append(p.detach().cpu().numpy())
            full_power_results['all_rates'].append(rates.detach().cpu().numpy())
            full_power_results['violation_rate'].append(violation_rate.item())
            full_power_results['mean_violation'].append(violation.mean().item())
            full_power_results['constrained_mean_rate'].append(torch.mean(rates.view(data.num_graphs, self.args.n)[:, :constrained_agents].mean(1).detach().cpu()).tolist())
            full_power_results['unconstrained_mean_rate'].append(torch.mean(rates.view(data.num_graphs, self.args.n)[:, constrained_agents:].mean(1).detach().cpu()).tolist())

        return test_results, unrolling_results if self.dual_trained else None, randPolicy_results, full_power_results
