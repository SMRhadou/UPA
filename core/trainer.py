import torch
import numpy as np
from utils import calc_rates
import wandb

from collections import defaultdict



class Trainer():
    def __init__(self, primal_model, dual_model, loader=None, optimizers=None, device=None, args=None, **kwargs):
        self.primal_model = primal_model
        self.dual_model = dual_model
        self.loader = loader
        self.primal_optimizer = optimizers['primal'] if optimizers is not None else None
        self.dual_optimizer = optimizers['dual'] if optimizers is not None else None
        self.device = device
        # N = -174 - 30 + 10 * np.log10(args.BW)
        self.noise_var = args.noise_var #np.power(10, N / 10)
        self.args = args
        self.dual_trained = kwargs.get('dual_trained', False)
        self.multipliers_table = []
        


    def multiplier_sampler(self, num_samplers, mu_max, *dist_param, dist='uniform', all_zeros=True):
        if dist == 'uniform':
            assert len(dist_param) == 1
            mu_initial = mu_max * torch.rand(num_samplers, 1)
        else:
            raise NotImplementedError
        
        zero_probability = dist_param[0] if len(dist_param) > 0 else 0.2
        if all_zeros:
            zero_ids = torch.rand(num_samplers, 1) < zero_probability
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
        
        return mu.view(-1, 1)


    def unroll_DA(self, data):
        data = data.to(self.device)
        edge_index_l, edge_weight_l, a_l, transmitters_index, num_graphs = \
            data.edge_index_l, data.edge_weight_l, data.weighted_adjacency_l, \
            data.transmitters_index, data.num_graphs
        


        mu = torch.cat((0.01 * torch.rand(num_graphs, int(np.floor(self.args.constrained_subnetwork*self.args.m))).to(self.device), 
                        30*torch.ones(num_graphs, int(np.ceil((1-self.args.constrained_subnetwork)*self.args.m))).to(self.device)), dim=1)
        mu = mu.view(num_graphs * self.args.n, 1)

        gamma = torch.ones(num_graphs * self.args.n, 1).to(self.device)
        outputs_list = []
        for block_id in range(self.dual_model.num_blocks):
            p = self.primal_model(mu, edge_index_l, edge_weight_l, transmitters_index)
            rates = calc_rates(p, gamma, a_l[:, :, :], self.noise_var, self.args.ss_param)
            L = self.primal_model.loss(rates, mu, p, constrained=False, metric='power' if self.primal_model.args.metric == 'power' else 'rates')/ self.args.n
            outputs_list.append((mu, p, rates, L))
            mu = mu + self.dual_model(block_id, mu, p, edge_index_l, edge_weight_l, transmitters_index)
            mu = torch.relu(mu)
        
        # Primal recovery
        p = self.primal_model(mu, edge_index_l, edge_weight_l, transmitters_index)
        rates = calc_rates(p, gamma, a_l[:, :, :], self.noise_var, self.args.ss_param)
        L = self.primal_model.loss(rates, mu, p, constrained=False, metric='power' if self.primal_model.args.metric == 'power' else 'rates') /self.args.n
        outputs_list.append((mu, p, rates, L))

        if self.args.use_wandb:
            wandb.log({'dual lagrangian loss': L.item()})
            wandb.log({'max_lambda': torch.max(mu.detach().view(-1, self.args.n)[:,:int(np.floor(self.args.constrained_subnetwork*self.args.n)) ].cpu()).item()})
            wandb.log({'mean rate': torch.mean(rates.view(num_graphs, self.args.n).mean(1).detach().cpu()).item()})
            wandb.log({'min rate': torch.min(rates.detach().cpu()).item()})
            wandb.log({'10th_percentile_rate': torch.quantile(rates.detach().cpu(), 0.1).item()})
            wandb.log({'90th_percentile_rate': torch.quantile(rates.detach().cpu(), 0.9).item()})
            wandb.log({'max rate': torch.max(rates.detach().cpu()).item()})
            wandb.log({'min P': torch.min(p.detach().cpu()).item()})
            wandb.log({'10th_percentile_P': torch.quantile(p.detach().cpu(), 0.1).item()})
            wandb.log({'90th_percentile_P': torch.quantile(p.detach().cpu(), 0.9).item()})
            wandb.log({'max P': torch.max(p.detach().cpu()).item()})

        return outputs_list



    def train(self, epoch, training_multipliers, mode='primal'):
        assert self.loader is not None, 'Data loader is not defined'
        assert self.primal_optimizer is not None, 'Primal optimizer is not defined'
        assert self.dual_optimizer is not None, 'Dual optimizer is not defined'
        assert self.primal_model is not None, 'Primal model is not defined'
        assert self.dual_model is not None, 'Dual model is not defined'
        assert self.device is not None, 'Device is not defined'
        assert self.args is not None, 'Args are not defined'

        num_samplers = self.args.num_samplers if hasattr(self.args, 'num_samplers') else 1
        # initialize the training multipliers
        

        for data, batch_idx in self.loader:
            self.primal_model.zero_grad()
            data = data.to(self.device)
            y, edge_index_l, edge_weight_l, _, \
            _, _, a_l, transmitters_index, num_graphs = \
                data.y, data.edge_index_l, data.edge_weight_l, data.edge_index, data.edge_weight, \
                data.weighted_adjacency, data.weighted_adjacency_l, \
                data.transmitters_index, data.num_graphs
            
            loss_list = []
            if num_samplers > 1:
                edge_index_l = edge_index_l.repeat(1, num_samplers)
                edge_weight_l = edge_weight_l.repeat(num_samplers)
                a_l = a_l.repeat(1, num_samplers, num_samplers)
                transmitters_index = torch.arange(num_samplers*self.args.n).to(self.device)

            if mode == 'primal':
                self.primal_model.train()
                self.dual_model.eval()
                
                mu = self.multiplier_sampler(num_graphs*num_samplers*self.args.n, self.args.mu_max, self.args.zero_probability, all_zeros=self.args.all_zeros).to(self.device)
                p = self.primal_model(mu.detach(), edge_index_l, edge_weight_l, transmitters_index)    # MU is normalized to [0, 1]
                gamma = torch.ones(num_samplers*num_graphs * self.args.n, 1).to(self.device) 
                rates = calc_rates(p, gamma, a_l[:, :, :], self.noise_var, self.args.ss_param)

                loss = self.primal_model.loss(rates, mu, p, constrained=False, metric=self.args.metric)
                self.primal_optimizer.zero_grad()
                loss.backward()
                self.primal_optimizer.step()

                loss_list.append(loss.item())

                if self.args.use_wandb:
                    wandb.log({'primal training loss': loss.item()})
                    wandb.log({'mean rate': torch.mean(rates.view(num_graphs*num_samplers, self.args.n).mean(1).detach().cpu()).item()})
                    wandb.log({'min rate': torch.min(rates.detach().cpu()).item()})
                    wandb.log({'10th_percentile_rate': torch.quantile(rates.detach().cpu(), 0.1).item()})
                    wandb.log({'90th_percentile_rate': torch.quantile(rates.detach().cpu(), 0.9).item()})
                    wandb.log({'max rate': torch.max(rates.detach().cpu()).item()})
                    wandb.log({'min P': torch.min(p.detach().cpu()).item()})
                    wandb.log({'10th_percentile_P': torch.quantile(p.detach().cpu(), 0.1).item()})
                    wandb.log({'90th_percentile_P': torch.quantile(p.detach().cpu(), 0.9).item()})
                    wandb.log({'max P': torch.max(p.detach().cpu()).item()})


            else:
                self.dual_model.train()
                self.primal_model.eval()

                if epoch == 0:
                    self.multipliers_table = []
                
                # Forward pass
                outputs_list = self.unroll_DA(data)
                self.multipliers_table.append(torch.from_numpy(np.stack([
                                        outputs_list[i][0].view(num_graphs,self.args.n).detach().cpu().numpy() for i in range(len(outputs_list))
                                        ])).view(-1, self.args.n))

                # calculate the loss
                loss, constraints_loss = self.dual_model.loss(outputs_list, self.args.r_min, num_graphs, constraint_eps=0.0, metric=self.args.metric,
                                                              resilient_weight_decay=self.args.dual_resilient_decay, 
                                                              supervised=self.args.supervised, target=data.target if self.args.supervised else None)
                L = loss + torch.sum(training_multipliers * constraints_loss)

                self.dual_optimizer.zero_grad()
                L.backward()                        # Minimize the loss
                self.dual_optimizer.step()

                # update the training multipliers with resilience
                if self.args.training_resilient_decay > 0:
                    training_multipliers = (1-1/self.args.training_resilient_decay) * training_multipliers + self.args.lr_dual_multiplier * constraints_loss.detach()
                else:
                    training_multipliers = training_multipliers + self.args.lr_dual_multiplier * constraints_loss.detach()
                training_multipliers = torch.maximum(training_multipliers, torch.zeros_like(training_multipliers))

                loss_list.append((loss + torch.maximum(constraints_loss, torch.zeros_like(constraints_loss)).sum()).item())
                # loss_list.append(loss.item())
                self.dual_trained = True

                if self.args.use_wandb:
                    wandb.log({'dual training loss': L.item()})
                    wandb.log({'constraint loss': torch.mean(constraints_loss.detach().cpu()).item()})
                    
        
        return np.stack(loss_list).mean(), training_multipliers if mode == 'dual' else None


            

    def eval(self, loader, num_iters=None, **kwargs):
        adjust_constraints = kwargs.get('adjust_constraints', True)

        test_results = defaultdict(list)
        unrolling_results = defaultdict(list)
        randPolicy_results = defaultdict(list)
        full_power_results = defaultdict(list)
        for data, _ in loader:
            # DA
            mu_over_time, L_over_time, all_Ps, all_rates, violation_dict = \
                self.dual_model.DA(self.primal_model, data, self.args.lr_DA_dual, self.args.dual_resilient_decay, 
                                        self.args.n, self.args.r_min, self.noise_var, num_iters, self.args.ss_param, self.device,
                                        adjust_constraints)
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


            slack_value = violation_dict['slack_value'] if adjust_constraints else 0

            # Unrolling 
            if self.dual_trained:
                self.primal_model.eval()
                self.dual_model.eval()
                
                outputs_list = self.unroll_DA(data)
                _, _, rates, _ = outputs_list[-1]
                # rates = rates.squeeze(-1)
                for i in range(len(outputs_list)):
                    violation = torch.minimum(outputs_list[i][2].detach() - self.args.r_min - slack_value, torch.zeros_like(rates)).abs()
                    violation_rate = (violation>0).sum().float()/violation.numel()
                    unrolling_results['violation_rate'].append(violation_rate.item())

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
                    randPolicy_results[f'rate_{percentile}th_percentile'].append(-1*np.percentile(-1 * violation.view(data.num_graphs, self.args.n)[:, :constrained_agents].detach().cpu().numpy(), percentile, axis=1).mean().tolist())
            randPolicy_results['all_Ps'].append(p.detach().cpu().numpy())
            randPolicy_results['all_rates'].append(rates.detach().cpu().numpy())
            randPolicy_results['violation_rate'].append(violation_rate.item())
            randPolicy_results['mean_violation'].append(violation.mean().item())
            randPolicy_results['constrained_mean_rate'].append(torch.mean(rates.view(data.num_graphs, self.args.n)[:, :constrained_agents].mean(1).detach().cpu()).tolist())


            # Full power
            p = self.args.P_max * torch.ones(data.num_graphs * self.args.n).to(self.device)
            rates = calc_rates(p, gamma, data.weighted_adjacency_l[:, :, :], self.noise_var, self.args.ss_param)
            violation = torch.minimum(rates - self.args.r_min - slack_value, torch.zeros_like(rates)).abs() 
            violation_rate = (violation>0).sum().float()/violation.numel()

            full_power_results['rate_mean'].append(torch.mean(rates.view(data.num_graphs, self.args.n).mean(1).detach().cpu()).tolist())
            full_power_results['rate_min'].append(torch.min(rates.detach().cpu()).tolist())
            for percentile in percentile_list:
                    full_power_results[f'rate_{percentile}th_percentile'].append(-1*np.percentile(-1 * violation.view(data.num_graphs, self.args.n)[:, :constrained_agents].detach().cpu().numpy(), percentile, axis=1).mean().tolist())
            full_power_results['all_Ps'].append(p.detach().cpu().numpy())
            full_power_results['all_rates'].append(rates.detach().cpu().numpy())
            full_power_results['violation_rate'].append(violation_rate.item())
            full_power_results['mean_violation'].append(violation.mean().item())
            full_power_results['constrained_mean_rate'].append(torch.mean(rates.view(data.num_graphs, self.args.n)[:, :constrained_agents].mean(1).detach().cpu()).tolist())

        return test_results, unrolling_results if self.dual_trained else None, randPolicy_results, full_power_results
