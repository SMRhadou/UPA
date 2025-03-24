import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LEConv, TAGConv, global_add_pool, global_max_pool, global_mean_pool
from torch_scatter import scatter

import torch_geometric.nn as pyg_nn
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from core.ResidualGNN import SinusoidalTimeEmbedding, NormLayer


class ResGraphConvBlock(nn.Module):
    """ 
    Residual GCN block.
    """
    def __init__(self, conv_layer, norm_layer, activation, dropout_rate, res_connection, layer_ord = ['conv', 'norm', 'act', 'dropout']):
        super().__init__()
        self.conv_layer = conv_layer
        self.norm_layer = norm_layer
        self.activation = activation
        self.res_connection = res_connection
        self.dropout_rate = dropout_rate
        self.layer_ord = layer_ord


    def forward(self, y: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None):
        if self.layer_ord == ['conv', 'norm', 'act', 'dropout']:

            if any([isinstance(self.conv_layer, _) for _ in [LEConv, TAGConv]]):
                h = self.conv_layer(y, edge_index=edge_index, edge_weight=edge_weight)
            else:
                h = self.conv_layer(y, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1))

            h = self.norm_layer(h, batch = batch) # identity layer if no batch/graph normalization is used.
            h = self.activation(h)
            h = F.dropout(h, p = self.dropout_rate, training=self.training)
            out = self.res_connection(y) + h

        else:
            raise NotImplementedError
        
        return out
    

class res_gnn_backbone(torch.nn.Module):
    def __init__(self, num_features_list, **kwargs):
        super(res_gnn_backbone, self).__init__()

        self.layer_ord = ['conv', 'norm', 'act', 'dropout']

        k_hops = kwargs.get('k_hops', 2)
        num_layers = len(num_features_list)
        activation = kwargs.get('activation', 'leaky_relu')
        # aggregation = kwargs.get('aggregation', None)

        self.dropout_rate = kwargs.get('dropout_rate', 0.0)
        # num_heads = kwargs.get('num_heads', 2)
        norm = kwargs.get('norm_layer', 'batch')
        global_pooling = kwargs.get('global_pooling', None)

        # Define activation functions
        if activation == 'leaky_relu':
            self.activation = F.leaky_relu
        elif activation == 'relu':
            self.activation = F.relu
        elif activation == 'sigmoid':
            self.activation = F.sigmoid
        elif activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'softplus':
            self.activation = F.softplus
        else:
            raise NotImplementedError
        
        # Define optional pooling layers after the last conv + (batch norm + nonlinearity + dropout) layer
        self.global_pooling_layer = None
        if global_pooling is not None:
            if global_pooling == 'max':
                self.global_pooling_layer = global_max_pool
            elif global_pooling == 'mean':
                self.global_pooling_layer = global_mean_pool
            elif global_pooling == 'add':
                self.global_pooling_layer = global_add_pool
        

        self.blocks = nn.ModuleList()
        # self.norm_layers = nn.ModuleList()
        # self.res_connections = nn.ModuleList()

        for i in range(num_layers - 1):
            conv_layer = TAGConv(in_channels=num_features_list[i], out_channels=num_features_list[i+1], K=k_hops, normalize=False)
            # self.conv_layers.append(conv_layer)

            norm_layer = NormLayer(norm=norm,
                                   in_channels=num_features_list[i+1] if self.layer_ord.index('norm') > self.layer_ord.index('conv') else num_features_list[i])
            # self.norm_layers.append(norm_layer)

            # If the number of input channels is not equal to the number of output channels we have to project the shortcut connection
            if num_features_list[i] != num_features_list[i+1]:
                res_connection = nn.Linear(in_features=num_features_list[i], out_features=num_features_list[i+1], bias = False)
            else:
                res_connection = nn.Identity()
            # self.res_connections.append(res_connection)
                
            res_block = ResGraphConvBlock(conv_layer=conv_layer, norm_layer=norm_layer, activation=self.activation,
                                          dropout_rate=self.dropout_rate, res_connection=res_connection,
                                          layer_ord=self.layer_ord
                                          )
            self.blocks.append(res_block)


            
    def forward(self, y, edge_index, edge_weight, batch = None):
        # Apply normalization or get sinusoidal embeddings of lambdas before passing through graph-conv layers.
        # pos_embedding_scaling = 50 / LAMBDAS_MAX

        # for i, (norm_layer, conv_layer, res_connection) in enumerate(zip(self.norm_layers, self.conv_layers, self.res_connections)):
            
        #     if any([isinstance(conv_layer, _) for _ in [LEConv, TAGConv]]):
        #         y = conv_layer(y, edge_index=edge_index, edge_weight=edge_weight)
        #     else:
        #         y = conv_layer(y, edge_index=edge_index, edge_attr=edge_weight.unsqueeze(-1))

        #     # if i < len(self.conv_layers)-1:
        #     y = norm_layer(y, batch = batch) # identity layer if no batch normalization is used.
        #     y = self.activation(y)
        #     y = F.dropout(y, p = self.dropout_rate, training=self.training)

        for block in self.blocks:
            y = block(y=y, edge_index=edge_index, edge_weight=edge_weight, batch=batch)

        if self.global_pooling_layer is not None and batch is not None:
            y = self.global_pooling_layer(y, batch)
            
        return y

# backbone GNN class
class gnn_backbone(torch.nn.Module):
    def __init__(self, num_features_list, k_hops=1, primal=True):
        super(gnn_backbone, self).__init__()
        num_layers = len(num_features_list)
        self.num_features_list = num_features_list
        self.primal = primal
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.res_connections = nn.ModuleList()

        for i in range(num_layers - 1):
            self.layers.append(TAGConv(num_features_list[i], num_features_list[i + 1], K=k_hops, normalize=False))
            self.batch_norms.append(nn.BatchNorm1d(num_features_list[i + 1]))
            self.norms.append(nn.LayerNorm(num_features_list[i + 1]))

            if self.num_features_list[i] != self.num_features_list[i+1]:
                self.res_connections.append(nn.Linear(in_features=self.num_features_list[i], out_features=self.num_features_list[i+1], bias = False))
            else:
                self.res_connections.append(nn.Identity())


    def forward(self, y, edge_index, edge_weight):
        for i, block in enumerate(zip(self.layers, self.batch_norms, self.res_connections)):
            h = block[0](y, edge_index=edge_index, edge_weight=edge_weight)
            if not self.primal:    
                h = block[1](h)
            y = h + block[2](y)
            if self.primal:
                y = F.leaky_relu(y)
            else:
                y = F.tanh(y)
        return y
    
# main GNN module
class GNN(torch.nn.Module):
    def __init__(self, num_features_list, P_max, primal=True):
        super(GNN, self).__init__()
        self.gnn_backbone = gnn_backbone(num_features_list, primal)
        self.b_p = nn.Linear(num_features_list[-1], 1, bias=False)
        self.P_max = P_max
        self.activation = torch.sigmoid if primal else torch.tanh

        self.apply(self.init_weights)
        
    def forward(self, y, edge_index, edge_weight, transmitters_index):
        y = self.gnn_backbone(y, edge_index, edge_weight) # derive node embeddings
        Tx_embeddings = scatter(y, transmitters_index, dim=0, reduce='mean')        
        p = self.P_max * torch.sigmoid(self.b_p(Tx_embeddings)) # derive power levels for transmitters
        return p
    

    def init_weights(self, m):
        """ Custom weight initialization. """
        if isinstance(m, pyg_nn.TAGConv):
            # print("m: ", m)
            # Apply He initialization to the weight matrix in GraphConv
            for lin in m.lins:
                torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity='leaky_relu')
                # if self.activation._get_name() == "LeakyReLU":
                #     torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity='leaky_relu')  # He init for weight
                # elif self.activation._get_name() == 'ReLU':
                #     torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity='relu')  # He init for weight
                # else:
                #     pass
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)  # Bias is initialized to zero
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier init for final layers
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
    


class PrimalGNN(torch.nn.Module):
    def __init__(self, num_features_list, P_max, **kwargs):
        super(PrimalGNN, self).__init__()
        
        lambdas_embedding = kwargs.get('lambdas_embedding', None)
        self.lambdas_max = kwargs.get('lambdas_max', 0.0)
        self.conv_layer_normalize = kwargs.get("conv_layer_normalize", False)

        # self.resolve_lambdas_embedding(lambdas_embedding=lambdas_embedding, embed_dim=num_features_list[0])
        
        # self.gnn_backbone = gnn_backbone(conv_model_architecture=conv_model, num_features_list=num_features_list, **kwargs)
        self.gnn_backbone = res_gnn_backbone(num_features_list=num_features_list, **kwargs)
        self.b_p = nn.Linear(num_features_list[-1], 1, bias=True)
        self.P_max = P_max
        

        self.apply(self.init_weights)

        self.n_iters_trained = 0

    @property
    def pos_embedding_scaling(self):
        if hasattr(self, 'lambdas_max') and self.lambdas_max is not None and self.lambdas_max > 0:
            return 10000 / self.lambdas_max
        else:
            return 100
    
    def resolve_lambdas_embedding(self, lambdas_embedding, embed_dim):
        if lambdas_embedding is None or lambdas_embedding in ['none']:
            assert embed_dim == 1, "Input node signals should have a single channel if no embeddings are used."
            self.lambdas_embedding = nn.Identity()
        elif lambdas_embedding == 'sinusoidal':
            hidden_dim = embed_dim * 4
            self.lambdas_embedding = nn.Sequential(SinusoidalTimeEmbedding(n_channels=hidden_dim), nn.Linear(hidden_dim, embed_dim, bias=False))
        else:
            raise NotImplementedError

    
    def forward(self, y, edge_index, edge_weight, transmitters_index, batch = None, activation=None):

        if self.conv_layer_normalize:
            add_self_loops = True
            improved = add_self_loops
            edge_index, edge_weight = gcn_norm(  # yapf: disable
                    edge_index, edge_weight, y.size(0),
                    improved=improved, add_self_loops=add_self_loops, flow=self.gnn_backbone.blocks[0].conv_layer.flow,
                    dtype=y.dtype)
            print("GCN normalization applied to edge_weights.") if self.n_iters_trained == 0 else None
            # edge_weight = self.spectral_laplacian_normalize(edge_index, edge_weight, num_nodes=y.size(0))
            # print("Spectral Laplacian normalization applied to edge_weights.")

        # y = y.view(-1, self.gnn_backbone.conv_layers[0].in_channels)
        # y = y.view(-1, 1)
        # y = self.lambdas_embedding(self.pos_embedding_scaling * y)
        # y = y / y.sum(dim = -1, keepdim = True).clamp(min = 1) # normalization layer
        y = self.gnn_backbone(y, edge_index, edge_weight, batch = batch) # derive node embeddings
        Tx_embeddings = self.b_p(scatter(y, transmitters_index, dim=0, reduce='mean')) 
        if activation == 'sigmoid':      
            p = self.P_max * torch.sigmoid(Tx_embeddings) # derive power levels for transmitters
        elif activation == 'tanh':
            p = torch.tanh(Tx_embeddings)
        elif activation == None:
            p = Tx_embeddings
        else:
            raise NotImplementedError
            
        self.n_iters_trained += 1
        
        return p
    

    # Manual Laplacian Normalization
    def laplacian_normalize(self, edge_index, edge_weight, num_nodes):
        row, col = edge_index
        deg = degree(row, num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 0  # Avoid division by zero
        return deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
    

    def spectral_laplacian_normalize(self, edge_index, edge_weight, num_nodes):
        """Computes spectral Laplacian normalization for a batched PyG graph."""
        row, col = edge_index

        # Compute the degree matrix
        deg = degree(row, num_nodes, dtype=edge_weight.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg == 0] = 0  # Avoid division by zero

        # Compute normalized adjacency: A_norm = D^(-1/2) * A * D^(-1/2)
        normalized_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

        # Compute spectral Laplacian: L = I - A_norm
        identity_weight = torch.ones(num_nodes, dtype=edge_weight.dtype, device=edge_weight.device)
        spectral_laplacian_weight = identity_weight - normalized_weight  # L = I - A_norm

        return spectral_laplacian_weight
    

    def init_weights(self, m):
        """ Custom weight initialization. """
        if isinstance(m, pyg_nn.TAGConv):
            # print("m: ", m)
            # Apply He initialization to the weight matrix in GraphConv
            for lin in m.lins:
                # if self.activation._get_name() == "LeakyReLU":
                torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity='leaky_relu')  # He init for weight
                # elif self.activation._get_name() == 'ReLU':
                    # torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity='relu')  # He init for weight
                # else:
                    # pass
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)  # Bias is initialized to zero
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier init for final layers
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
     

