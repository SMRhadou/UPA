import torch 
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torch_geometric.nn import GCNConv, TopKPooling, TAGConv, LEConv, GATv2Conv, BatchNorm, LayerNorm, DiffGroupNorm, GraphNorm, GraphUNet

import torch_geometric.nn as pyg_nn
from torch_geometric.nn.models.mlp import MLP as pyg_mlp
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from typing import Optional
import torch.nn.functional as F
# from torch import Tensor
# from torch.nn import Module

SINUSOIDAL_TIME_EMBED_MAX_T = 500

class NormLayer(nn.Module):
    def __init__(self, norm, in_channels, **kwargs):
        super().__init__()
        self.norm = norm
        self.in_channels = in_channels
        self.n_groups = kwargs.get('n_groups', 8)
        self.resolve_norm_layer(norm = self.norm, in_channels=in_channels, **kwargs)


    def resolve_norm_layer(self, norm, in_channels, **kwargs):
        n_groups = kwargs.get('n_groups', 8)

        if norm == 'batch':
            self.norm_layer = BatchNorm(in_channels)
        elif norm == 'layer':
            self.norm_layer = LayerNorm(in_channels, mode = 'node')
        elif norm == 'group':
            self.norm_layer = DiffGroupNorm(in_channels=in_channels, groups=n_groups)
        elif norm == 'graph':
            self.norm_layer = GraphNorm(in_channels=in_channels)
        elif norm == 'none' or norm is None:
            self.norm_layer = nn.Identity()


    def forward(self, x, batch = None, batch_size = None):
        if self.norm in ['batch', 'layer', 'group', 'none', None]:
            return self.norm_layer(x)
        elif self.norm in ['graph']:
            # print('x.shape: ', x.shape)
            # print('batch.shape: ', batch.shape)
            return self.norm_layer(x, batch = batch, batch_size = batch_size)
        else:
            raise NotImplementedError
    

class SinusoidalTimeEmbedding(nn.Module):
    """
    https://nn.labml.ai/diffusion/ddpm/unet.html 
    """
    def __init__(self, n_channels: int):
        super().__init__()
        self.n_channels = n_channels
        # self.act = Swish()
        self.act = nn.LeakyReLU()

        self.lin_embed = nn.Sequential(nn.Flatten(start_dim=-2),
                                       nn.Linear(self.n_channels // 4, self.n_channels),
                                       self.act,
                                       nn.Linear(self.n_channels, self.n_channels)
                                       )
        
    def forward(self, t: torch.Tensor):
        half_dim = self.n_channels // 8
        emb = torch.log(torch.Tensor([SINUSOIDAL_TIME_EMBED_MAX_T])) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device = t.device) * -emb.to(t.device))
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat((torch.sin(emb), torch.cos(emb)), dim = -1)

        emb = self.lin_embed(emb)
        return emb
    

class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()

        architecture = kwargs.get('architecture', 'GCNConv')
        k_hops = kwargs.get('k_hops', 2)
        normalize = kwargs.get('normalize', False)
        add_self_loops = kwargs.get('add_self_loops', True)

        if architecture == 'TAGConv':
            self.conv_layer = TAGConv(in_channels=in_channels, out_channels=out_channels, K=k_hops, normalize = normalize)
            print(f"TAGConv is initialized with F_l={in_channels}, F_(l+1)={out_channels}, K_hops = {k_hops}, normalization = {normalize}.")
        elif architecture == 'LEConv':
            self.conv_layer = LEConv(in_channels=in_channels, out_channels=out_channels)
        elif architecture == 'GCNConv':
            self.conv_layer = GCNConv(in_channels=in_channels, out_channels=out_channels,
                                      improved=True, add_self_loops=add_self_loops, normalize=normalize if add_self_loops is False else True)
        else:
            raise ValueError
    
    def forward(self, x, edge_index, edge_weight, batch = None):
        x = self.conv_layer(x, edge_index = edge_index, edge_weight = edge_weight)
        return x



class ResidualLayer(nn.Module):
    r"""
    The DeepGCN with residual connections block adapted to my implementation:

    The skip connection operations from the
    `"DeepGCNs: Can GCNs Go as Deep as CNNs?"
    <https://arxiv.org/abs/1904.03751>`_ and `"All You Need to Train Deeper
    GCNs" <https://arxiv.org/abs/2006.07739>`_ papers.
    The implemented skip connections includes the pre-activation residual
    connection (:obj:`"res+"`), the residual connection (:obj:`"res"`),
    the dense connection (:obj:`"dense"`) and no connections (:obj:`"plain"`).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        conv_architecture: str = "TAGConv",
        k_hops: int = 2,
        norm: str = "batch",
        mlp: Optional[nn.Module] = None,
        act: Optional[nn.Module] = nn.LeakyReLU(),
        res_connection: str = 'res+',
        dropout: float = 0.,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.k_hops = k_hops
        self.conv_architecture = conv_architecture
        assert self.conv_architecture in ['LeConv', 'TAGConv', 'GCN']
        self.norm = norm
        assert self.norm in ['batch', 'group', 'graph', 'layer', 'none']
        self.act = act
        self.res_connection = res_connection.lower()
        assert self.res_connection in ['res+', 'res', 'dense', 'plain']
        self.mlp = mlp
        self.dropout = dropout
        self.use_checkpointing = use_checkpointing

        ### Initialize the convolution layer
        self.conv = ConvLayer(in_channels=in_channels, out_channels=out_channels,
                              architecture = self.conv_architecture, k_hops=self.k_hops,
                              normalize = False,
                              add_self_loops = False
                              )
            
        self.mlp = nn.Identity() if self.mlp is None else self.mlp
        self.norm = NormLayer(norm=norm, in_channels=in_channels)

        if self.in_channels != self.out_channels:
            self.res = nn.Linear(self.in_channels, self.out_channels, bias=False)
        else:
            self.res = nn.Identity()
            


    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.conv.reset_parameters()
        self.norm.reset_parameters() if self.norm is not None else None
        self.mlp.reset_parameters() if self.mlp is not None else None


    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor, batch: torch.Tensor = None):        

        if self.res_connection == 'res+':
            h = x
            if self.norm is not None:
                h = self.norm(h, batch = batch, batch_size = None) # batch size computation TO DO if GraphNorm is used.

                norms = [{"layer": "N", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).mean().item()},
                         {"layer": "N-var", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).var().item()}]

            if self.act is not None:
                h = self.act(h)
                norms += [{"layer": "A", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).mean().item()},
                          {"layer": "A-var", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).var().item()}]
                
            h = F.dropout(h, p=self.dropout, training=self.training)

            if self.use_checkpointing:
                h = checkpoint.checkpoint(self.custom_conv_embed(self.conv), h, edge_index, edge_weight, use_reentrant=False)
            else:
                h = self.conv(h, edge_index=edge_index, edge_weight=edge_weight)
            
            norms += [{"layer": "C", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).mean().item()},
                      {"layer": "C-var", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).var().item()}]
            
            h = self.mlp(h)
            # assert h.size(1) == x.size(1), f"Hidden channels mismatch. {h.size(1)}-{x.size(1)}"
            h = h + self.res(x)  # residual connection

            norms += [{"layer": "O", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).mean().item()},
                      {"layer": "O-var", "norm": torch.linalg.vector_norm(h, ord = 2, dim = 1).var().item()}]

            return h, norms

        else:
            
            raise NotImplementedError
        

            # if self.conv is not None and self.ckpt_grad and x.requires_grad:
            #     h = checkpoint(self.conv, x, *args, use_reentrant=True,
            #                    **kwargs)
            # else:
            #     h = self.conv(x, *args, **kwargs)
            # if self.norm is not None:
            #     h = self.norm(h)
            # if self.act is not None:
            #     h = self.act(h)

            # if self.block == 'res':
            #     h = x + h
            # elif self.block == 'dense':
            #     h = torch.cat([x, h], dim=-1)
            # elif self.block == 'plain':
            #     pass

            # return F.dropout(h, p=self.dropout, training=self.training)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(res={self.res_connection})'


class ResidualGNN(nn.Module):
    def __init__(self,
                in_channels: int,
                hidden_channels: int,
                out_channels: int,
                depth: int,
                norm: str = 'batch',
                **kwargs
                ):
        super(ResidualGNN, self).__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        # self.out_channels = out_channels  in_channels
        self.depth = depth 

        self.conv_layer_normalize = kwargs.get("conv_layer_normalize", False)
        self.k_hops = kwargs.get("k_hops", 2)
        self.res_connection = kwargs.get("res_connection", "res+")
        self.use_checkpointing = kwargs.get("use_checkpointing", False)

        self.activation = nn.LeakyReLU()
        self.norm = norm
        conv_architecture = "TAGConv"
        self.time_embed = SinusoidalTimeEmbedding(n_channels=hidden_channels // 4)
        self.x_embed = nn.Linear(in_channels, hidden_channels // 4)

        self.res_conv_layers = nn.ModuleList()
        # self.time_embed_layers = [self.time_embed] # nn.ModuleList()
        # self.x_embed_layers = [self.x_embed]

        self.res_conv_layers.append(ResidualLayer(in_channels=hidden_channels // 4,
                                                  out_channels=hidden_channels,
                                                  conv_architecture=conv_architecture,
                                                  k_hops=self.k_hops,
                                                  norm=self.norm,
                                                  mlp=pyg_mlp(channel_list = [hidden_channels, 4 * hidden_channels, hidden_channels], act = self.activation),
                                                  res_connection=self.res_connection,
                                                  dropout=0.0,
                                                  use_checkpointing=self.use_checkpointing
                                                  )
                                    )
        
        # hidden_channels = self.res_conv_layers[-1].out_channels
        # print(f"Hidden channels = {self.hidden_channels} increased from 1st layer to: ", hidden_channels)

        # self.norm_layers.append(NormLayer(norm=self.norm, in_channels=self.hidden_channels))

        # self.time_embed_layers.append(nn.Sequential(nn.Linear(self.in_channels, self.hidden_channels), nn.LeakyReLU()))
        for i in range(1, self.depth):
            # self.time_embed_layers.append(nn.Identity())
            # self.x_embed_layers.append(nn.Identity())

            res_conv_layer = ResidualLayer(in_channels=hidden_channels,
                                           out_channels=hidden_channels,
                                           conv_architecture=conv_architecture,
                                           k_hops=self.k_hops,
                                           norm=self.norm,
                                           mlp=pyg_mlp(channel_list = [hidden_channels, 4 * hidden_channels, hidden_channels], act = self.activation),
                                           res_connection=self.res_connection,
                                           dropout=0.0,
                                           use_checkpointing=self.use_checkpointing
                                           ) 

            self.res_conv_layers.append(res_conv_layer)

        # assert len(self.res_conv_layers) == len(self.time_embed_layers) == len(self.x_embed_layers) == self.depth, \
        #     f"Mismatch in the number of layers. {len(self.res_conv_layers)}-res-conv-layers, {len(self.time_embed_layers)}-time-embed-layers, {len(self.x_embed_layers)}-x-embed-layers."
        
        self.out_layers = nn.Linear(hidden_channels, self.in_channels)

        # Apply custom weight initialization
        self.apply(self.init_weights)
        self.n_iters_trained = 0


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



    def forward(self, x, edge_index, edge_weight=None, batch=None, t=None, return_attn_weights=False, debug_forward_pass=False):

        if self.conv_layer_normalize:
            try:
                add_self_loops = True
                improved = add_self_loops
                edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(0),
                        improved=improved, add_self_loops=add_self_loops, # flow=self.conv_layers[0].flow,
                        dtype=x.dtype)
                print("GCN normalization applied to edge_weights.") if self.n_iters_trained == 0 else None

            except:
                print("GCN normalization failed.") if self.n_iters_trained == 0 else None

        all_norms = []

        x = self.x_embed(x)
        t = self.time_embed(t)
        assert x.size(1) == t.size(1), f"Hidden channels mismatch. {x.size(1)}-{t.size(1)}" 

        x = x + t # add time embeddings only to the input layer
        x, norms = self.res_conv_layers[0](x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)

        # Compute individual layer statistics and label them by layer_id.
        for norm in norms:
            norm["layer"] = f"0{norm['layer']}"
        all_norms += norms

        for layer_id, conv_layer in enumerate(self.res_conv_layers[1:], start = 1):
            
            # if layer_id == 0:
            #     x = x_embed_layer(x)
            #     t = t_embed_layer(t)
            #     assert x.size(1) == t.size(1), f"Hidden channels mismatch. {x.size(1)}-{t.size(1)}" 
            # # assert t.size(1) == self.hidden_channels, f"Hidden channels mismatch. {t.size(1)}-{self.hidden_channels}"

            # x = x + (layer_id == 0) * t # add time embeddings only to the input layer

            x, norms = conv_layer(x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)

            # Compute individual layer statistics and label them by layer_id.
            for norm in norms:
                norm["layer"] = f"{layer_id}{norm['layer']}"
            all_norms += norms

        out = self.out_layers(x)
        all_norms += [{"layer": "-1", "norm": torch.linalg.vector_norm(out, ord = 2, dim = 1).mean().item()},
                  {"layer": "-1-var", "norm": torch.linalg.vector_norm(out, ord = 2, dim = 1).var().item()}]

        attn_weights = None

        if debug_forward_pass:
            debug_data = {
                        #   'edge_indices': [e.detach().cpu() for e in edge_indices],
                          'edge_indices': [edge_index.detach().cpu()],
                        #   'edge_weights': [w.detach().cpu() for w in edge_weights],
                          'edge_weights': [edge_weight.detach().cpu()],
                          'batches': [batch.detach().cpu()],
                          'model_layer_norms': all_norms
                          }
        else:
            debug_data = None

        self.n_iters_trained += 1
        return out, attn_weights, debug_data
    

    def init_weights(self, m):
        """ Custom weight initialization. """
        if isinstance(m, pyg_nn.TAGConv):
            # print("m: ", m)
            # Apply He initialization to the weight matrix in GraphConv
            for lin in m.lins:
                if self.activation._get_name() == "LeakyReLU":
                    torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity='leaky_relu')  # He init for weight
                elif self.activation._get_name() == 'ReLU':
                    torch.nn.init.kaiming_uniform_(lin.weight, nonlinearity='relu')  # He init for weight
                else:
                    pass
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)  # Bias is initialized to zero
        elif isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)  # Xavier init for final layers
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)