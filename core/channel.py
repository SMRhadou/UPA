import numpy as np
from scipy.spatial import distance
import numpy.matlib
from core.links import drop_links
import torch
import networkx as nx

# Global parameters
min_D_TxTx = 30 # minimum Tx-Tx distance
min_D_TxRx = 20 # minimum Tx-Rx distance
max_D_TxRx = 60 # maximum Tx-Rx distance
shadowing = 7 # shadowing standard deviation
f_c = 2.4e9 # carrier frequency (Hz)
speed = 1 # receiver speed (m/s)

def UDN_PL(D, alpha1=2):
    m, n = np.shape(D)
    L = np.zeros((m, n))
    k0 = 39
    a1 = alpha1
    a2 = 4
    db = 100

    CONST = 10 * np.log10(db ** (a2-a1))
    
    for i in range(m):
        for j in range(n):
            d = D[i,j] 
            if d <= db:
                L[i,j] = k0 + 10 * a1 * np.log10(d)
            else:
                L[i,j] = k0 + 10 * a2 * np.log10(d) - CONST
    return L

def short_term_fading(T, N, f_c, speed):
    """ Rayleigh fading model via sum of sinusoids
    from "Model of independent Rayleigh faders" by Z. Wu
    """
    t_vec = np.array(range(T)) / 1e3
    a_0 = np.pi / (2*N)
    w_M = 2 * np.pi * f_c * speed / 3e8
    
    N0 = N // 4
    alpha = a_0 + 2 * np.pi * np.array(range(N0)) / N
    theta = 2 * np.pi * np.random.uniform(0, 1, (N0, 1))
    theta_p = 2 * np.pi * np.random.uniform(0, 1, (N0, 1))
    
    I = np.cos(w_M * np.outer(np.cos(alpha) , t_vec) + np.matlib.repmat(theta, 1, len(t_vec)))
    Q = np.sin(w_M * np.outer(np.sin(alpha) , t_vec) + np.matlib.repmat(theta_p, 1, len(t_vec)))
    
    h = sum(I + 1j * Q) / np.sqrt(N0)
    
    return h

def create_channel_matrix_over_time(m, n, T, R, graph_type='CR'):
    # # (Navid implementation) specify transmitter locations
    # while True:
    #     locTx = np.random.uniform(0, R, (m, 2)) - R / 2
    #     D_TxTx = distance.cdist(locTx, locTx, 'euclidean')
    #     for Tx in range(m):
    #         D_TxTx[Tx, Tx] = float('Inf')
    #     if np.min(D_TxTx) >= min_D_TxTx:
    #         break
            
    # # specify receiver locations
    # while True:
    #     phi = 2 * np.pi * np.random.uniform(n)
    #     r = np.sqrt(np.random.uniform(low=min_D_TxRx ** 2, high=max_D_TxRx ** 2, size=(n,)))
    #     locRx = np.clip(locTx + np.stack((r * np.cos(phi), r * np.sin(phi)), axis=1), a_min= -R / 2, a_max=R / 2)
        
    #     D_TxRx = distance.cdist(locTx, locRx, 'euclidean')
    #     if np.min(D_TxRx) < min_D_TxRx:
    #         continue

    #     L = UDN_PL(D_TxRx) + shadowing * np.random.randn(m, n) # Loss matrix in dB
    #     H_l = np.sqrt(np.power(10, -L / 10)) # large-scale fading matrix
    #     associations = (H_l == np.max(H_l, axis=0, keepdims=True))
    #     if min(np.sum(associations, axis=1)) > 0: # each transmitter has at least one associated reciever
    #         break

    if graph_type == 'CR':
        locTx, locRx = drop_links(m, min_D_TxRx, max_D_TxRx, R, min_D_TxTx)
    elif graph_type == 'regular':
        locTx, locRx = create_grid_channel_matrix(m, n, R)
    
    
    D_TxTx = distance.cdist(locTx, locTx, 'euclidean')
    for Tx in range(m):
        D_TxTx[Tx, Tx] = float('Inf')
    D_TxRx = distance.cdist(locTx, locRx, 'euclidean')

    L = UDN_PL(D_TxRx) + shadowing * np.random.randn(m, n) # Loss matrix in dB
    H_l = np.sqrt(np.power(10, -L / 10)) # large-scale fading matrix
    associations = (H_l == np.max(H_l, axis=0, keepdims=True))

    H = np.zeros((m, n, T), dtype=complex)
    for i in range(m):
        for j in range(n):
            H[i, j] = short_term_fading(T, 100, f_c, speed)
            
    H *= np.expand_dims(H_l, axis=2)
    
    return np.abs(H) ** 2, np.abs(H_l) ** 2


def create_grid_channel_matrix(m, n, grid_size=10, connection_radius=None):
    """
    Creates a grid-based channel matrix where transmitters and receivers are placed
    at the center of grid cells and connected based on distance.
    
    Args:
        m: Number of transmitters
        n: Number of receivers (should equal m for a grid layout)
        grid_size: Size of the grid (grid_size x grid_size)
        connection_radius: Radius for connecting points. If None, it will be 
                           calculated to achieve approximately 4 neighbors per node
    
    Returns:
        h_l: Channel gain matrix as torch tensor (m x n)
        positions: Dictionary of node positions {node_id: (x,y)}
    """
    # if n != grid_size**2:
    #     print(f"Warning: Number of receivers ({n}) doesn't match grid size ({grid_size}x{grid_size}={grid_size**2})")
    R = grid_size//np.sqrt(m)
    # Create points at the center of each grid cell
    points = []
    positions = {}
    
    for i in range(int(np.sqrt(n))):
        for j in range(int(np.sqrt(m))):
            x = (i + 0.5)* R  # Center of cell
            y = (j + 0.5) * R  # Center of cell
            points.append((x, y))
            positions[len(points)-1] = (x, y)
    
    total_nodes = len(points)
    locTx = np.stack(points)
    # permute the transmitters
    locTx = np.random.permutation(locTx)  
    locTx += np.random.uniform(-R/20, R/20, locTx.shape)  # Add some noise to Tx locations

    # drop Rx within min_D_TxRx from each Tx
    locRx = np.zeros((m, 2))
    for i in range(m): 
        while True:
            phi = 2 * np.pi * np.random.uniform()
            r = np.sqrt(np.random.uniform(low=min_D_TxRx ** 2, high=max_D_TxRx ** 2, size=(1,)))
            locRx[i] = np.clip(locTx[i] + np.stack((r * np.cos(phi), r * np.sin(phi)), axis=0).squeeze(), a_min=0, a_max=grid_size)
            D_TxRx = distance.cdist(locTx, locRx, 'euclidean')
            if np.min(D_TxRx) >= min_D_TxRx:
                break


    # # Calculate appropriate connection radius if not provided
    # if connection_radius is None:
    #     # For a grid, a radius of ~1.5 cell units should give approximately 4 neighbors
    #     connection_radius = 1.1 * R
    
    # # Create adjacency matrix with channel gains based on distance
    # h_l = torch.zeros((total_nodes, total_nodes))
    
    # # Connect nodes within the radius with distance-based channel gains
    # for i in range(total_nodes):
    #     for j in range(total_nodes):
    #         if i != j:  # No self-loops
    #             x1, y1 = points[i]
    #             x2, y2 = points[j]
    #             dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    #             if dist <= connection_radius:
    #                 # Channel gain model: simplified path loss model
    #                 # h_ij = distance^(-alpha) where alpha is path loss exponent
    #                 alpha = 3.0  # Path loss exponent
    #                 h_l[i, j] = dist**(-alpha)
    # h_l = h_l * (1 + 0.1 * np.random.randn(m, n))
    
    # # For visualization/debug - create networkX graph
    # G = nx.Graph()
    # for i in range(total_nodes):
    #     G.add_node(i)
    
    # for i in range(total_nodes):
    #     for j in range(i+1, total_nodes):
    #         if h_l[i, j] > 0:
    #             G.add_edge(i, j)
    
    # # Calculate average degree for verification
    # avg_degree = sum(dict(G.degree()).values()) / len(G.nodes())
    # print(f"Average node degree: {avg_degree:.2f}")
    
    return locTx, locRx#h_l, h_l, positions, G
