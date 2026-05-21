import time
import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv, SAGEConv,GATv2Conv,ChebConv
import networkx as nx
import jax.numpy as jnp
from networkx import convert_node_labels_to_integers
import dgl
import numpy as np
import scipy.sparse
from itertools import chain
from co_corefunc import loss_func
from co_corefunc import create_Q_matrix
from co_corefunc import repair_to_independent_set_tensor
from co_corefunc import count_selected_edge_conflicts_tensor
from co_corefunc import get_undirected_edge_pairs


# Runtime and database configuration.
class Config:
    epochs = 200
    lr = 0.005
    epochs_ds = 200
    lr_ds = 0.005
    out = 10
    IterNum = 1
    patience = 50
    tol = 1e-6
    num_samples = 64
    weight_decay = 5e-4
    current_dataset = "MIS_ER_n1000_p005_1"
    path = "./data-MIS/MIS_ER_n1000_p005_1.txt"

    DB_USER = "postgres"
    DB_PASSWORD = "123456"
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_NAME = "MIS"

    DB_POOL_RECYCLE = 60
    DB_POOL_PRE_PING = True
    DB_POOL_USE_LIFO = True
    DB_ECHO_POOL = True
    DB_POOL_SIZE = 2

    DB_CONN_STR = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    @property
    def db_conn_str(self):
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


# Dataset shape configuration.
class DATASET:
    in_channels = 10
    out_channels  = 1

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Load graph data into NetworkX, DGL, and PyTorch edge-index formats.
def get_edge_index(allRows, total_nodes=None):
    G = nx.Graph()
    G.add_edges_from(allRows)

    if total_nodes is not None:
        G.add_nodes_from(range(1, total_nodes + 1))
    G = convert_node_labels_to_integers(G, first_label=0, ordering="default", label_attribute=None)

    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))

    graph_dgl = dgl.from_networkx(nx_graph=H)
    graph_dgl = graph_dgl.to(device)

    n_nodes = len(G.nodes())
    adj_matrix = nx.adjacency_matrix(G)
    coo_matrix = scipy.sparse.coo_matrix(adj_matrix)

    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    edge_index_A = torch.LongTensor(indices)
    edge_index_A = edge_index_A.to(device)

    return edge_index_A, graph_dgl, G, n_nodes


# Create the model, node embedding, and optimizer.
def get_gnn_params(in_features, n_nodes, MyGraphNetwork):
    net = MyGraphNetwork()
    dim_embedding = in_features
    net = net.type(dtype).to(device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(dtype).to(device)

    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, lr=Config.lr, weight_decay=Config.weight_decay)

    return net, embed, optimizer


# Train the GNN against the QUBO objective.
def run_gnn_training_qubo(
        embed,
        dgl_graph,
    nx_graph,
        q_torch,
        net,
        optimizer,
        edge_index,
        edge_u,
        edge_v,
        prob_threshold_scan_step=0.02,
        scan_frequency=100,
        prob_threshold=0.5
):
    inputs = embed.weight
    data = torch_geometric.data.Data(x=inputs, edge_index=edge_index)
    prev_loss = 1.0
    count = 0
    patience = Config.patience
    tol = Config.tol
    number_epochs = Config.epochs
    out = Config.out

    losses = []
    epochs = []

    best_bitstring = torch.zeros((dgl_graph.number_of_nodes(),), dtype=dtype).to(device)
    best_loss = loss_func(best_bitstring.float(), q_torch)
    last_best_threshold = prob_threshold

    def scan_best_threshold(probs):
        """
        Scan candidate thresholds and find the best bitstring, MIS objective, and threshold.
        probs: node selection probabilities from the GNN, shape [num_nodes].
        Return: best bitstring, best MIS objective, and best threshold.
        """

        thresholds = torch.linspace(
            start=0.0,
            end=1.0,
            steps=int(1.0 / prob_threshold_scan_step) + 1,
            device=probs.device
        )
        best_cut = -float('inf')
        best_th = prob_threshold
        best_bs = None

        for th in thresholds:
            bitstring = (probs.detach() >= th).float()
            bitstring = repair_to_independent_set_tensor(bitstring, edge_u, edge_v, scores=probs)
            bitstring_row = bitstring.unsqueeze(0)
            cut_value = -(bitstring_row @ q_torch @ bitstring_row.T).item()
            if cut_value > best_cut:
                best_cut = cut_value
                best_th = th.item()
                best_bs = bitstring

        num_samples = Config.num_samples

        samples = torch.bernoulli(probs.repeat(num_samples, 1)).float().to(device)

        for i in range(num_samples):
            sample_bs = repair_to_independent_set_tensor(samples[i], edge_u, edge_v, scores=probs)
            sample_row = sample_bs.unsqueeze(0)
            sample_cut = -(sample_row @ q_torch @ sample_row.T).item()
            if sample_cut > best_cut:
                best_cut = sample_cut
                best_bs = sample_bs

        return best_bs, best_cut, best_th

    t_gnn_start = time.time()
    for epoch in range(number_epochs):
        net.train()
        optimizer.zero_grad()

        probs = net(data.x, data.edge_index)[:, 0]

        if (epoch % scan_frequency == 0) or (epoch == 0):
            current_bitstring, current_cut, last_best_threshold = scan_best_threshold(probs)
            current_loss = -current_cut
        else:
            current_bitstring = (probs.detach() >= last_best_threshold).float()
            current_bitstring = repair_to_independent_set_tensor(current_bitstring, edge_u, edge_v, scores=probs)

            bitstring_row = current_bitstring.unsqueeze(0)
            current_cut = -(bitstring_row @ q_torch @ bitstring_row.T).item()
            current_loss = -current_cut

        loss = loss_func(probs, q_torch)
        loss_val = loss.item()

        loss.backward()
        optimizer.step()

        if current_loss < best_loss:
            best_loss = current_loss
            best_cut = current_cut
            best_bitstring = current_bitstring
            count = 0
        else:
            count += 1

        if count >= patience:
            print(f'Early stopping at epoch {epoch} (no best-solution improvement for {count} consecutive epochs)')
            break

        if epoch % out == 0:
            conflicts = count_selected_edge_conflicts_tensor(current_bitstring, edge_u, edge_v)
            print(f'Epoch: {epoch}, Loss: {loss_val:.6f}, best loss: {best_loss:.6f}, conflicts: {conflicts}')
            losses.append(loss_val)
            epochs.append(epoch)

    t_gnn = time.time() - t_gnn_start

    final_bitstring, _, _ = scan_best_threshold(probs)
    print(f'Training took {t_gnn:.3f}s')
    print(f'Best loss: {best_loss:.6f}')

    return net, epoch, final_bitstring, best_bitstring, losses, epochs, t_gnn


# Train and evaluate one generated model.
def train(model_class=None, save_dir="./gnn_save"):
    data_path = Config.path
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        allRows = [list(map(int, row[0].split())) for row in reader]
    total_nodes = allRows[0][0]
    allRows = allRows[1:]
    for i in range(len(allRows)):
        del allRows[i][2]
    allRows = list(map(tuple, allRows))

    edge_index, graph_dgl, G, n_nodes = get_edge_index(allRows, total_nodes=total_nodes)

    Q = create_Q_matrix(G, is_max_cut=False).to(device)

    dim_embedding = DATASET.in_channels
    in_features = dim_embedding
    IterNum = Config.IterNum

    net, embed, optimizer = get_gnn_params(
        in_features=in_features,
        n_nodes=n_nodes,
        MyGraphNetwork=model_class
    )

    cut_vals = []
    run_times = []
    for i in range(IterNum):
        print(f'Running iteration {i + 1}/{IterNum}')

        net, embed, optimizer = get_gnn_params(
            in_features=in_features,
            n_nodes=n_nodes,
            MyGraphNetwork=model_class
        )

        edge_u, edge_v = get_undirected_edge_pairs(edge_index)

        net, epoch, final_bitstring, best_bitstring, losses, epochs,runtime = run_gnn_training_qubo(
            embed=embed,
            dgl_graph=graph_dgl,
            nx_graph=G,
            q_torch=Q,
            net=net,
            optimizer=optimizer,
            edge_index=edge_index,
            edge_u=edge_u,
            edge_v=edge_v
        )
        run_times.append(runtime)

        best_bitstring = repair_to_independent_set_tensor(best_bitstring.to(device), edge_u, edge_v)
        mis_size = int(best_bitstring.sum().item())
        conflicts = count_selected_edge_conflicts_tensor(best_bitstring, edge_u, edge_v)
        print(f'Iteration {i + 1} repaired conflicts: {conflicts}')
        cut_vals.append(mis_size)

    best_mis_size = max(cut_vals)
    acc = best_mis_size / n_nodes
    time = sum(run_times) / IterNum
    print(f'Best MIS size: {best_mis_size}')
    print(f'MIS ratio (size/n_nodes): {acc:.4f}')

    with open("experiment.txt", "a") as f:
        f.write(f"GEN_GNN_MODEL     {best_mis_size}     {acc:.4f}\n")

    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'net': net.state_dict(),
        'embed': embed.state_dict()
    }, f"{save_dir}/model.pt")
    final_model_dirname = f"best_mis_{best_mis_size}"

    return final_model_dirname, best_mis_size, acc, time
