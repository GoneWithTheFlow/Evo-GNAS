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


# Runtime and database configuration.
class Config:
    epochs = 1000
    lr = 0.005
    epochs_ds = 2000
    lr_ds = 0.005
    out = 100
    IterNum = 1
    patience = 2000
    tol = 1e-6
    weight_decay = 5e-4
    close_frac = 0.95
    best_known = {
        "G14": 3064,
        "G15": 3050,
        "G22": 13359,
        "G49": 6000,
        "G50": 5880,
        "G55": 10294,
        "G70": 9541,
    }
    in_channels_by_dataset = {
        "G14": 64,
        "G15": 64,
        "G22": 64,
        "G49": 64,
        "G50": 64,
        "G55": 64,
        "G70": 64,
    }
    current_dataset = "G14"
    path = "./data/G14.txt"

    DB_USER = "postgres"
    DB_PASSWORD = "123456"
    DB_HOST = "localhost"
    DB_PORT = 5432
    DB_NAME = "QUBO"

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
    in_channels = Config.in_channels_by_dataset[Config.current_dataset]
    out_channels  = 1

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Build a PyTorch Geometric edge index from the QUBO matrix.
def get_edge_index(allRows):
    G = nx.from_edgelist(allRows)
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


# Return supported GNN operation types and activation choices.
def get_gnn_params(in_features, n_nodes, MyGraphNetwork):
    net = MyGraphNetwork()
    dim_embedding = in_features
    net = net.type(dtype).to(device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(dtype).to(device)

    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, lr=Config.lr, weight_decay=Config.weight_decay)

    return net, embed, optimizer


# Train a generated GNN on a QUBO instance and return its score.
def run_gnn_training_qubo(
        embed,
        dgl_graph,
        q_torch,
        net,
        optimizer,
        edge_index,
        prob_threshold_scan_step=0.01,
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

    best_known_cut = Config.best_known[Config.current_dataset]
    close_threshold = Config.close_frac * best_known_cut
    early_stopping_enabled = True

    def scan_best_threshold(probs):
        """
        Scan candidate thresholds and find the best bitstring, cut value, and threshold.
        probs: node assignment probabilities from the GNN, shape [num_nodes].
        Return: best bitstring, best cut value, and best threshold.
        """
        probs_det = probs.detach()

        thresholds = torch.linspace(
            start=0.0,
            end=1.0,
            steps=int(1.0 / prob_threshold_scan_step) + 1,
            device=probs.device
        )

        threshold_bitstrings = (probs_det.unsqueeze(0) >= thresholds.unsqueeze(1)).float()
        threshold_cuts = -((threshold_bitstrings @ q_torch) * threshold_bitstrings).sum(dim=1)
        best_idx = int(torch.argmax(threshold_cuts).item())
        best_cut = float(threshold_cuts[best_idx].item())
        best_th = float(thresholds[best_idx].item())
        best_bs = threshold_bitstrings[best_idx]

        num_samples = 2048

        samples = torch.bernoulli(probs_det.expand(num_samples, -1)).float()

        sample_cuts = -((samples @ q_torch) * samples).sum(dim=1)
        sample_best_idx = int(torch.argmax(sample_cuts).item())
        sample_best_cut = float(sample_cuts[sample_best_idx].item())

        if sample_best_cut > best_cut:
            best_cut = sample_best_cut
            best_bs = samples[sample_best_idx]

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

            if best_cut >= close_threshold and early_stopping_enabled:
                early_stopping_enabled = False
                print(f"Reached {Config.close_frac * 100:.1f}% of SOTA ({best_cut:.0f}/{best_known_cut}), disabling early stopping")
        else:
            count += 1

        if early_stopping_enabled and count >= patience:
            print(f'Early stopping at epoch {epoch} (no best-solution improvement for {count} consecutive epochs)')
            break

        if epoch % out == 0:
            print(f'Epoch: {epoch}, Loss: {loss_val:.6f}, best loss: {best_loss:.6f}')
            losses.append(loss_val)
            epochs.append(epoch)

    t_gnn = time.time() - t_gnn_start

    final_bitstring, _, _ = scan_best_threshold(probs)
    print(f'Training took {t_gnn:.3f}s')
    print(f'Best loss: {best_loss:.6f}')

    return net, epoch, final_bitstring, best_bitstring, losses, epochs, t_gnn


# Import generated model code, train it, and return evaluation results.
def train(model_class=None, save_dir="./gnn_save"):
    data_path = Config.path
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        allRows = [list(map(int, row[0].split())) for row in reader]
    allRows = allRows[1:]
    for i in range(len(allRows)):
        del allRows[i][2]
    allRows = list(map(tuple, allRows))

    edge_index, graph_dgl, G, n_nodes = get_edge_index(allRows)
    all_edges = G.number_of_edges()

    Q = create_Q_matrix(G, is_max_cut=True).to(device)

    dim_embedding = Config.in_channels_by_dataset[Config.current_dataset]
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

        net, epoch, final_bitstring, best_bitstring, losses, epochs,runtime = run_gnn_training_qubo(
            embed=embed,
            dgl_graph=graph_dgl,
            q_torch=Q,
            net=net,
            optimizer=optimizer,
            edge_index=edge_index
        )
        run_times.append(runtime)

        best_bitstring = best_bitstring.to(device)

        best_bitstring_row = best_bitstring.unsqueeze(0)
        cut_value = -(best_bitstring_row @ Q @ best_bitstring_row.T).item()
        cut_vals.append(cut_value)

    best_cut = max(cut_vals)
    acc = best_cut / all_edges
    time = sum(run_times) / IterNum
    print(f'Best cut value: {best_cut}')
    print(f'Accuracy: {acc:.4f}')

    with open("experiment.txt", "a") as f:
        f.write(f"GEN_GNN_MODEL     {best_cut}     {acc:.4f}\n")

    os.makedirs(save_dir, exist_ok=True)
    torch.save({
        'net': net.state_dict(),
        'embed': embed.state_dict()
    }, f"{save_dir}/model.pt")
    final_model_dirname = f"best_cut_{best_cut}"

    return final_model_dirname,best_cut, acc, time
