import os
import subprocess
from train_gnn import DATASET


# Convert gene unit parameters (gene_units_params) into generated model code.
def gene_to_code(gene_units_params):
    gene_units_params = optim_gene_code(gene_units_params)
    gene_length = 0
    space_8 = "        "

    class_head = '''class GEN_GNN_MODEL(torch.nn.Module):
    def __init__(self):
        super(GEN_GNN_MODEL, self).__init__()'''

    gnn_layers = ""

    forward_code = ""

    pos = 0
    channel_record = []

    for unit_params in gene_units_params:
        gene_length = gene_length + 1

        if unit_params[0] == '-1' or unit_params[0] == '-2':
            pos = pos + 1
            gene_length = gene_length - 1
            continue

        if unit_params[0] == '0':
            channel_record.append((pos, DATASET.in_channels))
            pos = pos + 1
            continue

        if unit_params[0] == '255':
            forward_code += space_8 + f"return torch.sigmoid(x_{pos-1})\n"
            break
        input_num = int(unit_params[0])

        if input_num == 1:
            input_offset = int(unit_params[1])
            input_pos = max(0, pos - input_offset)

            in_channels = next(ch for p, ch in channel_record if p == input_pos)

            input_name = f'x_{input_pos}'
            gnn_type = int(unit_params[2])
            out_channels = unit_params[3]
            agg_param = unit_params[4]

            dropout_prob = float(unit_params[5]) / 100.0
            if gnn_type == 1:
                gnn_layers += space_8 + f"self.gnn_{pos} = GCNConv(in_channels={in_channels}, out_channels={out_channels})\n"
            elif gnn_type == 2:
                gnn_layers += space_8 + f"self.gnn_{pos} = GATConv(in_channels={in_channels}, out_channels={out_channels}, heads={agg_param},concat=False,dropout={dropout_prob})\n"
            elif gnn_type == 3:
                gnn_layers += space_8 + f"self.gnn_{pos} = SAGEConv(in_channels={in_channels}, out_channels={out_channels})\n"
            elif gnn_type == 4:
                gnn_layers += space_8 + f"self.gnn_{pos} = GATv2Conv(in_channels={in_channels}, out_channels={out_channels}, heads={agg_param}, concat=False, dropout={dropout_prob})\n"
            elif gnn_type == 5:
                agg_param_K = str(2 + (int(agg_param) % 3))
                gnn_layers += space_8 + f"self.gnn_{pos} = ChebConv(in_channels={in_channels}, out_channels={out_channels}, K={agg_param_K})\n"
            elif gnn_type == 6:
                mlp_mul = 1 + (int(agg_param) % 3)
                mlp_hidden = int(out_channels) * mlp_mul
                gnn_layers += space_8 + (
                    f"self.mlp_{pos} = nn.Sequential("
                    f"nn.Linear({in_channels}, {mlp_hidden}), "
                    f"nn.ReLU(), "
                    f"nn.Linear({mlp_hidden}, {out_channels})"
                    f")\n"
                )
                gnn_layers += space_8 + f"self.gnn_{pos} = GINConv(self.mlp_{pos})\n"
            elif gnn_type == 7:
                arma_stacks = 1 + (int(agg_param) % 3)
                arma_layers = 2
                gnn_layers += space_8 + (
                    f"self.gnn_{pos} = ARMAConv(in_channels={in_channels}, out_channels={out_channels}, "
                    f"num_stacks={arma_stacks}, num_layers={arma_layers}, dropout={dropout_prob})\n"
                )
            elif gnn_type == 8:
                gnn_layers += space_8 + f"self.gnn_{pos} = nn.Identity()\n"
            elif gnn_type == 9:
                gnn_layers += space_8 + f"self.gnn_{pos} = nn.Linear({in_channels}, {out_channels})\n"

            channel_record.append((pos, int(out_channels)))

            activation_num = int(unit_params[6])
            if activation_num == 1:
                activation = "F.relu"
            elif activation_num == 2:
                activation = "F.elu"
            elif activation_num == 3:
                activation = "F.leaky_relu"
            else:
                activation = "None"

            if gnn_type in (8, 9):
                if activation != "None":
                    forward_code += space_8 + f"x_{pos} = {activation}(self.gnn_{pos}({input_name}))\n"
                else:
                    forward_code += space_8 + f"x_{pos} = self.gnn_{pos}({input_name})\n"
            else:
                if activation != "None":
                    forward_code += space_8 + f"x_{pos} = {activation}(self.gnn_{pos}({input_name}, edge_index))\n"
                else:
                    forward_code += space_8 + f"x_{pos} = self.gnn_{pos}({input_name}, edge_index)\n"

            if pos != len(gene_units_params) - 2:
                forward_code += f"{space_8}x_{pos} = F.dropout(x_{pos}, p={dropout_prob}, training=self.training)\n"
        elif input_num == 2:
            input_offset_a = int(unit_params[1])
            input_offset_b = int(unit_params[2])
            input_pos_a = max(0, pos - input_offset_a)
            input_pos_b = max(0, pos - input_offset_b)

            in_channels_a = next(ch for p, ch in channel_record if p == input_pos_a)
            in_channels_b = next(ch for p, ch in channel_record if p == input_pos_b)

            in_channels = in_channels_a if in_channels_a == in_channels_b else in_channels_a

            input_name_a = f'x_{input_pos_a}'
            input_name_b = f'x_{input_pos_b}'

            unit_type = int(unit_params[3])
            if unit_type == 2:
                forward_code += space_8 + f"x_{pos} = {input_name_a} + {input_name_b}\n"
                channel_record.append((pos, in_channels))
            elif unit_type == 3:
                forward_code += space_8 + f"x_{pos} = torch.cat([{input_name_a}, {input_name_b}], dim=1)\n"

                channel_record.append((pos, in_channels_a + in_channels_b))
        pos = pos + 1

    code = f"{class_head}\n{gnn_layers}\n    def forward(self, x_0, edge_index):\n{forward_code}"
    return code, gene_length


# Prune unused gene units before code generation.
def optim_gene_code(gene_units_params):
    used = [False] * len(gene_units_params)

    used[0] = True
    used[-1] = True

    def mark_used(pos):
        """Recursively mark the current gene unit and its dependencies as used."""
        if pos < 0 or pos >= len(gene_units_params) or used[pos]:
            return

        used[pos] = True
        unit_params = gene_units_params[pos]

        if unit_params[0] == '0' or unit_params[0] == '255':
            return

        input_num = int(unit_params[0])
        if input_num == 1:
            input_offset = int(unit_params[1])
            input_pos = max(0, pos - input_offset)
            mark_used(input_pos)
        elif input_num == 2:
            input_offset_a = int(unit_params[1])
            input_offset_b = int(unit_params[2])
            input_pos_a = max(0, pos - input_offset_a)
            input_pos_b = max(0, pos - input_offset_b)
            mark_used(input_pos_a)
            mark_used(input_pos_b)

    for i in range(len(gene_units_params) - 2, 0, -1):
        unit_params = gene_units_params[i]

        if i == len(gene_units_params) - 2:
            mark_used(i)

    changed = True
    while changed:
        changed = False
        for i in range(len(gene_units_params)):
            if used[i]:
                unit_params = gene_units_params[i]

                if unit_params[0] == '0' or unit_params[0] == '255':
                    continue

                input_num = int(unit_params[0])
                if input_num == 1:
                    input_offset = int(unit_params[1])
                    input_pos = max(0, i - input_offset)
                    if not used[input_pos]:
                        used[input_pos] = True
                        changed = True
                elif input_num == 2:
                    input_offset_a = int(unit_params[1])
                    input_offset_b = int(unit_params[2])
                    input_pos_a = max(0, i - input_offset_a)
                    input_pos_b = max(0, i - input_offset_b)
                    if not used[input_pos_a]:
                        used[input_pos_a] = True
                        changed = True
                    if not used[input_pos_b]:
                        used[input_pos_b] = True
                        changed = True

    for i in range(len(gene_units_params)):
        if not used[i]:
            unit_params = gene_units_params[i]
            if unit_params[0] == '2':
                gene_units_params[i][0] = '-2'
            elif unit_params[0] not in ['0', '255']:
                gene_units_params[i][0] = '-1'

    return gene_units_params


# Write generated neural-network model code to a local Python module.
def code_to_file(id: int, code: str):
    file_head = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GATv2Conv, ChebConv, GINConv, ARMAConv'''
    file_content = file_head + "\n\n\n" + code
    if not os.path.exists("./models/"):
        os.mkdir("./models/")

        with open("./models/__init__.py", "w", encoding="utf-8") as f:
            pass

    with open('./models/model_{}.py'.format(id), 'w', encoding='utf-8') as f:
        f.write(file_content)
    return 'model_{}.py'.format(id), str(id)
