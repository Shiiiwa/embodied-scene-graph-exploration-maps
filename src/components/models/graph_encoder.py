import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from torch_geometric.nn import HGTConv
from torch_geometric.data import HeteroData


class NodeEdgeGATEncoder(nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_channels, out_channels):
        super().__init__()
        # Edge attributes als zusätzliches Input
        self.gat1 = GATConv(in_channels, hidden_channels, edge_dim=edge_in_channels)
        self.gat2 = GATConv(hidden_channels, out_channels, edge_dim=edge_in_channels)
        self.edge_mlp = nn.Sequential(nn.Linear(edge_in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.gat1(x, edge_index, edge_attr)
        x = F.elu(x)
        x = self.gat2(x, edge_index, edge_attr)  # Node output: [num_nodes, out_channels]
        e = self.edge_mlp(edge_attr)  # Edge output: [num_edges, out_channels]
        return x, e, batch


class NodeEdgeHGTEncoder(nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_channels, out_channels, relation_types, num_heads=4, num_layers=2):
        super().__init__()
        # metadata für HGTConv (Knoten- und Kantentypen)
        node_types = ["object"]
        edge_types = [("object", rel, "object") for rel in relation_types]
        metadata = (node_types, edge_types)
        self.layers = nn.ModuleList()
        self.layers.append(HGTConv(in_channels=in_channels, out_channels=hidden_channels, metadata=metadata, heads=num_heads))
        for _ in range(num_layers - 1):
            self.layers.append(HGTConv(in_channels=hidden_channels, out_channels=hidden_channels, metadata=metadata, heads=num_heads))
        self.fc = nn.Linear(hidden_channels, out_channels)
        self.edge_mlp = nn.Sequential(nn.Linear(edge_in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels))

    def forward(self, data: "HeteroData"):
        x_dict, edge_index_dict, edge_attr_dict = data.x_dict, data.edge_index_dict, data.edge_attr_dict

        for layer in self.layers:
            x_dict = layer(x_dict, edge_index_dict)

        node_embeds = self.fc(x_dict["object"])  # [num_nodes, out_channels]

        edge_embeds = []
        for rel in edge_attr_dict:
            edge_attr = edge_attr_dict[rel]  # [num_edges_of_this_type, edge_in_channels]
            if edge_attr.numel() > 0:
                edge_embeds.append(self.edge_mlp(edge_attr))
        if edge_embeds:
            edge_embeds = torch.cat(edge_embeds, dim=0)  # [total_num_edges, out_channels]
        else:
            edge_embeds = torch.zeros((0, self.fc.out_features), device=node_embeds.device)

        return node_embeds, edge_embeds
