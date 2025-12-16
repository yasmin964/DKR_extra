import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv


class DenseBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout_rate: float):
        super(DenseBlock, self).__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(num_features=in_channels),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features=in_channels, out_features=out_channels)
        )

    def forward(self, x: torch.FloatTensor):
        return self.block(x)


class GATModel(nn.Module):
    def __init__(self, args, num_nodes: int):
        super(GATModel, self).__init__()
        self.args = args

        # Use only embeddings (without initial features)
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim

        # Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=num_nodes,
            embedding_dim=args.embedding_dim
        )

        # First GAT layer: from embedding_dim to hidden_dim
        self.gat1 = GATConv(
            in_feats=args.embedding_dim,
            out_feats=args.hidden_dim // args.num_heads,
            num_heads=args.num_heads,
            feat_drop=args.gnn_dropout_rate,
            attn_drop=args.gnn_dropout_rate,
            activation=F.elu,
            residual=True
        )

        # Second layer
        self.gat2 = GATConv(
            in_feats=args.hidden_dim,
            out_feats=args.hidden_dim,
            num_heads=1,  # for the final representation
            feat_drop=args.gnn_dropout_rate,
            attn_drop=args.gnn_dropout_rate,
            activation=F.elu,
            residual=True
        )

        # Normalisation
        self.bn1 = nn.BatchNorm1d(num_features=args.hidden_dim)
        self.bn2 = nn.BatchNorm1d(num_features=args.hidden_dim)

        # MLP (Multi-Layer Perceptron) part for link prediction
        # Input: concatenated features from both nodes + any pairwise features        mlp_input_dim = args.hidden_dim * 2 + args.num_pairwise_features
        mlp_input_dim = args.hidden_dim * 2 + args.num_pairwise_features
        self.dense_1 = DenseBlock(
            in_channels=mlp_input_dim,
            out_channels=args.dnn_hidden_dim,
            dropout_rate=args.dnn_dropout_rate
        )

        # Dense layers with "DenseNet" style connections
        # Each layer's output is concatenated to the input for the next layer
        self.dense_2 = DenseBlock(
            in_channels=mlp_input_dim + args.dnn_hidden_dim,
            out_channels=args.dnn_hidden_dim,
            dropout_rate=args.dnn_dropout_rate
        )


        self.dense_3 = DenseBlock(
            in_channels=mlp_input_dim + args.dnn_hidden_dim * 2,
            out_channels=args.dnn_hidden_dim,
            dropout_rate=args.dnn_dropout_rate
        )

        self.dense_4 = DenseBlock(
            in_channels=mlp_input_dim + args.dnn_hidden_dim * 3,
            out_channels=args.dnn_hidden_dim,
            dropout_rate=args.dnn_dropout_rate
        )

        # Final layer outputs a single score for link prediction
        self.dense_5 = DenseBlock(
            in_channels=mlp_input_dim + args.dnn_hidden_dim * 4,
            out_channels=1,
            dropout_rate=args.dnn_dropout_rate
        )

    def forward(self, graph, node_features, vertex_pairs, pairwise_features):
        #  Determine if node_features are indices or precomputed embeddings:
        if node_features.dtype == torch.long:
            # Convert node indices to embeddings using the embedding layer
            h = self.embedding(node_features)
        else:
            # node_features are already embeddings
            h = node_features

        # First GAT layer: learn node representations using graph structure
        h1 = self.gat1(graph, h)  # [N, num_heads, hidden_dim_per_head]
        h1 = h1.view(-1, h1.size(1) * h1.size(2))  # [N, hidden_dim]
        h1 = F.elu(self.bn1(h1))
        h1 = F.dropout(h1, p=self.args.gnn_dropout_rate)

        # Second GAT layer: refine node representations
        h2 = self.gat2(graph, h1)  # [N, 1, hidden_dim]
        h2 = h2.squeeze(1)  # [N, hidden_dim]
        graph_features = F.elu(self.bn2(h2)) + h1  # Residual connection
        graph_features = F.dropout(graph_features, p=self.args.gnn_dropout_rate)

        # create a feature vector representing the PAIR (u, v)
        hidden_states = torch.cat([
            graph_features[vertex_pairs[:, 0]],
            graph_features[vertex_pairs[:, 1]],
            pairwise_features
        ], dim=1)

        # MLP с residual connections
        d1 = self.dense_1(hidden_states)
        hidden_states = torch.cat([hidden_states, d1], dim=1)

        d2 = self.dense_2(hidden_states)
        hidden_states = torch.cat([hidden_states, d2], dim=1)

        d3 = self.dense_3(hidden_states)
        hidden_states = torch.cat([hidden_states, d3], dim=1)

        d4 = self.dense_4(hidden_states)
        hidden_states = torch.cat([hidden_states, d4], dim=1)

        predictions = self.dense_5(hidden_states).squeeze(1)

        return predictions