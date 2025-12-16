import argparse
import os
import pickle
import random
import json
import csv
from dataclasses import dataclass
from typing import Any, Tuple, Dict, List
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional as F
import dgl
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from models.gat import GATModel



def set_seed(seed: int) -> None:
    """Set random seed for reproducibility across all libraries"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def parse_dataset_structure(obj: Any) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract data from FutureOfAIviaAI .pkl format
    Format:
    - obj[0]: (E,3) edges with timestamp (we use only first 2 columns)
    - obj[1]: (M,2) candidate pairs for evaluation
    - obj[2]: (M,) labels (1=positive edge, 0=negative edge)
    """
    if not isinstance(obj, (list, tuple)) or len(obj) < 3:
        raise ValueError("Expected list/tuple dataset with at least 3 elements.")

    edges_train = obj[0]  # (E, 3)
    pairs = obj[1]  # (M, 2)
    labels = obj[2]  # (M,)

    return edges_train[:, :2], pairs, labels.astype(np.int64)


def build_dgl_graph(edges: np.ndarray, num_nodes: int) -> dgl.DGLGraph:
    """Create DGL graph from edges array"""
    src = edges[:, 0]
    dst = edges[:, 1]

    graph = dgl.graph((src, dst), num_nodes=num_nodes)
    graph = dgl.add_self_loop(graph)  # добавляем self-loops для GAT

    return graph


def create_node_features(num_nodes: int, embedding_dim: int = 16) -> Tuple[torch.Tensor, torch.nn.Embedding]:
    """
    Create simple node features:
    1. Node indices for embedding lookup
    2. Embedding layer that will be learned
    """
    node_indices = torch.arange(num_nodes, dtype=torch.long)
    embedding = torch.nn.Embedding(num_nodes, embedding_dim)

    return node_indices, embedding


@dataclass
class GATConfig:
    embedding_dim: int = 256
    hidden_dim: int = 256
    dropout: float = 0.3
    lr: float = 0.001
    weight_decay: float = 1e-4
    epochs: int = 5
    batch_size: int = 1024
    num_heads: int = 4
    dnn_hidden_dim: int = 256


class Logger:
    """Logging utility to save training progress and metrics"""
    def __init__(self, log_dir: str, dataset_name: str):
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{dataset_name}_{timestamp}.log")
        self.metrics_file = os.path.join(log_dir, f"{dataset_name}_{timestamp}_metrics.json")
        self.metrics = {
            "dataset": dataset_name,
            "timestamp": timestamp,
            "epochs": [],
            "config": {},
            "final_auc": 0.0
        }

    def log(self, message: str):
        """Log message with timestamp to file and console"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        print(log_message)
        with open(self.log_file, 'a') as f:
            f.write(log_message + '\n')

    def log_metrics(self, epoch: int, loss: float, auc: float):
        epoch_metrics = {
            "epoch": epoch,
            "loss": loss,
            "auc": auc
        }
        self.metrics["epochs"].append(epoch_metrics)

    def save_metrics(self, config: Dict, final_auc: float):
        self.metrics["config"] = config
        self.metrics["final_auc"] = final_auc
        with open(self.metrics_file, 'w') as f:
            json.dump(self.metrics, f, indent=2)


def sample_negative_edges_fast(num_nodes: int, num_samples: int,
                               existing_edges: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Fast sampling of negative edges (non-existing edges)"""
    if existing_edges.size(1) > 100000:
        idx = torch.randint(0, existing_edges.size(1), (100000,), device=device)
        existing_subset = existing_edges[:, idx]
    else:
        existing_subset = existing_edges

    existing_set = set(zip(existing_subset[0].cpu().numpy(),
                           existing_subset[1].cpu().numpy()))

    neg_edges_list = []
    attempts = 0
    max_attempts = num_samples * 10

    while len(neg_edges_list) < num_samples and attempts < max_attempts:
        # Sample a batch
        batch_size = min(num_samples * 2, 10000)
        u = torch.randint(0, num_nodes, (batch_size,), device=device)
        v = torch.randint(0, num_nodes, (batch_size,), device=device)

        for i in range(batch_size):
            if len(neg_edges_list) >= num_samples:
                break

            ui, vi = u[i].item(), v[i].item()
            if ui != vi and (ui, vi) not in existing_set:
                neg_edges_list.append((ui, vi))

        attempts += 1

    # If we couldn't sample enough, fill with random edges
    while len(neg_edges_list) < num_samples:
        u = torch.randint(0, num_nodes, (1,), device=device).item()
        v = torch.randint(0, num_nodes, (1,), device=device).item()
        if u != v:
            neg_edges_list.append((u, v))

    neg_edges = torch.tensor(neg_edges_list, device=device).t()
    return neg_edges


def train_gat(
        train_graph: dgl.DGLGraph,
        test_pairs: torch.Tensor,
        test_labels: torch.Tensor,
        node_indices: torch.Tensor,
        embedding_layer: torch.nn.Embedding,
        num_nodes: int,
        cfg: GATConfig,
        device: torch.device,
        logger: Logger,
        dataset_name: str
) -> float:
    # Create GAT model configuration
    class Args:
        num_node_features = 0  #  don't use original node features
        num_pairwise_features = 0  #  don't use pairwise features
        embedding_dim = cfg.embedding_dim
        hidden_dim = cfg.hidden_dim
        num_heads = cfg.num_heads
        dnn_hidden_dim = cfg.dnn_hidden_dim
        gnn_dropout_rate = cfg.dropout
        dnn_dropout_rate = cfg.dropout

    args = Args()
    model = GATModel(args, num_nodes=num_nodes).to(device)
    model.embedding = embedding_layer.to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay
    )

    train_graph = train_graph.to(device)
    node_indices = node_indices.to(device)
    test_pairs = test_pairs.to(device)
    test_labels = test_labels.float().to(device)

    # Prepare training data
    src, dst = train_graph.edges()
    train_edges = torch.stack([src, dst], dim=0).to(device)  # [2, E]

    E = train_edges.size(1)

    logger.log(f"Dataset: {dataset_name}")
    logger.log(f"Training GAT: nodes={num_nodes}, train_edges={E}, test_pairs={len(test_pairs)}")

    # Use subset for evaluation (for speed)
    eval_sample_size = min(50000, len(test_pairs))
    if len(test_pairs) > eval_sample_size:
        eval_indices = torch.randperm(len(test_pairs))[:eval_sample_size]
        test_pairs_eval = test_pairs[eval_indices]
        test_labels_eval = test_labels[eval_indices]
    else:
        test_pairs_eval = test_pairs
        test_labels_eval = test_labels

    logger.log(f"Using {len(test_pairs_eval)} samples for evaluation")

    best_auc = 0.0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        logger.log(f"\n{'=' * 60}")
        logger.log(f"Starting epoch {epoch}/{cfg.epochs}")
        logger.log(f"{'=' * 60}")

        model.train()
        epoch_loss = 0
        num_batches = 0

        iterations_per_epoch = max(1, min(5, E // cfg.batch_size))

        for batch_idx in tqdm(range(iterations_per_epoch),
                              desc=f"Epoch {epoch} training",
                              leave=False):
            # Take batch of positive edges
            if E > cfg.batch_size:
                idx = torch.randint(0, E, (cfg.batch_size,), device=device)
                pos_edges = train_edges[:, idx]
            else:
                pos_edges = train_edges

            # Sample negative edges
            neg_edges = sample_negative_edges_fast(
                num_nodes=num_nodes,
                num_samples=pos_edges.size(1),
                existing_edges=train_edges,
                device=device
            )

            # Combine positive and negative
            all_pairs = torch.cat([pos_edges.t(), neg_edges.t()], dim=0)
            all_labels = torch.cat([
                torch.ones(pos_edges.size(1), device=device),
                torch.zeros(neg_edges.size(1), device=device)
            ])

            # Shuffle
            perm = torch.randperm(len(all_pairs))
            batch_pairs = all_pairs[perm]
            batch_labels = all_labels[perm]

            # Split into mini-batches
            mini_batch_size = cfg.batch_size // 2
            for i in range(0, len(batch_pairs), mini_batch_size):
                end_idx = min(i + mini_batch_size, len(batch_pairs))
                if end_idx <= i:
                    continue

                mini_pairs = batch_pairs[i:end_idx]
                mini_labels = batch_labels[i:end_idx]

                # Predictions
                predictions = model(
                    graph=train_graph,
                    node_features=node_indices,
                    vertex_pairs=mini_pairs,
                    pairwise_features=torch.zeros((len(mini_pairs), 0)).to(device)
                )

                # Loss
                loss = F.binary_cross_entropy_with_logits(predictions, mini_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

        avg_loss = epoch_loss / max(1, num_batches)

        # Evaluation
        logger.log(f"Training completed. Starting evaluation...")
        model.eval()
        with torch.no_grad():
            all_predictions = []
            eval_batch_size = 2048

            for i in tqdm(range(0, len(test_pairs_eval), eval_batch_size),
                          desc=f"Epoch {epoch} evaluation",
                          leave=False):
                batch_pairs = test_pairs_eval[i:i + eval_batch_size]
                batch_predictions = model(
                    graph=train_graph,
                    node_features=node_indices,
                    vertex_pairs=batch_pairs,
                    pairwise_features=torch.zeros((len(batch_pairs), 0)).to(device)
                )
                all_predictions.append(torch.sigmoid(batch_predictions))

            predictions = torch.cat(all_predictions).cpu().numpy()
            auc = roc_auc_score(test_labels_eval.cpu().numpy(), predictions)

        logger.log(f"Epoch {epoch}: Avg Loss = {avg_loss:.4f}, AUC = {auc:.4f}")
        logger.log_metrics(epoch, avg_loss, auc)

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            # Save best model for this dataset
            model_path = f"models/best_{dataset_name}.pth"
            os.makedirs("models", exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'embedding_state_dict': embedding_layer.state_dict(),
                'auc': auc
            }, model_path)
            logger.log(f"  ✅ Saved best model to {model_path} (AUC: {auc:.4f})")

    logger.log(f"Training completed for {dataset_name}")
    logger.log(f"Best AUC: {best_auc:.4f} at epoch {best_epoch}")

    return best_auc


def save_results_to_csv(results: List[Dict], output_file: str = "results.csv"):
    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", output_file)

    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to {output_path}")


def save_results_table(results: List[Dict], output_file: str = "results_table.csv"):
    table_data = {}

    for result in results:
        key = (result['delta'], result['minedge'])
        if key not in table_data:
            table_data[key] = {}
        table_data[key][result['cutoff']] = result['auc']

    os.makedirs("results", exist_ok=True)
    output_path = os.path.join("results", output_file)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["Area under the Curve (AUC) for prediction of new edge_weights of 1"])
        writer.writerow(["", "cutoff=0", "cutoff=5", "cutoff=25"])

        for delta in [1, 3, 5]:
            for minedge in [1, 3]:
                row = [f"delta={delta}, minedge={minedge}"]
                for cutoff in [0, 5, 25]:
                    key = (delta, minedge)
                    if key in table_data and cutoff in table_data[key]:
                        row.append(f"{table_data[key][cutoff]:.4f}")
                    else:
                        row.append("N/A")
                writer.writerow(row)

        writer.writerow([])

        writer.writerow(["Area under the Curve (AUC) for prediction of new edge_weights of 3"])
        writer.writerow(["", "cutoff=0", "cutoff=5", "cutoff=25"])

        for delta in [1, 3, 5]:
            for minedge in [1, 3]:
                row = [f"delta={delta}, minedge={minedge}"]
                for cutoff in [0, 5, 25]:
                    key = (delta, minedge)
                    if key in table_data and cutoff in table_data[key]:
                        row.append(f"{table_data[key][cutoff]:.4f}")
                    else:
                        row.append("N/A")
                writer.writerow(row)

    print(f"\nResults table saved to {output_path}")

    json_path = os.path.join("results", "results_summary.json")
    with open(json_path, 'w') as f:
        json.dump(table_data, f, indent=2)

    print(f"Results summary saved to {json_path}")


def process_dataset(data_path: str, cfg: GATConfig, device: torch.device,
                    log_dir: str = "logs") -> Dict:
    """Process one dataset"""
    # Extract parameters from filename
    filename = os.path.basename(data_path)

    # Parsing for format: SemanticGraph_delta_1_cutoff_0_minedge_1.pkl
    parts = filename.replace('SemanticGraph_', '').replace('.pkl', '').split('_')

    # parts = ['delta', '1', 'cutoff', '0', 'minedge', '1']
    if len(parts) >= 6:
        delta = int(parts[1])    # parts[1] = '1'
        cutoff = int(parts[3])   # parts[3] = '0'
        minedge = int(parts[5])  # parts[5] = '1'
    else:
        # Old format for backward compatibility
        delta = int(parts[0].replace('delta=', ''))
        cutoff = int(parts[1].replace('cutoff=', ''))
        minedge = int(parts[2].replace('minedge=', ''))

    dataset_name = f"delta_{delta}_cutoff_{cutoff}_minedge_{minedge}"

    print(f"Processing dataset: {dataset_name}")
    print(f"File: {filename}")

    # Create logger
    logger = Logger(log_dir, dataset_name)
    logger.log(f"Starting processing for {dataset_name}")
    logger.log(f"Config: {cfg}")

    try:
        # Load data
        with open(data_path, "rb") as f:
            obj = pickle.load(f)

        # Extract edges and test pairs
        train_edges_raw, test_pairs_raw, test_labels_raw = parse_dataset_structure(obj)

        # Determine number of nodes
        num_nodes = int(max(train_edges_raw.max(), test_pairs_raw.max())) + 1

        logger.log(f"Dataset stats:")
        logger.log(f"  Num nodes: {num_nodes}")
        logger.log(f"  Train edges: {len(train_edges_raw)}")
        logger.log(f"  Test pairs: {len(test_pairs_raw)}")

        # Check label distribution
        pos_count = (test_labels_raw == 1).sum()
        neg_count = (test_labels_raw == 0).sum()
        logger.log(f"  Positive test pairs: {pos_count} ({pos_count / len(test_labels_raw) * 100:.1f}%)")
        logger.log(f"  Negative test pairs: {neg_count} ({neg_count / len(test_labels_raw) * 100:.1f}%)")

        # Create DGL graph
        train_graph = build_dgl_graph(train_edges_raw, num_nodes)

        # Create node features
        node_indices, embedding_layer = create_node_features(num_nodes, cfg.embedding_dim)

        # Convert test data to tensors
        test_pairs = torch.tensor(test_pairs_raw, dtype=torch.long)
        test_labels = torch.tensor(test_labels_raw, dtype=torch.float32)

        # Training
        auc = train_gat(
            train_graph=train_graph,
            test_pairs=test_pairs,
            test_labels=test_labels,
            node_indices=node_indices,
            embedding_layer=embedding_layer,
            num_nodes=num_nodes,
            cfg=cfg,
            device=device,
            logger=logger,
            dataset_name=dataset_name
        )

        # Save metrics
        config_dict = {
            'embedding_dim': cfg.embedding_dim,
            'hidden_dim': cfg.hidden_dim,
            'dropout': cfg.dropout,
            'lr': cfg.lr,
            'weight_decay': cfg.weight_decay,
            'epochs': cfg.epochs,
            'batch_size': cfg.batch_size,
            'num_heads': cfg.num_heads
        }
        logger.save_metrics(config_dict, auc)

        result = {
            'dataset': dataset_name,
            'filename': filename,
            'delta': delta,
            'cutoff': cutoff,
            'minedge': minedge,
            'auc': auc,
            'num_nodes': num_nodes,
            'train_edges': len(train_edges_raw),
            'test_pairs': len(test_pairs_raw),
            'pos_ratio': pos_count / len(test_labels_raw)
        }

        logger.log(f"Completed {dataset_name} with AUC: {auc:.4f}")

        return result

    except Exception as e:
        logger.log(f"ERROR processing {dataset_name}: {str(e)}")
        print(f"Error processing {data_path}: {e}")
        return {
            'dataset': dataset_name,
            'filename': filename,
            'delta': delta,
            'cutoff': cutoff,
            'minedge': minedge,
            'auc': 0.0,
            'error': str(e)
        }


def main():
    parser = argparse.ArgumentParser(description="Train GAT on multiple datasets")
    parser.add_argument("--data_dir", type=str, default="data",
                        help="Directory containing .pkl dataset files")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of epochs per dataset")
    parser.add_argument("--batch_size", type=int, default=1024,
                        help="Batch size for training")
    parser.add_argument("--embedding_dim", type=int, default=256,
                        help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3,
                        help="Dropout rate")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--dnn_hidden_dim", type=int, default=256,
                        help="DNN hidden dimension")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--device", type=str, default="cpu",
                        choices=["cpu", "mps", "cuda"],
                        help="Device to use")
    parser.add_argument("--datasets", type=str, default="all",
                        help="Comma-separated list of datasets to process, or 'all'")
    args = parser.parse_args()

    # Set seed
    set_seed(args.seed)
    device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Random seed: {args.seed}")

    # Config
    cfg = GATConfig(
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_heads=args.num_heads,
        dnn_hidden_dim=args.dnn_hidden_dim
    )

    print(f"\nGAT Config:")
    for key, value in cfg.__dict__.items():
        print(f"  {key}: {value}")

    # Find all datasets
    all_files = []
    for file in os.listdir(args.data_dir):
        if file.endswith(".pkl") and file.startswith("SemanticGraph"):
            all_files.append(os.path.join(args.data_dir, file))

    all_files.sort()
    print(f"\nFound {len(all_files)} dataset files:")
    for f in all_files:
        print(f"  {os.path.basename(f)}")

    # Select which datasets to process
    if args.datasets.lower() == "all":
        datasets_to_process = all_files
    else:
        selected_names = args.datasets.split(',')
        datasets_to_process = []
        for f in all_files:
            basename = os.path.basename(f).replace('.pkl', '')
            if any(name in basename for name in selected_names):
                datasets_to_process.append(f)

    print(f"\nProcessing {len(datasets_to_process)} datasets:")
    for f in datasets_to_process:
        print(f"  {os.path.basename(f)}")

    # Create directories for logs and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f"logs/gat_experiment_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)

    print(f"\nLogs will be saved to: {log_dir}")

    results = []

    for i, data_path in enumerate(datasets_to_process, 1):
        print(f"\n\nProcessing dataset {i}/{len(datasets_to_process)}")

        result = process_dataset(data_path, cfg, device, log_dir)
        results.append(result)

        # Save intermediate results
        save_results_to_csv(results, f"results_interim_{timestamp}.csv")

    # Save final results
    save_results_to_csv(results, f"results_final_{timestamp}.csv")
    save_results_table(results, f"results_table_{timestamp}.csv")

    print("EXPERIMENT COMPLETE")

    print("\nSummary of results:")
    print(f"{'Dataset':<40} {'AUC':<10} {'Nodes':<10} {'Edges':<12} {'Test Pairs':<12}")

    for result in results:
        if 'error' in result:
            print(f"{result['dataset']:<40} ERROR")
        else:
            print(f"{result['dataset']:<40} {result['auc']:.4f}   {result['num_nodes']:<10} "
                  f"{result['train_edges']:<12,} {result['test_pairs']:<12,}")

    print(f"\nAll results saved to: results/results_final_{timestamp}.csv")
    print(f"Results table saved to: results/results_table_{timestamp}.csv")
    print(f"Logs saved to: {log_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()