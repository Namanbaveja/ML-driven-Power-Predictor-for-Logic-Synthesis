# aig_qor/dataset.py

import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from torch_geometric.data import Data
from .parse import parse_bench


def preprocess_and_save_graphs(aigs_dir, output_dir, aig_list, parse_bench_fn=parse_bench):
    os.makedirs(output_dir, exist_ok=True)
    for aig_name in tqdm(aig_list):
        bench_path = os.path.join(aigs_dir, aig_name)
        try:
            graph = parse_bench_fn(bench_path)
            torch.save(graph, os.path.join(output_dir, aig_name + '.pt'))
        except Exception as e:
            print(f"Failed to process {aig_name}: {e}")


def ensure_graph_cache(aigs_dir, graph_cache_dir, data_df, parse_bench_fn=parse_bench):
    to_process = []
    for aig in data_df['aig'].unique():
        cached_path = os.path.join(graph_cache_dir, aig + '.pt')
        if not os.path.exists(cached_path):
            to_process.append(aig)
    if to_process:
        print(f"Preprocessing {len(to_process)} graphs...")
        preprocess_and_save_graphs(aigs_dir, graph_cache_dir, to_process, parse_bench_fn)


class AIGDataset(Dataset):
    def __init__(self, aigs_dir, csv_file, max_recipe_len=5, vocab=None, graph_cache_dir="graph_cache"):
        self.aigs_dir = aigs_dir
        self.graph_cache_dir = graph_cache_dir
        self.data_df = pd.read_csv(csv_file)
        self.max_recipe_len = max_recipe_len
        self.vocab = vocab or {
            'b': 0, 'rw': 1, 'rf': 2, 'rs': 3, 'st': 4,
            'rwz': 5, 'rfz': 6, 'rsz': 7, 'dc2': 8, 'f': 9
        }

        all_qors = []
        for i in range(1, self.max_recipe_len + 1):
            all_qors += self.data_df[f"qor{i}"].astype(float).apply(np.log1p).tolist()
        self.global_mean = np.mean(all_qors)
        self.global_std = np.std(all_qors) + 1e-8

        self.qor_stats = {}
        for name in self.data_df["aig"].unique():
            self.qor_stats[name] = {"mean": self.global_mean, "std": self.global_std}
            if not name.endswith(".txt"):
                self.qor_stats[name + ".txt"] = {"mean": self.global_mean, "std": self.global_std}

        ensure_graph_cache(self.aigs_dir, self.graph_cache_dir, self.data_df, parse_bench)

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        row = self.data_df.iloc[idx]
        aig_name = row['aig']
        recipe_str = row['recipe']

        tokens = [t.strip() for t in recipe_str.split(';') if t.strip()]
        tokens = tokens[:self.max_recipe_len]
        while len(tokens) < self.max_recipe_len:
            tokens.append('')
        unk_idx = len(self.vocab)
        token_indices = [self.vocab.get(tok, unk_idx) for tok in tokens]
        recipe_tokens = torch.tensor(token_indices, dtype=torch.long)

        qor_vals = [float(row[f"qor{i+1}"]) for i in range(self.max_recipe_len)]
        log_qors = np.log1p(qor_vals)
        norm_qors = (log_qors - self.global_mean) / self.global_std
        norm_qor_seq = torch.tensor(norm_qors, dtype=torch.float)

        graph_path = os.path.join(self.graph_cache_dir, aig_name + '.pt')
        graph = torch.load(graph_path, weights_only=False)

        return graph, recipe_tokens, norm_qor_seq, aig_name


def denormalize_log_qors(preds, global_mean, global_std):
    if isinstance(preds, torch.Tensor):
        log_qors = preds * global_std + global_mean
        return torch.expm1(log_qors)
    else:
        log_qors = preds * global_std + global_mean
        return np.expm1(log_qors)
