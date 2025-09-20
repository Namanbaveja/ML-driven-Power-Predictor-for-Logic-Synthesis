# aig_qor/evaluate.py

import os
import pandas as pd
import torch
from torch_geometric.data import Batch

from .dataset import denormalize_log_qors, AIGDataset
from .utils import predict_design_qor


def test_model(model, test_dataset, batch_size=4, device=torch.device("cuda")):
    """
    Evaluate model on a test dataset split and compute mean relative error.
    """
    def collate_fn(batch):
        graphs, recipe_tokens, target_seq, names = zip(*batch)
        return (
            Batch.from_data_list(graphs),
            torch.stack(recipe_tokens),
            torch.stack(target_seq),
            names,
        )

    loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    model.eval()
    total_error = 0.0
    total_samples = 0

    # Access mean/std from underlying dataset
    mean = test_dataset.dataset.global_mean
    std = test_dataset.dataset.global_std

    with torch.no_grad():
        for G, rec, tgt, names in loader:
            preds = model(G, rec)[:, -1, 0].cpu()
            targets = tgt[:, -1].cpu()
            denorm_preds = denormalize_log_qors(preds, mean, std)
            denorm_tgts = denormalize_log_qors(targets, mean, std)
            total_error += (
                (denorm_preds - denorm_tgts).abs().div(denorm_tgts + 1e-8).sum().item()
            )
            total_samples += preds.size(0)

    mean_rel_error = total_error / total_samples if total_samples > 0 else None
    print(f"Mean Relative Error: {mean_rel_error * 100:.2f}%")
    return mean_rel_error


def evaluate_model_on_csv_and_print_accuracy(
    model,
    csv_file,
    aigs_dir,
    vocab,
    max_recipe_len,
    qor_stats=None,
    dataset=None,
    device=torch.device("cpu"),
):
    """
    Evaluate model on a CSV of designs/recipes and compute ±3% accuracy.
    Saves results to 'evaluation_results.csv'.
    """
    if qor_stats is None:
        if dataset is None:
            raise ValueError("Either `qor_stats` or `dataset` must be provided")
        base = getattr(dataset, "dataset", dataset)
        global_mean = base.global_mean
        global_std = base.global_std
    else:
        global_mean = None
        global_std = None

    df = pd.read_csv(csv_file)
    results = []
    in_tolerance_scores = []

    for idx, row in df.iterrows():
        design = row["aig"]
        recipe = row["recipe"]
        actual_qor = float(row["power"])

        if qor_stats and design in qor_stats:
            mean = qor_stats[design]["mean"]
            std = qor_stats[design]["std"]
        else:
            if global_mean is None or global_std is None:
                raise ValueError("Global stats not available for fallback")
            mean = global_mean
            std = global_std

        pred_list = predict_design_qor(
            model,
            design,
            recipe,
            std,
            mean,
            vocab,
            aigs_dir=aigs_dir,
            max_recipe_len=max_recipe_len,
            device=device,
        )
        pred_qor = pred_list[-1]
        lower, upper = 0.97 * actual_qor, 1.03 * actual_qor
        score = 1 if (lower <= pred_qor <= upper) else 0
        in_tolerance_scores.append(score)

        results.append(
            {
                "aig": design,
                "recipe": recipe,
                "actual_qor": actual_qor,
                "predicted_qor": pred_qor,
                "within_3pct": score,
            }
        )

        print(
            f"Design: {design}\n"
            f"Recipe: {recipe}\n"
            f"Actual QoR: {actual_qor:.2f}  Predicted QoR: {pred_qor:.2f}  "
            f"{'✅' if score else '❌'} within ±3%\n-----"
        )

    total = len(in_tolerance_scores)
    correct = sum(in_tolerance_scores)
    accuracy_pct = (correct / total) * 100 if total else None

    results_df = pd.DataFrame(results)
    results_df.to_csv("evaluation_results.csv", index=False)

    print(f"\nNumber within ±3%: {correct}/{total}")
    print(f"Accuracy (within ±3%): {accuracy_pct:.2f}%")

    return accuracy_pct, results_df
