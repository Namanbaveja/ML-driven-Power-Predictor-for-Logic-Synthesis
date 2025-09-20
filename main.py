# main.py

import argparse
import functools
import torch
import torch.nn as nn

from aig_qor.dataset import AIGDataset
from aig_qor.models import build_model
from aig_qor.train import do_train
from aig_qor.evaluate import do_test
from aig_qor.finetune import do_finetune
from aig_qor.mcts import do_mcts


def main():
    parser = argparse.ArgumentParser(description="AIG + Transformer QoR predictor")
    sub = parser.add_subparsers(dest="mode", required=True)

    # Default paths
    default_train_csv = "drive/MyDrive/mleda/Dataset/Final_Dataset/main_design_training.csv"
    default_test_csv = "drive/MyDrive/mleda/Dataset/Mid/Testing/tv80_orig.csv"
    default_train_model = "drive/MyDrive/mleda/models/best_91_final.pth"
    default_ft_model = "drive/MyDrive/mleda/models/finetune_best.pth"

    # All-in-one pipeline
    p_all = sub.add_parser("all")
    p_all.add_argument("--epochs", type=int, default=20)
    p_all.add_argument("--ft_epochs", type=int, default=20)
    p_all.add_argument("--batch_size", type=int, default=16)
    p_all.add_argument("--lr", type=float, default=1e-3)
    p_all.add_argument("--lr_fc", type=float, default=1e-3)
    p_all.add_argument("--lr_backbone", type=float, default=1e-4)
    p_all.add_argument("--train_csv", type=str, default=default_train_csv)
    p_all.add_argument("--test_csv", type=str, default=default_test_csv)
    p_all.add_argument("--model_train_path", type=str, default=default_train_model)
    p_all.add_argument("--model_ft_path", type=str, default=default_ft_model)
    p_all.add_argument("--ft_csv", type=str, default="drive/MyDrive/mleda/Dataset/Mid/Training/tv80_orig.csv")
    p_all.add_argument("--aigs", type=str, default="drive/MyDrive/mleda/aigs/")
    p_all.add_argument("--design", type=str, default="bp_be_orig.txt")
    p_all.add_argument("--recipe", type=str, default="st;")
    p_all.add_argument("--max_recipe_len", type=int, default=20)
    p_all.add_argument("--mcts_iters", type=int, default=100)
    p_all.add_argument("--freeze_backbone", action="store_true")

    # Train only
    p_train = sub.add_parser("train")
    p_train.add_argument("--epochs", type=int, default=1)
    p_train.add_argument("--batch_size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=2e-4)
    p_train.add_argument("--train_csv", type=str, default=default_train_csv)
    p_train.add_argument("--aigs", type=str, default="drive/MyDrive/mleda/aigs/")
    p_train.add_argument("--save_path", type=str, default=default_train_model)
    p_train.add_argument("--max_recipe_len", type=int, default=20)

    # Test only
    p_test = sub.add_parser("test")
    p_test.add_argument("--csv", type=str, default=default_test_csv)
    p_test.add_argument("--aigs", type=str, default="drive/MyDrive/mleda/aigs/")
    p_test.add_argument("--model", type=str, default=default_train_model)
    p_test.add_argument("--max_recipe_len", type=int, default=20)
    p_test.add_argument("--batch_size", type=int, default=4)

    # Fine-tune
    p_ft = sub.add_parser("finetune")
    p_ft.add_argument("--pretrained", type=str, default=default_train_model)
    p_ft.add_argument("--ft_csv", type=str, default="drive/MyDrive/mleda/Dataset/Small/Training/sqrt_orig.csv")
    p_ft.add_argument("--aigs", type=str, default="drive/MyDrive/mleda/aigs/")
    p_ft.add_argument("--ft_epochs", type=int, default=1)
    p_ft.add_argument("--lr_fc", type=float, default=1e-3)
    p_ft.add_argument("--lr_backbone", type=float, default=1e-4)
    p_ft.add_argument("--batch_size", type=int, default=8)
    p_ft.add_argument("--max_recipe_len", type=int, default=20)
    p_ft.add_argument("--freeze_backbone", action="store_true")
    p_ft.add_argument("--ft_save", type=str, default=default_ft_model)

    # MCTS search
    p_mcts = sub.add_parser("mcts")
    p_mcts.add_argument("--model", type=str, default=default_ft_model)
    p_mcts.add_argument("--aigs", type=str, default="drive/MyDrive/mleda/aigs/")
    p_mcts.add_argument("--csv", type=str, default=default_test_csv)
    p_mcts.add_argument("--ft_csv", type=str, default="drive/MyDrive/mleda/data/c7552_orig.csv")
    p_mcts.add_argument("--design", type=str, default="c7552_orig.txt")
    p_mcts.add_argument("--recipe", type=str, default="st;")
    p_mcts.add_argument("--max_recipe_len", type=int, default=20)
    p_mcts.add_argument("--mcts_iters", type=int, default=1000)

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(device)

    # Patch attention to always return weights
    for module in model.modules():
        if isinstance(module, nn.MultiheadAttention):
            orig_forward = module.forward

            @functools.wraps(orig_forward)
            def forward_with_weights(*args, **kwargs):
                kwargs["need_weights"] = True
                return orig_forward(*args, **kwargs)

            module.forward = forward_with_weights

    if args.mode == "all":
        args.save_path = args.model_train_path
        do_train(args, model, device)
        args.model = args.model_train_path
        args.csv = args.test_csv
        do_test(args, model, device)
        args.pretrained = args.model_train_path
        args.ft_save = args.model_ft_path
        do_finetune(args, model, device)
        args.model = args.model_ft_path
        args.csv = args.test_csv
        do_test(args, model, device)
        args.model = args.model_train_path
        do_mcts(args, model, device)
    elif args.mode == "train":
        do_train(args, model, device)
    elif args.mode == "test":
        do_test(args, model, device)
    elif args.mode == "finetune":
        do_finetune(args, model, device)
    elif args.mode == "mcts":
        do_mcts(args, model, device)


if __name__ == "__main__":
    main()
