import argparse
import os
import sys

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

from torch.utils.data import DataLoader
import wandb
import torch
import numpy as np
import pandas as pd

from src.dataset import load_dataset
from src.experiment import run_experiment, eval_run
from src.utils import correlation_matrices
from src.model import loss_fn


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Description of your program",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4,
        help="Seed for random number generation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Batch size for training",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=1,
        help="Number of epochs for training",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        default=2000,
        help="Dimension of hidden layers",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of layers in the model",
    )
    parser.add_argument(
        "--n_masks",
        type=int,
        default=1,
        help="Number of masks",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.001,
        help="Weight decay",
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=30,
        help="Step size",
    )
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1000,
        help="Length of sequence",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        required=True,
        help="Directory for logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for training",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=0,
        help="Number of GPUs to use",
    )
    parser.add_argument(
        "--use_wandb",
        type=bool,
        default=True,
        help="Flag to use wandb",
    )
    parser.add_argument(
        "--restore",
        action="store_true",
        help="Flag to restore from checkpoint",
    )

    args = parser.parse_args()
    return args


def create_loaders(
    train_data,
    val_data,
    test_data,
    batch_size,
):
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)
    return train_loader, val_loader, test_loader


def create_subset_loader(train_path, size):
    pre_data = pd.read_csv(train_path).head(size)
    df = pre_data.drop("Index", axis=1)
    subset_data = df.to_numpy()
    subset_data = np.where(subset_data > 0, 1, 0)
    subset_data = subset_data[:, np.newaxis, np.newaxis, :]
    subset_train_loader = DataLoader(subset_data, args.batch_size, shuffle=True)
    return subset_train_loader, subset_data


def load_checkpoint(
    model,
    metrics,
    log_dir,
    device,
):
    checkpoint = torch.load(
        os.path.join(log_dir, f'trainer_state_{metrics["lowest_val_epoch"]}.ckpt')
    )
    model_state_dict = {
        k: v for k, v in checkpoint["model"].items() if k in model.state_dict()
    }
    model.load_state_dict(model_state_dict)
    model = model.to(device)
    model.eval()
    return model


if __name__ == "__main__":
    args = parse_arguments()

    if args.device == "cuda":
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0

    # set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_path = r"../../data/processed/train.csv"
    test_path = r"../../data/processed/test.csv"

    train_data, val_data, test_data = load_dataset(
        train_path=train_path,
        test_path=test_path,
    )
    # @Todo: for now, keep loading this way, but eventually add seed to load_dataset function to split data in a seeded way directly

    # create data loaders
    train_loader, val_loader, test_loader = create_loaders(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        batch_size=args.batch_size,
    )

    # create subset of training set
    subset_size = 100
    subset_train_loader, subset_data = create_subset_loader(
        train_path=train_path, size=subset_size
    )

    # initiialize wandb
    run = wandb.init(
        # Set the project where this run will be logged
        project="gene_flows",
        # Track hyperparameters and run metadata
        config={
            # "learning_rate": learning_rate,
            "epochs": args.n_epochs,
            "hidden_dims": args.hidden_dims,
            "num_layers": args.num_layers,
            "n_masks": args.n_masks,
            "learning_rate": args.lr,
            "step_size": args.step_size,
            "log_dir": args.log_dir,
            "device": args.device,
            "seed": args.seed,
            "n_gpus": args.n_gpus,
            "restore": args.restore,
            "weight_decay": args.weight_decay,
        },
        dir=args.log_dir,
    )

    # run training
    model_trainer = run_experiment(
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=args.n_epochs,
        log_dir=args.log_dir,
        n_gpus=args.n_gpus,
        device_id=args.device,
        sequence_length=args.sequence_length,
        hidden_dims=args.hidden_dims,
        n_masks=args.n_masks,
        num_layers=args.num_layers,
        use_wandb=args.use_wandb,
        loss_fn=loss_fn,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        step_size=args.step_size,
        restore=args.restore,
        weight_decay=args.weight_decay,
    )

    # retrieve model and metrics
    model = model_trainer.model
    metrics = model_trainer.metrics

    print(
        f'Lowest validation error: {metrics["lowest_val_error"]:.3f} in epoch {metrics["lowest_val_epoch"]}'
    )

    # load the best checkpoint
    model = load_checkpoint(
        model=model,
        metrics=metrics,
        log_dir=args.log_dir,
        device=args.device,
    )

    # generate samples for visualization
    viz_data_pred = model.sample(500)[:, 0, 0, :20].detach().cpu().numpy()
    viz_data_gt = subset_data[:, 0, 0, :20]
    fig = correlation_matrices(viz_data_pred, viz_data_gt)

    print("Lowest val error checkpoint:")
    print(
        f"Subset Train: {round(eval_run(subset_train_loader, model, args.device).item(), 5)}"
    )
    print(f"Train: {round(eval_run(train_loader, model, args.device).item(), 5)}")
    print(f"Val: {round(eval_run(val_loader, model, args.device).item(), 5)}")
    print(f"Test: {round(eval_run(test_loader, model, args.device).item(), 5)}")

    # find lowest train epoch loss
    lowest_train_epoch_loss = np.inf
    lowest_train_epoch = 0
    for epoch, values in metrics.items():
        if type(epoch) == int:
            if values["train"]["loss"] < lowest_train_epoch_loss:
                lowest_train_epoch_loss = values["train"]["loss"]
                lowest_train_epoch = epoch

    print(lowest_train_epoch)

    print(
        f"Lowest train error: {lowest_train_epoch_loss:.3f} in epoch {lowest_train_epoch}"
    )

    checkpoint = torch.load(
        os.path.join(args.log_dir, f"trainer_state_{lowest_train_epoch}.ckpt")
    )
    model_state_dict = {
        k: v for k, v in checkpoint["model"].items() if k in model.state_dict()
    }
    model.load_state_dict(model_state_dict)
    model.eval()

    viz_data_pred = model.sample(500)[:, 0, 0, :20].detach().cpu()
    viz_data_gt = subset_data[:, 0, 0, :20]
    fig = correlation_matrices(viz_data_pred, viz_data_gt)

    print("Lowest train error checkpoint:")
    print(
        f"Subset Train: {round(eval_run(subset_train_loader, model, args.device).item(), 5)}"
    )
    print(f"Train: {round(eval_run(train_loader, model, args.device).item(), 5)}")
    print(f"Val: {round(eval_run(val_loader, model, args.device).item(), 5)}")
    print(f"Test: {round(eval_run(test_loader, model, args.device).item(), 5)}")
