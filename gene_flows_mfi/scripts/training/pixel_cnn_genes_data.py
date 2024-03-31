import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from torch.utils.data import DataLoader
import wandb
import torch
import numpy as np
import pandas as pd
import argparse

from src.dataset import load_dataset

# from src.experiment import run_experiment, eval_run
from src.utils import correlation_matrices
from src.model import loss_fn
from torch import optim
from torch.nn import functional as F
import os

from src.model import loss_fn
from src.pytorch_generative.pytorch_generative import models, trainer


def run_experiment(
    train_loader,
    val_loader,
    n_epochs,
    log_dir,
    n_gpus,
    device_id,
    sequence_length,
    hidden_dims,
    num_layers,
    n_masks,
    loss_fn,
    batch_size,
    learning_rate,
    step_size,
    weight_decay=0,
    use_tensorboard=False,
    use_wandb=True,
    restore=True,
):
    """Training script with defaults to reproduce results.

    The code inside this function is self contained and can be used as a top level
    training script, e.g. by copy/pasting it into a Jupyter notebook.

    Args:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size to use for training and evaluation.
        log_dir: Directory where to log trainer state and TensorBoard summaries.
        n_gpus: Number of GPUs to use for training the model. If 0, uses CPU.
        device_id: The device_id of the current GPU when training on multiple GPUs.
        debug_loader: Debug DataLoader which replaces the default training and
            evaluation loaders if not 'None'. Do not use unless you're writing unit
            tests.
    """

    # @Todo: adjust to allow multiple layers
    model = models.PixelCNN(residual_channels=128, head_channels=32)
    # model = models.PixelSNAIL(in_channels=1, out_channels=1)
    # .MADE(
    #     input_dim=sequence_length,
    #     hidden_dims=[hidden_dims] * num_layers,
    #     n_masks=n_masks,
    # )
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # @Todo: debug scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size * batch_size,
    )

    # train_loss = round(eval_run(train_loader, model, "mps").item(), 5)
    # print(f"Train loss before training: {train_loss}")
    # val_loss = round(eval_run(val_loader, model, "mps").item(), 5)
    # print(f"Val loss before training: {val_loss}")

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
    print(log_dir)

    model_trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        eval_loader=val_loader,
        log_dir=log_dir,
        n_gpus=n_gpus,
        device_id=device_id,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
    )
    model_trainer.interleaved_train_and_eval(n_epochs, restore=restore)

    return model_trainer


def eval_run(data_loader, model, device):
    # Evaluate.
    n_examples = 0
    losses = 0
    model.eval()
    for batch in data_loader:
        batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
        x, y = batch
        n_batch_examples = x.shape[0]
        n_examples += n_batch_examples
        x = x.float().to(device)
        preds = model(x.float())
        loss = loss_fn(x.float(), y, preds).detach()
        losses += loss
    return losses / n_examples


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
        default=256,
        help="Batch size for training",
    )
    parser.add_argument(
        "--n_epochs",
        type=int,
        default=100,
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
        default="../tmp/run/genes_3",
        help="Directory for logging",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device for training",
    )
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
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


if __name__ == "__main__":
    args = parse_arguments()

    if args.device == "cuda":
        assert torch.cuda.is_available()
        assert torch.cuda.device_count() > 0

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_path = r"../data/processed/train.csv"
    test_path = r"../data/processed/test.csv"

    train_data, val_data, test_data = load_dataset(
        train_path=train_path,
        test_path=test_path,
    )

    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, args.batch_size, shuffle=True)

    pre_data = pd.read_csv(train_path).head(100)
    df = pre_data.drop("Index", axis=1)
    subset_data = df.to_numpy()
    subset_data = np.where(subset_data > 0, 1, 0)
    subset_data = subset_data[:, np.newaxis, np.newaxis, :]
    subset_train_loader = DataLoader(subset_data, args.batch_size, shuffle=True)

    if args.use_wandb:
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

    model = model_trainer.model
    metrics = model_trainer.metrics

    print(
        f'Lowest validation error: {metrics["lowest_val_error"]:.3f} in epoch {metrics["lowest_val_epoch"]}'
    )

    checkpoint = torch.load(
        os.path.join(args.log_dir, f'trainer_state_{metrics["lowest_val_epoch"]}.ckpt')
    )
    model_state_dict = {
        k: v for k, v in checkpoint["model"].items() if k in model.state_dict()
    }
    model.load_state_dict(model_state_dict)
    model = model.to(args.device)
    model.eval()

    viz_data_pred = model.sample(500)[:, 0, 0, :20].detach().cpu().numpy()
    # print(data_samples.shape)
    viz_data_gt = subset_data[:, 0, 0, :20]
    fig = correlation_matrices(viz_data_pred, viz_data_gt)

    print(round(eval_run(subset_train_loader, model, args.device).item(), 5))
    print(round(eval_run(train_loader, model, args.device).item(), 5))
    print(round(eval_run(val_loader, model, args.device).item(), 5))
    print(round(eval_run(test_loader, model, args.device).item(), 5))

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
    # print(data_samples.shape)
    viz_data_gt = subset_data[:, 0, 0, :20]
    fig = correlation_matrices(viz_data_pred, viz_data_gt)

    print(round(eval_run(subset_train_loader, model, args.device).item(), 5))
    print(round(eval_run(train_loader, model, args.device).item(), 5))
    print(round(eval_run(val_loader, model, args.device).item(), 5))
    print(round(eval_run(test_loader, model, args.device).item(), 5))
