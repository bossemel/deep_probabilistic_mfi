import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from torch.utils.data import DataLoader
import wandb
import numpy as np
import pandas as pd

import torch
from torch import optim
import pprint

from src.model import loss_fn
from src.pytorch_generative.pytorch_generative import models, trainer
from src.pytorch_generative.pytorch_generative import models, trainer
from src.dataset import load_dataset


def build_dataset(batch_size):
    train_path = r"../data/processed/train.csv"
    test_path = r"../data/processed/test.csv"

    train_data, val_data, test_data = load_dataset(
        train_path=train_path,
        test_path=test_path,
    )

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)

    pre_data = pd.read_csv(train_path).head(100)
    df = pre_data.drop("Index", axis=1)
    subset_data = df.to_numpy()
    subset_data = np.where(subset_data > 0, 1, 0)
    subset_data = subset_data[:, np.newaxis, np.newaxis, :]
    subset_train_loader = DataLoader(subset_data, batch_size, shuffle=True)
    return subset_train_loader, train_loader, val_loader, test_loader


def build_network(sequence_length, hidden_dims, num_layers, n_masks):
    # @Todo: adjust to allow multiple layers
    model = models.MADE(
        input_dim=sequence_length,
        hidden_dims=[hidden_dims] * num_layers,
        n_masks=n_masks,
    )
    return model


def build_optimizer(model, learning_rate, weight_decay, step_size, batch_size):
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
    return optimizer, scheduler


def train(config=None):
    sequence_length = 1000
    log_dir = "../tmp/test"
    n_gpus = 1
    debugging = False
    # Initialize a new wandb run
    with wandb.init(config=config):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        subset_train_loader, train_loader, val_loader, _ = build_dataset(
            config.batch_size
        )
        network = build_network(
            sequence_length,
            config.hidden_dims,
            config.num_layers,
            config.n_masks,
        )
        optimizer, scheduler = build_optimizer(
            network,
            config.learning_rate,
            config.weight_decay,
            config.lr_step_size,
            config.batch_size,
        )

        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
        print(log_dir)

        model_trainer = trainer.Trainer(
            model=network,
            loss_fn=loss_fn,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            train_loader=subset_train_loader if debugging else train_loader,
            eval_loader=val_loader,
            log_dir=log_dir,
            n_gpus=n_gpus,
            device_id=device,
            use_tensorboard=False,
            use_wandb=True,
            save_checkpoint_epochs=torch.inf,
        )
        model_trainer.interleaved_train_and_eval(config.n_epochs, restore=False)

        # for epoch in range(config.epochs):
        #     avg_loss = train_epoch(network, loader, optimizer)
        #     wandb.log({"loss": avg_loss, "epoch": epoch})


if __name__ == "__main__":
    # args = parse_arguments()

    seed = 4
    torch.manual_seed(seed)
    np.random.seed(seed)

    sweep_config = {"method": "bayes"}

    metric = {"name": "metrics/loss.eval", "goal": "minimize"}

    sweep_config["metric"] = metric

    parameters_dict = {
        "hidden_dims": {"values": [1028, 2048, 4096, 8192, 16384]},
        "num_layers": {"values": [2, 3, 4, 5, 10, 20]},
        "n_masks": {"values": [1, 2, 3, 4, 5, 10]},
        "lr_step_size": {"values": [5, 10, 20, 30, 50, 100]},
        "batch_size": {"values": [32, 64, 128, 256]},
    }

    parameters_dict.update(
        {
            "learning_rate": {
                # a flat distribution between 0 and 0.1
                "distribution": "uniform",
                "min": 0,
                "max": 0.1,
            },
            "weight_decay": {
                # a flat distribution between 0 and 0.1
                "distribution": "uniform",
                "min": 0,
                "max": 0.01,
            },
        }
    )

    sweep_config["parameters"] = parameters_dict
    debugging = False
    parameters_dict.update({"n_epochs": {"value": 1 if debugging is True else 100}})

    pprint.pprint(sweep_config)

    #sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    sweep_id = "nl0wrff5"
    wandb.agent(sweep_id, train, project="pytorch-sweeps-demo", count=5)
