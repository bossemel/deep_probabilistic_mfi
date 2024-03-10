import os
import sys

# module_path = os.path.abspath(os.path.join("../.."))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import more_itertools
import math
import itertools
import time
import torch
from torch.utils.data import DataLoader

from src.pytorch_generative.pytorch_generative import models
from src.model import nll_singles

torch.set_default_dtype(torch.float16)


class MfiDatasetIterator(DataLoader):
    """DataLoader function to create all joint probabiliy values for the MFI calculation. Contains the self.dataset as
    a generator, and creates each batch on the fly in __next__. Num factors is 2 for 2-factor MFI, etc. Value is the
    current interaction value of interest. Num features is the number of columns in the dataset, in our case 1000.
    """

    def __init__(self, num_features, batch_size, value, device, num_factors):
        self.value = value
        self.num_features = num_features
        self.num_factors = num_factors
        self.dataset_length = math.comb(
            self.num_features,
            self.num_factors,
        )
        self.dataset = itertools.combinations(
            range(self.num_features),
            self.num_factors,
        )

        self.batch_size = batch_size
        self.device = device
        self.num_batches = math.ceil(self.dataset_length / batch_size)

        self.current_batch = 0

        self.get_curr_batch_size()

        self.batch = torch.zeros(
            self.len_curr_batch,
            self.num_features,
            device=self.device,
            dtype=torch.bool,
        )

    def __iter__(self):
        return self

    def get_curr_batch_size(self):
        if (self.current_batch + 1) * self.batch_size < self.dataset_length:
            self.len_curr_batch = self.batch_size
        else:
            self.len_curr_batch = (
                self.dataset_length - self.current_batch * self.batch_size
            )

    def __next__(self):
        """Generates the next batch using the itertools combination generator."""
        combs = torch.IntTensor(
            list(itertools.islice(self.dataset, self.len_curr_batch))
        ).to(self.device)

        self.batch = torch.zeros(
            self.len_curr_batch,
            self.num_features,
            device=self.device,
            dtype=torch.bool,
        )
        self.batch[
            torch.arange(self.len_curr_batch, device=self.device).unsqueeze(1), combs
        ] = self.value

        self.current_batch += 1

        return self.batch.half()

    def __len__(self):
        return self.num_batches


def gen_mfi_dict(num_factors, preds, pred_zero):
    if num_factors == 2:
        return two_factor_mfi(preds, pred_zero)
    elif num_factors == 3:
        return three_factor_mfi(preds, pred_zero)
    elif num_factors == 4:
        return four_factor_mfi(preds, pred_zero)
    else:
        raise ValueError("Unknown number factors")


def pred(
    model,
    sequence_length,
    loader,
    value_to_save,
    device,
    num_factors,
    save_every,
):
    """Generates the joint probability over the dataset, allowing for intermediate saves."""
    zero_input = torch.zeros(
        (1, 1, 1, sequence_length),
        device=device,
        dtype=torch.float16,
    )
    pred_zero = -nll_singles(
        model=model,
        my_input=zero_input,
    )

    # save mfi values
    path = f"outputs/mfi/{num_factors}_factor/{value_to_save}"
    os.makedirs(path, exist_ok=True)

    torch.save(
        pred_zero,
        os.path.join(
            path,
            f"preds_batch_{args.num_factors}_zero.pt",
        ),
    )

    preds = torch.empty(
        min(save_every * loader.batch_size, loader.dataset_length),
        device=device,
        dtype=torch.float16,
    )

    for idx in tqdm(range(loader.num_batches)):
        loader.get_curr_batch_size()
        idx_since_save = idx % save_every

        preds[
            idx_since_save
            * loader.len_curr_batch : (idx_since_save + 1)
            * loader.len_curr_batch
        ] = -nll_singles(
            model=model,
            my_input=next(loader),
        )

        if idx % save_every != save_every - 1 and idx != (loader.num_batches) - 1:
            continue

        torch.save(
            preds,
            os.path.join(
                path,
                f"preds_batch_{args.num_factors}_{idx // save_every}.pt",
            ),
        )
        processed = idx * loader.batch_size

        preds = torch.empty(
            min(
                save_every * loader.batch_size,
                int(loader.dataset_length - processed),
            ),
            device=device,
            dtype=torch.float16,
        )
    return idx // save_every


def load_model_eval(
    sequence_length,
    hidden_dims,
    num_layers,
    n_masks,
    log_dir,
    val_epoch,
    device,
):
    # load the model
    model = models.MADE(
        input_dim=sequence_length,
        hidden_dims=[hidden_dims] * num_layers,
        n_masks=n_masks,
    ).to(device)

    checkpoint = torch.load(
        os.path.join(log_dir, f"trainer_state_{val_epoch}.ckpt"),
        map_location=device,
    )
    model_state_dict = {
        k: v for k, v in checkpoint["model"].items() if k in model.state_dict()
    }
    model.load_state_dict(model_state_dict)
    model = model.eval()
    model.compile()

    return model


def print_mfi_stats(mfi, sequence_length, num_factors):
    print("Average value: ", torch.mean(mfi).item())
    print("Standard deviation: ", torch.std(mfi).item())
    print("Median value: ", torch.median(mfi).item())
    print("\n")

    print(
        "Most interacting genes: ",
        more_itertools.nth_combination(
            range(len(range(sequence_length))), num_factors, torch.argmax(mfi).item()
        ),
    )

    print("Interaction value: ", torch.max(mfi).item())
    print("\n")
    print(
        "Least interacting genes: ",
        more_itertools.nth_combination(
            range(len(range(sequence_length))), num_factors, torch.argmin(mfi).item()
        ),
    )
    print("Interaction value: ", torch.min(mfi).item())


def plot_mfi_hist(mfi, bins=10000):
    # Create a seaborn histogram
    sns.histplot(mfi, bins=bins, color="skyblue")

    # Add labels and title
    plt.xlabel("Log Values")
    plt.ylabel("Frequency")
    plt.title("Histogram of Floats")

    # Show the plot
    plt.show()


def two_factor_mfi(preds, pred_zero):
    assert preds.shape[1] == 3
    pred_1_1 = preds[:, 0]
    pred_1_0 = preds[:, 1]
    pred_0_1 = preds[:, 2]

    mfi = (pred_1_1 + pred_zero) - (pred_1_0 + pred_0_1)
    return mfi


def three_factor_mfi(preds, pred_zero):
    assert preds.shape[1] == 7
    pred_1_1_1 = preds[:, 0]
    pred_1_0_0 = preds[:, 1]
    pred_0_1_0 = preds[:, 2]
    pred_0_0_1 = preds[:, 3]
    pred_1_1_0 = preds[:, 4]
    pred_1_0_1 = preds[:, 5]
    pred_0_1_1 = preds[:, 6]
    numerator = pred_1_1_1 + pred_1_0_0 + pred_0_1_0 + pred_0_0_1
    denominator = pred_1_1_0 + pred_1_0_1 + pred_0_1_1 + pred_zero
    mfi = numerator - denominator
    return mfi


def four_factor_mfi(preds, pred_zero):
    # numerator
    preds_1_1_1_1 = preds[:, 0]
    preds_1_1_0_0 = preds[:, 1]
    preds_1_0_1_0 = preds[:, 2]
    preds_1_0_0_1 = preds[:, 3]
    preds_0_1_1_0 = preds[:, 4]
    preds_0_1_0_1 = preds[:, 5]
    preds_0_0_1_1 = preds[:, 6]

    # denominator
    preds_1_1_1_0 = preds[:, 7]
    preds_1_1_0_1 = preds[:, 8]
    preds_1_0_1_1 = preds[:, 9]
    preds_0_1_1_1 = preds[:, 10]
    preds_0_0_0_1 = preds[:, 11]
    preds_0_0_1_0 = preds[:, 12]
    preds_0_1_0_0 = preds[:, 13]
    preds_1_0_0_0 = preds[:, 14]

    numerator = (
        preds_1_1_1_1
        + preds_1_1_0_0
        + preds_1_0_1_0
        + preds_1_0_0_1
        + preds_0_1_1_0
        + preds_0_1_0_1
        + preds_0_0_1_1
        + pred_zero
    )
    denominator = (
        preds_1_1_1_0
        + preds_1_1_0_1
        + preds_1_0_1_1
        + preds_0_1_1_1
        + preds_0_0_0_1
        + preds_0_0_1_0
        + preds_0_1_0_0
        + preds_1_0_0_0
    )
    mfi = numerator - denominator
    return mfi


def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for Model Training")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1000,
        help="Length of the sequence",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help="Batch size (default: 10000)",
    )
    parser.add_argument(
        "--num_factors",
        type=int,
        default=2,
        help="Number of factors (default: 3)",
    )
    parser.add_argument(
        "--val_epoch",
        type=int,
        default=84,
        help="Validation epoch (default: 84)",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="tmp/run/made_sweep_learnings/",
        help="Log directory (default: ../tmp/run/made_sweep_learnings/)",
    )
    parser.add_argument(
        "--hidden_dims",
        type=int,
        default=2500,
        help="Hidden dimensions (default: 2500)",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=2,
        help="Number of layers (default: 2)",
    )
    parser.add_argument(
        "--n_masks",
        type=int,
        default=1,
        help="Number of masks (default: 1)",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=1000,
        help="Save every N batches.",
    )
    parser.add_argument(
        "--value_to_save",
        type=int,
        default=0,
        help="Save value x from the values.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    start_time = time.time()

    args = parse_arguments()

    if args.num_factors == 2:
        values_1_1 = [True, True]
        values_1_0 = [True, False]
        values_0_1 = [False, True]
        values_list = [values_1_1, values_1_0, values_0_1]
    elif args.num_factors == 3:
        values_1_1_1 = [True, True, True]
        values_1_0_0 = [True, False, False]
        values_0_1_0 = [False, True, False]
        values_0_0_1 = [False, False, True]
        values_1_1_0 = [True, True, False]
        values_1_0_1 = [True, False, True]
        values_0_1_1 = [False, True, True]
        values_list = [
            values_1_1_1,
            values_1_0_0,
            values_0_1_0,
            values_0_0_1,
            values_1_1_0,
            values_1_0_1,
            values_0_1_1,
        ]
    elif args.num_factors == 4:
        # numerator
        values_1_1_1_1 = [True, True, True, True]
        values_1_1_0_0 = [True, True, False, False]
        values_1_0_1_0 = [True, False, True, False]
        values_1_0_0_1 = [True, False, False, True]
        values_0_1_1_0 = [False, True, True, False]
        values_0_1_0_1 = [False, True, False, True]
        values_0_0_1_1 = [False, False, True, True]

        # denominator
        values_1_1_1_0 = [True, True, True, False]
        values_1_1_0_1 = [True, True, False, True]
        values_1_0_1_1 = [True, False, True, True]
        values_0_1_1_1 = [False, True, True, True]
        values_0_0_0_1 = [False, False, False, True]
        values_0_0_1_0 = [False, False, True, False]
        values_0_1_0_0 = [False, True, False, False]
        values_1_0_0_0 = [True, False, False, False]
        values_list = [
            # numerator
            values_1_1_1_1,
            values_1_1_0_0,
            values_1_0_1_0,
            values_1_0_0_1,
            values_0_1_1_0,
            values_0_1_0_1,
            values_0_0_1_1,
            # denominator
            values_1_1_1_0,
            values_1_1_0_1,
            values_1_0_1_1,
            values_0_1_1_1,
            values_0_0_0_1,
            values_0_0_1_0,
            values_0_1_0_0,
            values_1_0_0_0,
        ]
    elif args.num_factors == 5:
        placeholder_value = [True, True, True, True, True]
        # (5 choose 4) + (5 choose 3) + (5 choose 2) + (5 choose 1) + 1
        values_list = [placeholder_value] * 31
        print(len(values_list))
    else:
        raise ValueError("Unknown number of factors.")

    value = (
        torch.Tensor(values_list[args.value_to_save]).to(args.device).bool()
    )  # , device=args.device)

    # assert len(values_list) == (3 if args.num_factors == 2 else 7)
    print("Creating dataloaders..")
    loader = MfiDatasetIterator(
        value=value,
        num_features=args.sequence_length,
        batch_size=args.batch_size,
        device=args.device,
        num_factors=args.num_factors,
    )

    print("loading model..")
    if args.sequence_length == 1000:
        model = load_model_eval(
            args.sequence_length,
            args.hidden_dims,
            args.num_layers,
            args.n_masks,
            args.log_dir,
            args.val_epoch,
            args.device,
        )
    else:
        print("DEBUGGING")
        # load the model
        model = models.MADE(
            input_dim=args.sequence_length,
            hidden_dims=[args.hidden_dims] * args.num_layers,
            n_masks=args.n_masks,
        )
        model = model.to(args.device)

    model = model.half()
    print(model.parameters().__next__().dtype)

    print("Running predictions..")
    max_save_counter = pred(
        model=model,
        sequence_length=args.sequence_length,
        loader=loader,
        device=args.device,
        value_to_save=args.value_to_save,
        num_factors=args.num_factors,
        save_every=args.save_every,
    )
    print("max_save_counter", max_save_counter)
    end_time = time.time()
    print("Execution Time:", end_time - start_time, "seconds")
