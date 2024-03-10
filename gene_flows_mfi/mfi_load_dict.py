import os
import sys

# module_path = os.path.abspath(os.path.join("../.."))
# if module_path not in sys.path:
#     sys.path.append(module_path)

import argparse
import numpy as np
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


class MfiDatasetIterator(DataLoader):
    def __init__(self, num_features, batch_size, values, device, num_factors):
        self.values = values
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

    def __iter__(self):
        return self

    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
        start_idx = self.current_batch * self.batch_size
        end_idx = min((self.current_batch + 1) * self.batch_size, self.dataset_length)

        assert end_idx > start_idx
        len_batch = end_idx - start_idx
        combs = torch.IntTensor([next(self.dataset) for _ in range(len_batch)]).to(
            self.device
        )

        batch = torch.zeros(
            len_batch,
            self.num_features,
            device=self.device,
            dtype=torch.float16,
        )
        for idx in range(self.num_factors):
            combs_idx = combs[:, idx]
            if self.values[idx] > 0:
                batch[torch.arange(batch.shape[0]), combs_idx] = 1

        self.current_batch += 1

        return batch

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
    loaders,
    values_list,
    device,
    num_factors,
    save_every,
):
    print("length value list", len(values_list))
    zero_input = torch.zeros(
        (1, 1, 1, sequence_length),
        device=device,
        dtype=torch.float16,
    )
    print("zero_input", zero_input.dtype)
    pred_zero = -nll_singles(
        model=model,
        my_input=zero_input,
    )

    # save mfi values
    path = f"outputs/mfi/{num_factors}_factor/"
    os.makedirs(path, exist_ok=True)

    torch.save(
        pred_zero,
        os.path.join(
            path,
            f"preds_batch_{args.num_factors}_zero.pt",
        ),
    )
    num_values = len(values_list)

    for value in range(num_values):
        save_counter = 0
        processed = 0

        preds = torch.empty(
            min(save_every * loaders[value].batch_size, loaders[value].dataset_length),
            device=device,
            dtype=torch.float16,  # @Todo: double check if we might want float64
        )

        # save mfi values
        path = f"outputs/mfi/{num_factors}_factor/{value}/"
        os.makedirs(path, exist_ok=True)

        start_idx = loaders[value].current_batch * loaders[value].batch_size
        end_idx = min(
            (loaders[value].current_batch + 1) * loaders[value].batch_size,
            loaders[value].dataset_length,
        )
        curr_batch_size = end_idx - start_idx

        for idx in tqdm(range(loaders[value].num_batches)):
            batch = next(loaders[value])

            processed += len(batch)
            preds_batch = -nll_singles(
                model=model,
                my_input=batch,
            )
            idx_since_save = idx % save_every

            preds[
                idx_since_save
                * curr_batch_size : (idx_since_save + 1)
                * curr_batch_size
            ] = preds_batch

            if (
                idx % save_every != save_every - 1
                and idx != (loaders[value].num_batches) - 1
            ):
                continue

            torch.save(
                preds,
                os.path.join(
                    path,
                    f"preds_batch_{args.num_factors}_{save_counter}.pt",
                ),
            )
            save_counter += 1

            if int(loaders[value].dataset_length - processed / num_values) == 0:
                continue

            preds = torch.empty(
                min(
                    save_every * loaders[value].batch_size,
                    int(loaders[value].dataset_length - processed),
                ),
                device=device,
                dtype=torch.float16,  # @Todo: double check if we might want float64
            )
    print(f"Done. Processed {processed}")
    return save_counter
    # return mfi


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

    checkpoint = torch.load(os.path.join(log_dir, f"trainer_state_{val_epoch}.ckpt"))
    model_state_dict = {
        k: v for k, v in checkpoint["model"].items() if k in model.state_dict()
    }
    model.load_state_dict(model_state_dict)
    model = model.eval()
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
        default=100,
        help="Length of the sequence",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=10000,
        help="Batch size (default: 10000)",
    )
    parser.add_argument(
        "--num_factors",
        type=int,
        default=3,
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
        default=100,
        help="Save every N batches.",
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
    else:
        raise ValueError("Unknown number of factors.")

    top_values = []
    top_indices = []

    # Iterate over each tensor
    counter = 0
    # save_counter = 0
    file_path = (
        f"outputs/mfi/{args.num_factors}_factor/preds_batch_{args.num_factors}_zero.pt"
    )
    pred_zero = torch.load(file_path).to(args.device)
    assert pred_zero.shape[0] > 0

    dataset_length = math.comb(
        args.sequence_length,
        args.num_factors,
    )
    max_save_counter = math.ceil(dataset_length / (args.batch_size * args.save_every))

    print("max_save_counter", max_save_counter)
    for save_counter in tqdm(range(max_save_counter)):
        preds = None

        for value in range(len(values_list)):
            file_path = f"outputs/mfi/{args.num_factors}_factor/{value}/preds_batch_{args.num_factors}_{save_counter}.pt"

            if not os.path.exists(file_path):
                raise ValueError(f"Cant find file {file_path}")
            preds_tensor = torch.load(file_path).to(args.device)
            assert preds_tensor.shape[0] > 0

            if preds == None:
                preds = torch.empty(
                    (preds_tensor.shape[0], len(values_list)),
                    device=args.device,
                )

            preds[:, value] = preds_tensor

        mfi_batch = gen_mfi_dict(
            num_factors=args.num_factors,
            preds=preds,
            pred_zero=pred_zero,
        ).cpu()

        # Find the top 5 values and indices within the current tensor
        values, indices = torch.topk(mfi_batch, k=5)
        assert len(values) > 0
        assert len(indices) > 0
        # Update the top values and indices if necessary
        top_values.extend(values)
        top_indices.extend(indices + counter)
        counter += mfi_batch.shape[0]

    assert len(top_values) > 0
    assert len(top_indices) > 0
    max_values, max_indices = torch.topk(torch.Tensor(top_values), 5)

    # Gather the indices of the maximum values using the index tensor
    highest_indices = torch.Tensor(top_indices)[max_indices]

    # Display the top 5 values and their corresponding indices
    for i, (value, index) in enumerate(zip(max_values, highest_indices)):
        print(f"Top {i+1} value: {value}, Index: {index}")
        # plot_mfi_hist(mfi, bins=10000)
        print(
            "Most interacting genes: ",
            more_itertools.nth_combination(
                range(args.sequence_length),
                args.num_factors,
                index,
            ),
        )
