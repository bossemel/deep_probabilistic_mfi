import os
import sys

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import more_itertools
import math
import time
import torch

torch.set_default_dtype(torch.float64)


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
    print(preds)
    assert preds.shape[1] == 3
    assert not torch.isnan(torch.sum(preds))
    pred_1_1 = preds[:, 0]
    pred_1_0 = preds[:, 1]
    pred_0_1 = preds[:, 2]
    print("preds shape", preds.shape)
    print(pred_1_1)
    print(pred_1_0)
    print(pred_0_1)
    mfi = (pred_1_1 + pred_zero) - (pred_1_0 + pred_0_1)
    print("mfi shape", mfi.shape)
    assert not torch.isnan(torch.sum(mfi))
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


def gen_mfi_dict(num_factors, preds, pred_zero, device):
    if num_factors == 2:
        mfi = two_factor_mfi(preds, pred_zero)
        assert not torch.isnan(torch.sum(mfi))
    elif num_factors == 3:
        mfi = three_factor_mfi(preds, pred_zero)
    elif num_factors == 4:
        mfi = four_factor_mfi(preds, pred_zero)
    else:
        raise ValueError("Unknown number factors")
    assert not torch.isnan(torch.sum(mfi))

    # marginalize out genes
    marginal = [torch.ones(preds.shape[0], device=device) * (pred_zero).unsqueeze(0)]
    for idx in range(num_factors):
        marginal.append((preds[:, idx]).unsqueeze(0))

    marginal = torch.exp(torch.logsumexp(torch.cat(marginal), dim=0))

    # calculate variance
    var = torch.ones(preds.shape[0], device=device) * (
        marginal / ((torch.exp(pred_zero)))
    )
    for idx in range(num_factors):
        var += marginal / (torch.exp(preds[:, idx]))

    # run z-test
    standard_error = torch.sqrt(var)
    z_statistic = mfi / standard_error
    p_value = 2 * (
        1 - torch.distributions.Normal(loc=0, scale=1).cdf(torch.abs(z_statistic).cpu())
    )

    print("min standard_error: ", torch.min(standard_error).item())
    print("max z_statistic: ", torch.max(z_statistic).item())
    print("min p_value: ", torch.min(p_value).item())

    print("max standard_error: ", torch.max(standard_error).item())
    print("min z_statistic: ", torch.min(z_statistic).item())
    print("max p_value: ", torch.max(p_value).item())

    # @Todo: fix p values all 0.5
    assert not torch.isnan(torch.sum(p_value))

    alpha = 0.05
    sign_p = p_value < alpha

    lower_bound = mfi - 1.96 * var
    upper_bound = mfi + 1.96 * var

    assert not (
        torch.isnan(torch.sum(lower_bound)) or torch.isnan(torch.sum(upper_bound))
    )
    lower_below = lower_bound < 0
    upper_above = upper_bound > 0
    sign = torch.logical_and(lower_below, upper_above)
    print(
        f"""mfi: {mfi[0].item()}\n \\
        var: {var[0].item()}\n \\
        lower_bound: {lower_bound[0].item()}\n \\
        upper_bound: {upper_bound[0].item()}\n \\
        sign: {sign[0].item()}\n \\
        p_value: {p_value[0]}\n \\
        sign_p: {sign_p[0]}\n"""
    )

    return mfi, var, lower_bound, upper_bound, sign, sign_p


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
    base_path = "../../outputs/mfi/"
    file_path = os.path.join(
        base_path,
        f"{args.num_factors}_factor",
        "0",
        f"preds_batch_{args.num_factors}_zero.pt",
    )
    assert os.path.isfile(file_path), f"{file_path}"

    pred_zero = torch.load(file_path).to(args.device).double()
    assert pred_zero.shape[0] > 0
    assert not torch.isnan(torch.sum(pred_zero))

    dataset_length = math.comb(
        args.sequence_length,
        args.num_factors,
    )
    max_save_counter = math.ceil(dataset_length / (args.batch_size * args.save_every))

    print("max_save_counter", max_save_counter)
    # @Todo: double check whether save_coutner is correct.
    for save_counter in tqdm(range(max_save_counter)):
        for idx, value in enumerate(range(len(values_list))):
            # @Todo: implement alternative to finding the file path that is using
            # index_itertools = more_itertools.combination_index(genes[0], range(1000))
            file_path = os.path.join(
                base_path,
                f"{args.num_factors}_factor",
                value,
                f"preds_batch_{args.num_factors}_{save_counter}.pt",
            )

            if not os.path.exists(file_path):
                raise ValueError(f"Cant find file {file_path}")
            preds_tensor = torch.load(file_path).to(args.device)
            assert preds_tensor.shape[0] > 0
            assert not torch.isnan(torch.sum(preds_tensor))

            if idx == 0:
                preds = torch.empty(
                    (preds_tensor.shape[0], len(values_list)),
                    device=args.device,
                )

            preds[:, value] = preds_tensor

        assert not torch.sum(preds) == 0

        mfi_batch, var, lower_bound, upper_bound, sign, sign_p = gen_mfi_dict(
            num_factors=args.num_factors,
            preds=preds,
            pred_zero=pred_zero,
            device=args.device,
        )

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
        genes = (
            more_itertools.nth_combination(
                range(args.sequence_length),
                args.num_factors,
                index,
            ),
        )
        print(genes)
        print("index: ", more_itertools.combination_index(genes[0], range(1000)))

        # plot_mfi_hist(mfi, bins=10000)
        print("Most interacting genes: ", genes)
    end_time = time.time()
    print("Execution Time:", end_time - start_time, "seconds")
