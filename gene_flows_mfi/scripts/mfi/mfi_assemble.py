#!/usr/bin/env python
# coding: utf-8


import os
import sys

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import time
import torch
import time
import numpy as np

from src.mfi import (
    calculate_significance,
    confirm_values,
    initialize_tensors,
    mfi_and_variance,
    mfi_hist_plot,
    save_mfi_tensors,
)
from src.utils import get_mfi_input_values


if __name__ == "__main__":
    # @Todo: make a config.yaml for these values and reuse between prediction and assembly
    num_factors = 2
    sequence_length = 1000
    batch_size = 100000
    save_every = 10000000000  # 1000
    batch_size = 1000
    device = "cpu"
    base_path = "../../outputs/mfi/"
    save_counter = 0

    start_time = time.time()
    values_list = get_mfi_input_values(num_factors)
    top_values = []
    top_indices = []

    # find dataset length and maximum index of saved tensor
    dataset_length = math.comb(
        sequence_length,
        num_factors,
    )
    max_save_counter = math.ceil(dataset_length / (batch_size * save_every))

    # retrieve zero prediction
    file_path = os.path.join(
        base_path,
        f"{num_factors}_factor",
        "0",
        f"preds_batch_{num_factors}_zero.pt",
    )
    assert os.path.isfile(file_path), f"{file_path}"
    pred_zero = torch.load(file_path, map_location=device).double()
    assert pred_zero.shape[0] > 0
    assert not torch.isnan(torch.sum(pred_zero))

    # @Todo: watch out, this assumes only one save in total.
    # for save_counter in tqdm(range(max_save_counter)):
    for idx, value in enumerate(range(len(values_list))):
        file_path = os.path.join(
            base_path,
            f"{num_factors}_factor",
            str(value),
            f"preds_batch_{num_factors}_{save_counter}.pt",
        )

        if not os.path.exists(file_path):
            raise ValueError(f"Cant find file {file_path}")

        # load tensor for value
        preds_tensor = torch.load(file_path, map_location=device)

        assert preds_tensor.shape[0] > 0
        assert not torch.isnan(torch.sum(preds_tensor))

        if idx == 0:
            preds = torch.empty(
                (preds_tensor.shape[0], len(values_list)),
                device=device,
                dtype=torch.float,
            )

        # @Todo: when allowing multiple saves, this needs to be assigned to specific rows or only loaded once
        preds[:, value] = preds_tensor

    assert not torch.sum(preds) == 0

    end_time = time.time()
    print("Execution Time:", end_time - start_time, "seconds")

    preds_exp = torch.exp(preds)
    assert torch.min(preds_exp) >= 0, f"{torch.min(preds_exp) }"
    assert torch.max(preds_exp) <= 1, f"{torch.max(preds_exp) }"

    # add pred zero to preds
    preds_total = torch.zeros(
        (preds.shape[0], preds.shape[1] + 1), device=device, dtype=torch.float64
    )
    preds_total[:, 0] = (
        torch.ones((preds.shape[0]), device=device, dtype=torch.float64) * pred_zero
    )
    preds_total[:, 1:] = preds

    # cast to float64
    log_preds = preds_total.double()

    # add zero tensor to values list
    values_list = [[False, False], *values_list]

    # @Todo: make this nicer
    num_values = 2**num_factors

    # initalize empty tensors
    (
        log_distr_sums,
        log_marginals_list,
        log_conditionals_list,
        log_mfi_list,
        log_var_list,
    ) = initialize_tensors(
        length=log_preds.shape[0],
        num_values=num_values,
        device=device,
    )

    # Calculate number of batches
    num_batches = math.ceil(log_preds.shape[0] / batch_size)

    # Loop over batches
    for batch_idx in tqdm(range(num_batches)):
        # Calculate start and end indices for this batch
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, log_preds.shape[0] + 1)

        # Get batch of log predictions
        log_pred = log_preds[start_idx:end_idx, :]

        # calculate variance
        log_marginal, log_mfi, log_conditional, log_var = mfi_and_variance(
            log_pred, num_factors=num_factors
        )
        assert log_pred.dtype == torch.float64
        assert log_marginal.dtype == torch.float64
        assert log_mfi.dtype == torch.float64
        assert log_conditional.dtype == torch.float64
        assert log_pred.shape == log_conditional.shape == log_marginal.shape

        log_conditionals_list[start_idx:end_idx, :] = log_conditional
        log_marginals_list[start_idx:end_idx, :] = log_marginal
        log_mfi_list[start_idx:end_idx] = log_mfi
        log_var_list[start_idx:end_idx] = log_var.squeeze()

    confirm_values(log_conditionals_list, log_marginal, log_mfi)

    log_distr = log_distr_sums  # @Todo: check this

    mask = (log_distr >= -1e-14) & (log_distr <= 1e-14)
    indices = torch.nonzero(mask).squeeze().tolist()
    print(
        "Percentage of conditionals summing with error < 0.000001",
        len(indices) / len(log_distr_sums),
    )

    # caclaulte significance
    log_mfi = log_mfi_list
    num_samples = 15296
    log_std, p_value, indices_sign, percentage_sign = calculate_significance(
        log_var_list=log_var_list,
        num_samples=num_samples,
        log_mfi=log_mfi,
    )

    # Save tensors
    output_dir = "../../outputs/mfi/significance" + str(num_factors)
    # Ensure the directory exists, creating it if necessary
    save_mfi_tensors(
        output_dir=output_dir,
        log_distr_sums=log_distr_sums,
        log_marginals_list=log_marginals_list,
        log_conditionals_list=log_conditionals_list,
        log_mfi_list=log_mfi_list,
        log_var_list=log_var_list,
        p_value=p_value,
    )

    mfi_hist_plot(log_mfi, log_var_list, p_value)
