import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calc_log_marginal(
    log_pred,
):
    log_marginal = torch.logsumexp(log_pred, axis=-1, keepdim=True)
    log_marginal = torch.tile(log_marginal, (1, log_pred.shape[-1]))

    return log_marginal


def calc_log_conditional(
    log_pred,
    log_marginal,
):
    log_conditional_list = torch.cat(
        [log_pred.unsqueeze(-1), -log_marginal.unsqueeze(-1)], axis=-1
    )
    log_conditional = torch.sum(log_conditional_list, axis=-1, keepdim=False)

    return log_conditional


def two_factor_mfi(
    log_pred,
):
    pred_0_0 = log_pred[:, 0]
    pred_1_1 = log_pred[:, 1]
    pred_1_0 = log_pred[:, 2]
    pred_0_1 = log_pred[:, 3]
    mfi = (pred_1_1 + pred_0_0) - (pred_1_0 + pred_0_1)
    return mfi


def three_factor_mfi(log_pred):
    assert log_pred.shape[1] == 8
    pred_0_0_0 = log_pred[:, 0]
    pred_1_1_1 = log_pred[:, 1]
    pred_1_0_0 = log_pred[:, 2]
    pred_0_1_0 = log_pred[:, 3]
    pred_0_0_1 = log_pred[:, 4]
    pred_1_1_0 = log_pred[:, 5]
    pred_1_0_1 = log_pred[:, 6]
    pred_0_1_1 = log_pred[:, 7]
    numerator = pred_1_1_1 + pred_1_0_0 + pred_0_1_0 + pred_0_0_1
    denominator = pred_1_1_0 + pred_1_0_1 + pred_0_1_1 + pred_0_0_0
    mfi = numerator - denominator
    return mfi


def four_factor_mfi(log_pred):
    assert log_pred.shape[1] == 16
    # numerator
    preds_0_0_0_0 = log_pred[:, 0]
    preds_1_1_1_1 = log_pred[:, 1]
    preds_1_1_0_0 = log_pred[:, 2]
    preds_1_0_1_0 = log_pred[:, 3]
    preds_1_0_0_1 = log_pred[:, 4]
    preds_0_1_1_0 = log_pred[:, 5]
    preds_0_1_0_1 = log_pred[:, 6]
    preds_0_0_1_1 = log_pred[:, 7]

    # denominator
    preds_1_1_1_0 = log_pred[:, 8]
    preds_1_1_0_1 = log_pred[:, 9]
    preds_1_0_1_1 = log_pred[:, 10]
    preds_0_1_1_1 = log_pred[:, 11]
    preds_0_0_0_1 = log_pred[:, 12]
    preds_0_0_1_0 = log_pred[:, 13]
    preds_0_1_0_0 = log_pred[:, 14]
    preds_1_0_0_0 = log_pred[:, 15]

    numerator = (
        preds_1_1_1_1
        + preds_1_1_0_0
        + preds_1_0_1_0
        + preds_1_0_0_1
        + preds_0_1_1_0
        + preds_0_1_0_1
        + preds_0_0_1_1
        + preds_0_0_0_0
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


def save_mfi_tensors(
    output_dir,
    log_distr_sums,
    log_marginals_list,
    log_conditionals_list,
    log_mfi_list,
    log_var_list,
    p_value,
):
    os.makedirs(output_dir, exist_ok=True)

    torch.save(log_distr_sums, os.path.join(output_dir, "log_distr_sums.pt"))
    torch.save(log_marginals_list, os.path.join(output_dir, "log_marginals_list.pt"))
    torch.save(
        log_conditionals_list, os.path.join(output_dir, "log_conditionals_list.pt")
    )
    torch.save(log_mfi_list, os.path.join(output_dir, "log_mfi_list.pt"))
    torch.save(log_var_list, os.path.join(output_dir, "log_var_list.pt"))

    torch.save(p_value, os.path.join(output_dir, "p_value.pt"))


def mfi_hist_plot(log_mfi, log_var_list, p_value):
    # Increase the default font size
    sns.set(font_scale=1.5)

    # Create a figure and a 1x2 grid of subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Create the second histogram on the second subplot
    sns.histplot(log_mfi.cpu().numpy(), bins=100, kde=False, ax=axs[0], stat="percent")
    axs[0].set_xlabel("Values")
    axs[0].set_ylabel("Log Frequency in %")
    axs[0].set_title("MFI")
    axs[0].set_yscale("log")

    # Create the second histogram on the second subplot
    # log_var_items = [elem[0] for elem in log_var_list]
    sns.histplot(
        log_var_list.cpu().numpy(), bins=100, kde=False, ax=axs[1], stat="percent"
    )
    axs[1].set_xlabel("Values")
    axs[1].set_ylabel("Frequency in %")
    axs[1].set_title("MFI Variance")

    # Create the first histogram on the first subplot
    sns.histplot(p_value.cpu().numpy(), bins=100, kde=False, ax=axs[2], stat="percent")
    axs[2].set_xlabel("Values")
    axs[2].set_ylabel("Log Frequency in %")
    axs[2].set_title("P-values")
    axs[2].set_yscale("log")

    # fig.set_title("Normalized Histograms for MFI Variance and Significance")
    # Show the plot
    plt.tight_layout()
    # plt.show()

    fig.savefig("../../outputs/figures/mfi_variance_pvalue.png", dpi=300)


def calculate_significance(log_var_list, num_samples, log_mfi):
    # calcaulte adjusted standard deviation
    log_std = 0.5 * log_var_list
    std = torch.exp(log_std)
    std_adjusted = std / np.sqrt(num_samples)

    # run z-test
    z_statistic = log_mfi / std_adjusted
    p_value = 2 * (
        1 - torch.distributions.Normal(loc=0, scale=1).cdf(torch.abs(z_statistic).cpu())
    )

    print("Using CDF:")
    mask = p_value < 0.05
    indices_sign = torch.nonzero(mask).squeeze().tolist()
    percentage_sign = len(indices_sign) / len(p_value)
    print("Total: ", len(indices_sign))
    print("Percentage: ", percentage_sign)

    print("Using bounds:")
    # using formula directly
    lower_bound = log_mfi - 1.96 * std_adjusted
    upper_bound = log_mfi + 1.96 * std_adjusted
    mask_sign = (lower_bound > 0) | (upper_bound < 0)

    indices_sign = torch.nonzero(mask_sign).squeeze()
    percentage_sign = len(indices_sign) / p_value.shape[0]
    print("Total: ", len(indices_sign))
    print("Percentage: ", percentage_sign)

    return log_std, p_value, indices_sign, percentage_sign


def confirm_values(log_conditionals_list, log_marginal, log_mfi):
    for log_conditional_single in log_conditionals_list:
        exp_cond = torch.exp(log_conditional_single)

        assert (
            torch.min(exp_cond) >= 0 and torch.max(exp_cond) <= 1
        ), f"{torch.min(exp_cond)}, {torch.max(exp_cond)}"

    marginals = torch.exp(log_marginal)

    assert (
        torch.min(marginals) >= 0 and torch.max(marginals) <= 1
    ), f"{torch.min(marginals)}, {torch.max(marginals)}"

    exp_mfi = torch.exp(log_mfi)

    assert torch.min(exp_mfi) >= 0, f"{torch.min(exp_mfi)}, {torch.max(exp_mfi)}"


def mfi_and_variance(log_pred, num_factors):

    log_marginal = calc_log_marginal(
        log_pred=log_pred,
    )

    if num_factors == 2:
        log_mfi = two_factor_mfi(
            log_pred,
        )
    elif num_factors == 3:
        log_mfi = three_factor_mfi(
            log_pred,
        )
    elif num_factors == 4:
        log_mfi = four_factor_mfi(
            log_pred,
        )

    # log var needs to sum the conditionals
    log_conditional = calc_log_conditional(
        log_pred=log_pred,
        log_marginal=log_marginal,
    )

    log_var = torch.logsumexp(-log_conditional, axis=-1, keepdim=True)

    return log_marginal, log_mfi, log_conditional, log_var


def initialize_tensors(length, num_values, device):
    log_distr_sums = torch.zeros((length), device=device, dtype=torch.float64)
    log_marginals_list = torch.zeros(
        (length, num_values), device=device, dtype=torch.float64
    )
    log_conditionals_list = torch.zeros(
        (length, num_values), device=device, dtype=torch.float64
    )
    log_mfi_list = torch.zeros((length), device=device, dtype=torch.float64)
    log_var_list = torch.zeros((length), device=device, dtype=torch.float64)
    return (
        log_distr_sums,
        log_marginals_list,
        log_conditionals_list,
        log_mfi_list,
        log_var_list,
    )
