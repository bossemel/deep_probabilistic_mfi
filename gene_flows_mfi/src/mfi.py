import itertools
import math
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import DataLoader


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


# @Todo: implement Dataset and then use build-in dataloader instead
class MfiDatasetIterator(DataLoader):
    """DataLoader function to create all joint probabiliy values for the MFI calculation. Contains the self.dataset as
    a generator, and creates each batch on the fly in __next__. Num factors is 2 for 2-factor MFI, etc. Value is the
    current interaction value of interest. Num features is the number of columns in the dataset, in our case 1000.
    """

    def __init__(self, num_features, batch_size, value, device, num_factors, dtype):
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
        self.dtype = dtype
        self.current_batch = 0

        self.get_curr_batch_size()

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
        if self.current_batch == self.num_batches:
            raise StopIteration
        self.get_curr_batch_size()
        combs = torch.tensor(
            list(itertools.islice(self.dataset, self.len_curr_batch)),
            device=self.device,
            dtype=torch.int,
        )

        batch = torch.zeros(
            (self.len_curr_batch, self.num_features),
            device=self.device,
            dtype=torch.bool,
        )

        batch[
            torch.arange(self.len_curr_batch, device=self.device).unsqueeze(1),
            combs,
        ] = self.value

        batch = batch.unsqueeze(1).unsqueeze(1)
        self.current_batch += 1

        if self.dtype == "float16":
            batch = batch.half()
        elif self.dtype == "float64":
            batch = batch.double()
        elif self.dtype == "float32":
            batch = batch.float()
        else:
            raise ValueError("Unknown dtype.")

        return batch

    def __len__(self):
        return self.num_batches


def pred_mfi_components(
    model,
    sequence_length,
    loader,
    value_to_save,
    device,
    num_factors,
    save_every,
    dtype,
    autocast,
    output_dir,
):
    """Generates the joint probability over the dataset, allowing for intermediate saves."""
    criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    def nll_singles_loop(my_input):
        fwd = model.forward(my_input)
        my_input, fwd = my_input.squeeze((1, 2)), fwd.squeeze((1, 2))

        # Create an instance of BCEWithLogitsLoss
        loss = criterion(fwd, my_input)

        loss = loss.sum(dim=1)
        return loss

    # zero inputs:
    batch = torch.zeros(
        (1, 1, 1, sequence_length),
        device=device,
        dtype=dtype,
    )
    with torch.cuda.amp.autocast():
        preds_batch = -nll_singles_loop(
            my_input=batch,
        )
    # save mfi values
    path = os.path.join(output_dir, f"{num_factors}_factor/{value_to_save}")
    os.makedirs(path, exist_ok=True)

    torch.save(
        preds_batch,
        os.path.join(
            path,
            f"preds_batch_{num_factors}_zero.pt",
        ),
    )

    preds = torch.empty(
        min(save_every * loader.batch_size, loader.dataset_length),
        device=device,
        dtype=dtype,
    )

    last_assigned = 0
    processed = 0

    for idx, batch in enumerate(
        tqdm(loader, total=loader.num_batches)
    ):  # @Todo: consider using torch progress bar instead
        idx_since_save = idx % save_every
        if autocast:
            with torch.cuda.amp.autocast():
                preds_batch = -nll_singles_loop(my_input=batch)
        else:
            preds_batch = -nll_singles_loop(my_input=batch)

        processed += preds_batch.shape[0]

        if loader.len_curr_batch == loader.batch_size:
            preds[
                idx_since_save
                * loader.len_curr_batch : (idx_since_save + 1)
                * loader.len_curr_batch
            ] = preds_batch
            last_assigned = (idx_since_save + 1) * loader.len_curr_batch
        else:
            preds[last_assigned:] = preds_batch

        if idx % save_every == save_every - 1 or idx == (loader.num_batches) - 1:
            torch.save(
                preds,
                os.path.join(
                    path,
                    f"preds_batch_{num_factors}_{idx // save_every}.pt",
                ),
            )

            if idx != (loader.num_batches) - 1:
                preds = torch.empty(
                    min(
                        save_every * loader.batch_size,
                        loader.dataset_length - processed,
                    ),
                    device=device,
                    dtype=dtype,
                )
                last_assigned = 0

    return idx // save_every
