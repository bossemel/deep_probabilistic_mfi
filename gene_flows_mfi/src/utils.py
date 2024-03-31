import argparse

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def correlation_matrices(pred, gt):
    # Create a figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

    # Compute the correlation matrix
    corr_pred = np.corrcoef(pred, rowvar=False)
    corr_gt = np.corrcoef(gt, rowvar=False)
    corr_diff = np.abs(corr_gt - corr_pred)

    # Create a mask to only show the lower triangle of the matrix
    mask = np.triu(np.ones_like(corr_pred, dtype=bool))
    sns.set(font_scale=1.5)

    # Set font sizes
    title_fontsize = 16
    label_fontsize = 16
    tick_fontsize = 16

    # Create a heatmap using seaborn
    sns.heatmap(
        corr_gt,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=False,
        yticklabels=False,
        ax=axes[0],
        cbar=False,
    )
    axes[0].set_title("Ground Truth", fontsize=title_fontsize)
    axes[0].set_xlabel("Features", fontsize=label_fontsize)
    axes[0].set_ylabel("Features", fontsize=label_fontsize)

    # Create a heatmap using seaborn
    sns.heatmap(
        corr_pred,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=False,
        yticklabels=False,
        ax=axes[1],
        cbar=True,
    )
    axes[1].set_title("Prediction", fontsize=title_fontsize)
    axes[1].set_xlabel("Features", fontsize=label_fontsize)

    # Create a heatmap using seaborn
    max_val = 0
    for row in corr_diff:
        for col in row:
            if col != np.nan:
                if col > max_val:
                    max_val = col
    print(max_val)
    sns.heatmap(
        corr_diff,
        mask=mask,
        cmap="coolwarm",
        vmin=0,
        vmax=max_val,
        center=0,
        xticklabels=False,
        yticklabels=False,
        ax=axes[2],
    )
    axes[2].set_title("Absolute Difference", fontsize=title_fontsize)
    axes[2].set_xlabel("Features", fontsize=label_fontsize)

    # Set tick font size
    for ax in axes:
        ax.tick_params(axis="both", labelsize=tick_fontsize)

    return fig


def single_corr_matrix(samples):
    correlation_matrix = np.corrcoef(samples, rowvar=False)

    # Create a mask to only show the lower triangle of the matrix
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

    # Create a heatmap using seaborn
    plt.figure(figsize=(2, 2))
    sns.heatmap(
        correlation_matrix,
        mask=mask,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        center=0,
        xticklabels=False,
        yticklabels=False,
    )
    plt.title("Correlation Matrix")
    plt.xlabel("Features")
    plt.ylabel("Features")
    plt.show()


def get_mfi_input_values(num_factors):
    match num_factors:

        case 1:
            values_1 = [True]
            values_list = [values_1]
        case 2:
            values_1_1 = [True, True]
            values_1_0 = [True, False]
            values_0_1 = [False, True]
            values_list = [values_1_1, values_1_0, values_0_1]
        case 3:
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
        case 4:
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
        case 5:
            placeholder_value = [True, True, True, True, True]  # --> 31 days

            # 31 gpus * 1 day + loading of values
            # 31 * 24 = 744 --> 10% of cdt budget

            # (5 choose 4) + (5 choose 3) + (5 choose 2) + (5 choose 1) + 1
            values_list = [placeholder_value] * 31
        case _:
            raise ValueError(f"Unknown number of factors: {num_factors}")
    return values_list
