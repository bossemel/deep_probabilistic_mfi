import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def split_test_and_save(
    subset_1_path,
    subset_2_path,
    output_directory,
    test_set_percentage=0.4,
    seed=4,
):
    # load both files
    subset_1 = pd.read_csv(subset_1_path)
    subset_1.rename(columns={"Unnamed: 0": "Index"}, inplace=True)

    subset_2 = pd.read_csv(Path(subset_2_path))
    subset_2.rename(columns={"Unnamed: 0": "Index"}, inplace=True)

    # create train and test set indices
    idx_list = subset_1.index.tolist()
    test_set_size = int(len(idx_list) * test_set_percentage)
    np.random.seed(seed)
    test_set_idxs = np.random.choice(idx_list, test_set_size, replace=False)
    train_set_idxs = np.setdiff1d(idx_list, test_set_idxs)  # Get the set difference
    assert len(train_set_idxs) + len(test_set_idxs) == len(
        idx_list
    ), f"{len(train_set_idxs)}, {len( test_set_idxs)}, {len(idx_list)}"

    # subset dataframe
    train_df = subset_1.iloc[train_set_idxs, :]
    test_df = subset_1.iloc[test_set_idxs, :]
    assert len(train_df) + len(test_df) == len(subset_1)

    # save test set dataframe
    outpath_test = Path(output_directory) / "test.csv"
    outpath_test.parent.mkdir(parents=True, exist_ok=True)
    test_df.to_csv(outpath_test, index=False)

    # concaneta remaining subset 1 data with subset 2 data
    merged_df = pd.concat([train_df, subset_2], ignore_index=True)
    outpath_train = Path(output_directory) / "train.csv"
    outpath_train.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(outpath_train, index=False)


if __name__ == "__main__":
    # get arguments
    parser = argparse.ArgumentParser(
        description="Split test script",
    )
    parser.add_argument(
        "--subset_1_path",
        type=str,
        help="Path to subset 1 CSV file",
    )
    parser.add_argument(
        "--subset_2_path",
        type=str,
        help="Path to subset 2 CSV file",
    )
    parser.add_argument("--output_directory", type=str, help="Output directory")
    parser.add_argument(
        "--test_set_percentage",
        type=float,
        default=0.4,
        help="Percentage of first dataset to be used for the test set",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=4,
        help="Random seed",
    )

    args = parser.parse_args()

    split_test_and_save(
        subset_1_path=args.subset_1_path,
        subset_2_path=args.subset_2_path,
        output_directory=args.output_directory,
        test_set_percentage=args.test_set_percentage,
        seed=args.seed,
    )
