import numpy as np
from torch.utils.data import Dataset, random_split
import pandas as pd
import torch


class gene_dataset(Dataset):
    def __init__(self, data_path, transform=None, reorder_columns=None):
        super().__init__()
        pre_data = pd.read_csv(data_path)
        df = pre_data.drop("Index", axis=1)  # Remove index column from dataframe
        if reorder_columns != None:
            df = df[reorder_columns]
        self.data = df.to_numpy()
        self.data = np.where(self.data > 0, 1, 0).astype(float)

        # This is for made
        self.data = self.data[:, np.newaxis, np.newaxis, :]
        self.data = self.data.astype(np.float32)

        # self.data = (
        #     torch.eye(2)[self.data].squeeze().float()
        # )  ## One hot encoding of each vocab /

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_gene_datasets(train_path, test_path, reorder_columns=None):
    return gene_dataset(train_path, reorder_columns=reorder_columns), gene_dataset(
        test_path, reorder_columns=reorder_columns
    )


# @Todo: double check if we want to fix seed like this
def load_dataset(train_path, test_path, seed=4, reorder_columns=None):
    train_data, test_data = load_gene_datasets(
        train_path, test_path, reorder_columns=reorder_columns
    )

    # @Todo: fix seed
    train_size = int(len(train_data) * 0.8)
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    return train_data, val_data, test_data
