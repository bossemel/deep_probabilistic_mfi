import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from torch.utils.data import DataLoader
import wandb
import torch
import numpy as np
from torch.utils.data import random_split
from scipy.stats import bernoulli

from src.model import likelihood
from src.experiment import run_experiment

from src.model import loss_fn
from tqdm import tqdm


def calculate_mse(differences):
    """
    Calculate the Mean Squared Error (MSE) given a list of differences.

    Arguments:
    differences: A list of differences.

    Returns:
    mse: The Mean Squared Error (MSE).
    """
    if len(differences) == 0:
        return 0  # Handle empty list case

    squared_errors = [(diff**2) for diff in differences]
    mse = sum(squared_errors) / len(differences)
    return mse


class CorrelatedBernoulli:
    def __init__(self, p_1, p_2_given_1, p_3):
        self.p_1 = p_1
        self.p_2_given_1 = p_2_given_1  # probability of conditional bernouli distribution (a conditional of any bernoulli is a bernoulli)
        self.p_3 = p_3

    def pdf(self, x_1, x_2, x_3):  # created to be applied in sampling method below
        def p(x, p):  # p is the pdf with p arg
            return p if x == 1 else 1 - p  # prob( x=1 ) = p , prob(x=0) = 1-p

        def p_cond(input1, input2, p):  # p_cond()
            if input2 == 1:
                return (
                    p if input1 == 1 else 1 - p
                )  # prob( x2 = 1 | x1 = 1) = p  & prob( x2 = 1 | x1 = 0) = 1-p
            if input2 == 0:
                return (
                    1 - p if input1 == 1 else p
                )  # prob( x2 = 0 | x1 = 1) = 1-p  & prob( x2 = 0 | x1 = 0) = p
                # This means that prob( x2 = 0 | x1 ) has the opposite shape than prob( x2 = 1 | x1 ) -> ?Why is this the case, why was it chosen like this?

        return p(x_1, self.p_1) * p_cond(x_2, x_1, self.p_2_given_1) * p(x_3, self.p_3)

    def sample(self, observations):
        #  observattions = number of observations
        x_1 = bernoulli.rvs(p=self.p_1, size=(observations, 1))

        x_2 = np.zeros(x_1.shape)
        for idx, obs in enumerate(x_1):
            if obs == 1:
                x_2[idx] = bernoulli.rvs(p=self.p_2_given_1, size=(1, 1))
            elif obs == 0:
                x_2[idx] = bernoulli.rvs(p=1 - self.p_2_given_1, size=(1, 1))

        x_3 = bernoulli.rvs(p=self.p_3, size=(observations, 1))
        return np.concatenate([x_1, x_2, x_3], axis=1)


class CondIndepBernoulli:
    def __init__(self, p_1, p_2_given_1, p_3_given_1):
        self.p_1 = p_1
        self.p_2_given_1 = p_2_given_1
        self.p_3_given_1 = p_3_given_1

    def pdf(self, x_1, x_2, x_3):
        def p(x, p):
            return p if x == 1 else 1 - p

        def p_cond(input1, input2, p):
            if input2 == 1:
                return p if input1 == 1 else 1 - p
            if input2 == 0:
                return 1 - p if input1 == 1 else p

        return (
            p(x_1, self.p_1)
            * p_cond(x_2, x_1, self.p_2_given_1)
            * p_cond(x_3, x_1, self.p_3_given_1)
        )

    def sample(self, observations):
        x_1 = bernoulli.rvs(p=self.p_1, size=(observations, 1))
        # torch.empty((observations)).bernoulli_(p=self.p_1)

        x_2 = np.zeros(x_1.shape)
        for idx, obs in enumerate(x_1):
            if obs == 1:
                x_2[idx] = bernoulli.rvs(p=self.p_2_given_1, size=(1, 1))
            elif obs == 0:
                x_2[idx] = bernoulli.rvs(p=1 - self.p_2_given_1, size=(1, 1))

        x_3 = np.zeros(x_1.shape)
        for idx, obs in enumerate(x_1):
            if obs == 1:
                x_3[idx] = bernoulli.rvs(p=self.p_3_given_1, size=(1, 1))
            elif obs == 0:
                x_3[idx] = bernoulli.rvs(p=1 - self.p_3_given_1, size=(1, 1))
        return np.concatenate([x_1, x_2, x_3], axis=1)


def gen_bernoulli_data_corr(train_path, val_path, test_path, batch_size):
    mv_bernoulli = CorrelatedBernoulli(0.5, 0.9, 0.1)

    bernoulli_samples = mv_bernoulli.sample(5000).astype(float)

    # Define the file path
    file_path = "../outputs/datasets/bernoulli_correlated.csv"

    # Save the NumPy array to a CSV file
    np.savetxt(file_path, bernoulli_samples, delimiter=",")
    bernoulli_samples = np.loadtxt(file_path, delimiter=",")

    bernoulli_samples = bernoulli_samples[:, np.newaxis, np.newaxis, :]

    train_size = int(len(bernoulli_samples) * 0.6)
    test_size = len(bernoulli_samples) - train_size
    train_data, test_data = random_split(bernoulli_samples, [train_size, test_size])

    test_size = int(len(test_data) * 0.5)
    val_size = len(test_data) - test_size
    test_data, val_data = random_split(test_data, [test_size, val_size])

    # Save the NumPy array to a CSV file
    # Define the file path
    np.savetxt(train_path, np.array(train_data)[:, 0, 0, :], delimiter=",")
    np.savetxt(val_path, np.array(val_data)[:, 0, 0, :], delimiter=",")
    np.savetxt(test_path, np.array(test_data)[:, 0, 0, :], delimiter=",")


def gen_bernoulli_data_cond(train_path, val_path, test_path, batch_size):
    mv_bernoulli = CondIndepBernoulli(0.5, 0.9, 0.8)

    bernoulli_samples = mv_bernoulli.sample(5000).astype(float)

    bernoulli_samples = bernoulli_samples[:, np.newaxis, np.newaxis, :]

    train_size = int(len(bernoulli_samples) * 0.6)
    test_size = len(bernoulli_samples) - train_size
    train_data, test_data = random_split(bernoulli_samples, [train_size, test_size])

    test_size = int(len(test_data) * 0.5)
    val_size = len(test_data) - test_size
    test_data, val_data = random_split(test_data, [test_size, val_size])

    # Save the NumPy array to a CSV file
    # Define the file path
    np.savetxt(train_path, np.array(test_data)[:, 0, 0, :], delimiter=",")

    np.savetxt(val_path, np.array(val_data)[:, 0, 0, :], delimiter=",")

    np.savetxt(test_path, np.array(test_data)[:, 0, 0, :], delimiter=",")


if __name__ == "__main__":

    seed = 4
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_path = "../outputs/datasets/bernoulli_corr/train.csv"
    val_path = "../outputs/datasets/bernoulli_corr/val.csv"
    test_path = "../outputs/datasets/bernoulli_corr/test.csv"
    batch_size = 128
    generate_new_data = False

    train_data = np.loadtxt(train_path, delimiter=",")
    train_data = train_data[:, np.newaxis, np.newaxis, :]

    val_data = np.loadtxt(val_path, delimiter=",")
    val_data = val_data[:, np.newaxis, np.newaxis, :]

    test_data = np.loadtxt(test_path, delimiter=",")
    test_data = test_data[:, np.newaxis, np.newaxis, :]

    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size, shuffle=True)

    from src.model import loss_fn

    n_epochs = 20
    batch_size = 512
    learning_rate = 0.01
    step_size = 100

    log_dir = "../tmp/run/genes_3"

    device = "cpu"
    use_wandb = False
    settings = wandb.Settings(disable_job_creation=True)

    num_runs = 100
    error_corr = []
    error_cond = []
    for run in tqdm(range(num_runs)):

        # @Todo: fix logging
        model_trainer = run_experiment(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            log_dir="../tmp/run/genes_5",
            n_gpus=0,
            device_id=device,
            sequence_length=3,
            hidden_dims=6,
            n_masks=1,
            num_layers=1,
            use_wandb=use_wandb,
            loss_fn=loss_fn,
            batch_size=batch_size,
            learning_rate=learning_rate,
            step_size=step_size,
            restore=False,
        )
        model = model_trainer.model

        # data_samples = model.sample(500)[:, 0, 0, :]
        # print(data_samples.shape)
        # gt_samples = train_data[:, 0, 0, :]
        # fig = correlation_matrices(data_samples, gt_samples)

        # in flow:
        # assume p_3 known and correclty estimated
        # separtely:
        p_3 = (
            likelihood(model, torch.Tensor([[[0.0, 0.0, 0.0]]]))
            + likelihood(model, torch.Tensor([[[0.0, 1.0, 0.0]]]))
            + likelihood(model, torch.Tensor([[[1.0, 0.0, 0.0]]]))
            + likelihood(model, torch.Tensor([[[1.0, 1.0, 0.0]]]))
        )

        px1_x2 = likelihood(model, torch.Tensor([[[0.0, 0.0, 0.0]]])) + likelihood(
            model, torch.Tensor([[[0.0, 0.0, 1.0]]])
        )
        separately = px1_x2 * p_3

        # together:
        p_x123 = likelihood(model, torch.Tensor([[[0.0, 0.0, 0.0]]]))
        together = p_x123

        error_corr.append((separately - together).detach().cpu())

        # ### Training on Toy Model 2:
        # ### $  p(x_1,x_2,x_3) =  p(x_1) \times p(x_2|x_1) \times p(x_3|x_1) $

        train_path = "../outputs/datasets/bernoulli_cond/train.csv"
        val_path = "../outputs/datasets/bernoulli_cond/val.csv"
        test_path = "../outputs/datasets/bernoulli_cond/test.csv"
        generate_new_data = False
        # @Todo: save dataset and only load later

        train_data = np.loadtxt(train_path, delimiter=",")
        train_data = train_data[:, np.newaxis, np.newaxis, :]

        val_data = np.loadtxt(val_path, delimiter=",")
        val_data = val_data[:, np.newaxis, np.newaxis, :]

        test_data = np.loadtxt(test_path, delimiter=",")
        test_data = test_data[:, np.newaxis, np.newaxis, :]

        train_loader = DataLoader(train_data, batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size, shuffle=False)

        settings = wandb.Settings(disable_job_creation=True)

        # @Todo: fix logging
        trainer = run_experiment(
            train_loader=train_loader,
            val_loader=val_loader,
            n_epochs=n_epochs,
            log_dir="../tmp/run/genes_5",
            n_gpus=0,
            device_id=device,
            sequence_length=3,
            hidden_dims=6,
            n_masks=1,
            num_layers=1,
            use_wandb=use_wandb,
            loss_fn=loss_fn,
            batch_size=batch_size,
            learning_rate=learning_rate,
            step_size=step_size,
            restore=False,
        )

        model = trainer.model

        model.eval()
        model = model.to("cpu")
        # data_samples = model.sample(500)[:, 0, 0, :]
        # print(data_samples.shape)
        # gt_samples = test_data[:, 0, 0, :]
        # fig = correlation_matrices(data_samples, gt_samples)

        # in flow:
        # assume p_3 known and correclty estimated
        # separtely:
        p_3_given_1 = (  # p_3_given_1 = p(x1,x3) / p(x1) = p(x1, x2 =0, x3 ) + p(x1, x2 =0, x3 ) /  p(x1)
            likelihood(model, torch.Tensor([[[0.0, 0.0, 0.0]]]))
            + likelihood(model, torch.Tensor([[[0.0, 1.0, 0.0]]]))
        ) / (
            likelihood(model, torch.Tensor([[[0.0, 0.0, 0.0]]]))
            + likelihood(model, torch.Tensor([[[0.0, 1.0, 0.0]]]))
            + likelihood(model, torch.Tensor([[[0.0, 0.0, 1.0]]]))
            + likelihood(model, torch.Tensor([[[0.0, 1.0, 1.0]]]))
        )

        px1_x2 = likelihood(
            model, torch.Tensor([[[0.0, 0.0, 0.0]]])
        ) + likelihood(  # px1 * p(x2|x1) = p(x1,x2)
            model, torch.Tensor([[[0.0, 0.0, 1.0]]])
        )
        separately = px1_x2 * p_3_given_1

        # together:
        p_x123 = likelihood(model, torch.Tensor([[[0.0, 0.0, 0.0]]]))
        together = p_x123

        error_cond.append((separately - together).detach().cpu())

    mean_err_corr = np.array(error_corr).mean()
    std_err_corr = np.array(error_corr).std()
    mean_err_cond = np.array(error_cond).mean()
    std_err_cond = np.array(error_cond).std()

    print("Mean of error_corr:", mean_err_corr)
    print("Standard Deviation of error_corr:", std_err_corr)
    print("Mean of error_cond:", mean_err_cond)
    print("Standard Deviation of error_cond:", std_err_cond)

    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create a boxplot using Seaborn
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.boxplot(data=mean_err_corr, palette="Set3")
    plt.title("Boxplot of error_corr and error_cond")
    plt.ylabel("Value")
    plt.xlabel("Variables")
    plt.show()
