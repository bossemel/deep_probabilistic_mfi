import torch
from torch.nn import functional as F


def loss_fn(x, _, preds):
    batch_size = x.shape[0]
    x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
    preds = preds  # .float()
    loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
    return loss.sum(dim=1).sum()


def nll_singles(model, my_input):
    preds = model.forward(my_input).detach()
    batch_size = my_input.shape[0]
    my_input, preds = my_input.view((batch_size, -1)), preds.view((batch_size, -1))
    prob = F.binary_cross_entropy_with_logits(preds, my_input, reduction="none").sum(
        dim=1
    )
    return prob


def nll(model, my_input):
    sample = torch.tensor(my_input)
    forward = model.forward(sample)
    loss = loss_fn(sample, None, forward)
    loss = loss / sample.shape[0]
    return loss


def likelihood(model, my_input):
    loss = nll(model, my_input)
    prob = torch.exp(-loss)
    return prob
