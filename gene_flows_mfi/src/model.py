import os
import torch
from torch.nn import functional as F

from src.pytorch_generative.pytorch_generative import models


def loss_fn(x, _, preds):
    batch_size = x.shape[0]
    x, preds = x.view((batch_size, -1)), preds.view((batch_size, -1))
    preds = preds  # .float()
    loss = F.binary_cross_entropy_with_logits(preds, x, reduction="none")
    return loss.sum(dim=1).sum()


def nll_singles(model, my_input):
    preds = model.forward(my_input)  # .detach()
    batch_size = my_input.shape[0]
    my_input, preds = my_input.view((batch_size, -1)), preds.view((batch_size, -1))
    prob = F.binary_cross_entropy_with_logits(
        preds.reshape(-1, 1), my_input, reduction="none"
    ).sum(dim=1)
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


def load_model_eval(
    sequence_length,
    hidden_dims,
    num_layers,
    n_masks,
    log_dir,
    val_epoch,
    device,
    quantize,
    compile_model,
    resample_masks,
):
    # load the model
    model = models.MADE(
        input_dim=sequence_length,
        hidden_dims=[hidden_dims] * num_layers,
        n_masks=n_masks,
        device=device,
        resample_masks=resample_masks,
    ).to(device)

    checkpoint = torch.load(
        os.path.join(log_dir, f"trainer_state_{val_epoch}.ckpt"),
        map_location=device,
    )

    model_state_dict = {
        k: v for k, v in checkpoint["model"].items() if k in model.state_dict()
    }
    model.load_state_dict(model_state_dict)

    model_state_dict = {
        k: v for k, v in checkpoint["model"].items() if k in model.state_dict()
    }
    model.load_state_dict(model_state_dict)

    for param in model.parameters():
        param.grad = None
    model = model.eval()

    if quantize:
        # Specify quantization configuration
        model.qconfig = torch.quantization.get_default_qconfig("x86")

        # Prepare the model for static quantization
        model = torch.quantization.prepare(model, inplace=False)

        model = torch.ao.quantization.quantize_dynamic(
            model,  # the original model
            {torch.nn.Linear},  # a set of layers to dynamically quantize
            dtype=torch.quint8,
        )  # the target dtype for quantized weightsx

    if compile_model:
        model.compile()

    return model
