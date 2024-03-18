import os
import sys

module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
from tqdm import tqdm
import math
import itertools
import time
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F

from src.pytorch_generative.pytorch_generative import models
from src.utils import get_mfi_input_values


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

        self.batch = torch.zeros(
            self.len_curr_batch,
            self.num_features,
            device=self.device,
            dtype=torch.bool,
        )

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
        combs = torch.tensor(
            list(itertools.islice(self.dataset, self.len_curr_batch)),
            device=self.device,
            dtype=torch.int,
        )

        self.batch = torch.zeros(
            self.len_curr_batch,
            self.num_features,
            device=self.device,
            dtype=torch.bool,
        )
        self.batch[
            torch.arange(self.len_curr_batch, device=self.device).unsqueeze(1), combs
        ] = self.value

        self.current_batch += 1

        self.batch = (
            self.batch.half() if self.dtype == "float16" else self.batch.float()
        )

    def __len__(self):
        return self.num_batches


def pred(
    model,
    sequence_length,
    loader,
    value_to_save,
    device,
    num_factors,
    save_every,
    dtype,
    autocast,
):
    """Generates the joint probability over the dataset, allowing for intermediate saves."""

    def nll_singles_loop(my_input):
        if device == "cuda" and autocast:
            with torch.autocast(device_type=device, dtype=torch.float16):
                fwd = model.forward(my_input)
        else:
            fwd = model.forward(my_input)
        prob = F.binary_cross_entropy_with_logits(
            fwd.squeeze(0).squeeze(0),
            my_input.squeeze(0).squeeze(0),
            reduction="none",
        ).sum(dim=1)
        return prob

    # zero inputs:
    batch = torch.zeros(
        (1, 1, 1, sequence_length),
        device=device,
        dtype=dtype,
    )
    preds_batch = -nll_singles_loop(
        my_input=batch,
    )

    # save mfi values
    path = os.path.join(args.output_dir, f"{num_factors}_factor/{value_to_save}")
    os.makedirs(path, exist_ok=True)

    torch.save(
        preds_batch,
        os.path.join(
            path,
            f"preds_batch_{args.num_factors}_zero.pt",
        ),
    )

    preds = torch.empty(
        min(save_every * loader.batch_size, loader.dataset_length),
        device=device,
        dtype=dtype,
    )

    last_assigned = 0
    processed = 0
    for idx in tqdm(range(loader.num_batches)):
        loader.get_curr_batch_size()
        idx_since_save = idx % save_every
        next(loader)

        preds_batch = -nll_singles_loop(
            my_input=loader.batch,
        )
        processed += preds_batch.shape[0]

        if loader.len_curr_batch == loader.batch_size:
            assert (
                last_assigned == (idx_since_save) * loader.len_curr_batch
            ), f"{last_assigned}, {(idx_since_save - 1) * loader.len_curr_batch}"
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
                    f"preds_batch_{args.num_factors}_{idx // save_every}.pt",
                ),
            )

            preds = torch.empty(
                min(
                    save_every * loader.batch_size,
                    loader.dataset_length - processed,
                ),
                device=device,
                dtype=dtype,
            )
            last_assigned = 0
    assert processed == loader.dataset_length
    return idx // save_every


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

    # @Todo: refix this
    checkpoint = torch.load(
        os.path.join(log_dir, f"trainer_state_{val_epoch}.ckpt"),
        map_location=device,
    )
    if "quantized" in log_dir:
        model = checkpoint
    else:
        model_state_dict = {
            k: v for k, v in checkpoint["model"].items() if k in model.state_dict()
        }
        model.load_state_dict(model_state_dict)
    for param in model.parameters():
        param.grad = None
    model = model.eval()

    if quantize:
        # qconfig = torch.ao.quantization.get_default_qconfig('x86')
        # # or, set the qconfig for QAT
        # qconfig = torch.ao.quantization.get_default_qat_qconfig('x86')
        # set the qengine to control weight packing
        if device == "cuda":
            torch.backends.quantized.engine = "x86"
        # set quantization config for server (x86)
        model.qconfig = torch.quantization.get_default_qconfig("x86")

        # insert observers
        torch.quantization.prepare(model, inplace=True)
        # Calibrate the model and collect statistics

        # convert to quantized version
        model = torch.ao.quantization.convert(model)

    if compile_model:
        model.compile()

    return model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Arguments for Model Training")
    parser.add_argument(
        "--sequence_length",
        type=int,
        default=1000,
        help="Length of the sequence",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Device to use (default: cuda)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help="Batch size (default: 10000)",
    )
    parser.add_argument(
        "--num_factors",
        type=int,
        default=2,
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
        default="../../tmp/run/made_sweep_learnings/",
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
        default=1000,
        help="Save every N batches.",
    )
    parser.add_argument(
        "--value_to_save",
        type=int,
        default=0,
        help="Save value x from the values.",
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        help="Quantize argument description",
    )
    parser.add_argument(
        "--compile_model",
        action="store_true",
        help="Compile model argument description",
    )
    parser.add_argument(
        "--resample_masks",
        action="store_true",
        help="Resample masks argument description",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        help="Data type argument description",
    )
    parser.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="Resample masks argument description",
    ),
    parser.add_argument(
        "--autocast",
        action="store_true",
        help="Resample masks argument description",
    ),
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../../outputs/mfi",
        help="Output directory path",
    )

    return parser.parse_args()


if __name__ == "__main__":

    with torch.no_grad():
        args = parse_arguments()
        print(args)

        if args.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True

        values_list = get_mfi_input_values(args.num_factors)

        value = torch.tensor(
            values_list[args.value_to_save],
            dtype=torch.bool,
            device=args.device,
        )

        print("Creating dataloaders..")
        loader = MfiDatasetIterator(
            value=value,
            num_features=args.sequence_length,
            batch_size=args.batch_size,
            device=args.device,
            num_factors=args.num_factors,
            dtype=args.dtype,
        )

        print("loading model..")
        if args.sequence_length == 1000:
            model = load_model_eval(
                args.sequence_length,
                args.hidden_dims,
                args.num_layers,
                args.n_masks,
                args.log_dir,
                args.val_epoch,
                args.device,
                args.quantize,
                args.compile_model,
                args.resample_masks,
            )
        else:
            print("DEBUGGING")
            # load the model
            model = models.MADE(
                input_dim=args.sequence_length,
                hidden_dims=[args.hidden_dims] * args.num_layers,
                n_masks=args.n_masks,
            )
            model = model.to(args.device)

        if args.dtype == "float16":
            model = model.half()

        print("Model type: ", model.parameters().__next__().dtype)

        print("Running predictions..")
        start_time = time.time()

        max_save_counter = pred(
            model=model,
            sequence_length=args.sequence_length,
            loader=loader,
            device=args.device,
            value_to_save=args.value_to_save,
            num_factors=args.num_factors,
            save_every=args.save_every,
            dtype=torch.float16 if args.dtype == "float16" else torch.float32,
            autocast=args.autocast,
        )

        end_time = time.time()
        print("Execution Time:", end_time - start_time, "seconds")
