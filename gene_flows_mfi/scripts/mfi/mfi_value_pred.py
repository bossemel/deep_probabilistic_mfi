import os
import sys


module_path = os.path.abspath(os.path.join("../.."))
if module_path not in sys.path:
    sys.path.append(module_path)

import argparse
import time
import torch

from src.pytorch_generative.pytorch_generative import models

from src.mfi import MfiDatasetIterator, pred_mfi_components
from src.model import load_model_eval
from src.utils import get_mfi_input_values


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
        default="cuda",
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

        if args.dtype == "float16":
            dtype = torch.float16
        elif args.dtype == "float32":
            dtype = torch.float32
        elif args.dtype == "float64":
            dtype = torch.float64
        else:
            raise ValueError()

        print("Running predictions..")
        start_time = time.time()

        max_save_counter = pred_mfi_components(
            model=model,
            sequence_length=args.sequence_length,
            loader=loader,
            device=args.device,
            value_to_save=args.value_to_save,
            num_factors=args.num_factors,
            save_every=args.save_every,
            dtype=dtype,
            autocast=args.autocast,
            output_dir=args.output_dir,
        )

        end_time = time.time()
        print("Execution Time:", end_time - start_time, "seconds")
