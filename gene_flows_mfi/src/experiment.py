from torch import optim
from torch.nn import functional as F
import os

from src.model import loss_fn
from src.pytorch_generative.pytorch_generative import models, trainer


def run_experiment(
    train_loader,
    val_loader,
    n_epochs,
    log_dir,
    n_gpus,
    device_id,
    sequence_length,
    hidden_dims,
    num_layers,
    n_masks,
    loss_fn,
    batch_size,
    learning_rate,
    step_size,
    weight_decay=0,
    use_tensorboard=False,
    use_wandb=True,
    restore=True,
    save_checkpoint_epochs=1,
):
    """Training script with defaults to reproduce results.

    The code inside this function is self contained and can be used as a top level
    training script, e.g. by copy/pasting it into a Jupyter notebook.

    Args:
        n_epochs: Number of epochs to train for.
        batch_size: Batch size to use for training and evaluation.
        log_dir: Directory where to log trainer state and TensorBoard summaries.
        n_gpus: Number of GPUs to use for training the model. If 0, uses CPU.
        device_id: The device_id of the current GPU when training on multiple GPUs.
        debug_loader: Debug DataLoader which replaces the default training and
            evaluation loaders if not 'None'. Do not use unless you're writing unit
            tests.
    """

    # @Todo: adjust to allow multiple layers
    model = models.MADE(
        input_dim=sequence_length,
        hidden_dims=[hidden_dims] * num_layers,
        n_masks=n_masks,
        device=device_id,
    )
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    # @Todo: debug scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=step_size * batch_size,
    )

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, "wandb"), exist_ok=True)
    print(log_dir)

    model_trainer = trainer.Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        train_loader=train_loader,
        eval_loader=val_loader,
        log_dir=log_dir,
        n_gpus=n_gpus,
        device_id=device_id,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        save_checkpoint_epochs=save_checkpoint_epochs,
    )
    model_trainer.interleaved_train_and_eval(n_epochs, restore=restore)

    return model_trainer


def eval_run(data_loader, model, device):
    # Evaluate.
    n_examples = 0
    losses = 0
    model = model.to(device)
    model.eval()
    for batch in data_loader:
        batch = batch if isinstance(batch, (tuple, list)) else (batch, None)
        x, y = batch
        n_batch_examples = x.shape[0]
        n_examples += n_batch_examples
        x = x.to(device)
        preds = model(x.float())
        loss = loss_fn(x.float(), y, preds).detach()
        losses += loss
    return losses / n_examples
