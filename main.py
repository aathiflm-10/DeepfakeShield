import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os

from data_loader import create_dataloaders
from model import get_trainable_params, create_model, print_model_params
from train import train


def main():

    # =====================================
    # Fix random seed for reproducibility
    # =====================================
    torch.manual_seed(0)
    np.random.seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # =====================================
    # Create checkpoint folder
    # =====================================
    os.makedirs("checkpoints", exist_ok=True)

    # =====================================
    # Training Parameters
    # =====================================
    params = {
        "train_data": "base",      # base | augment | both
        "use_hidden_layer": True,
        "dropout": 0.3,
        "max_lr": 0.0001,
        "weight_decay": 0.0001,
        "use_lr_scheduler": True,
        "n_epochs": 20,
        "pct_start": 0.3,
        "batch_size": 32,
        "sample_ratio": 1.0
    }

    print("===================================")
    print("DeepFake Detection Training")
    print("===================================")
    print(f"Training dataset : {params['train_data']}")
    print(f"Epochs           : {params['n_epochs']}")
    print(f"Batch size       : {params['batch_size']}")
    print("===================================\n")

    # =====================================
    # Create Datasets
    # =====================================
    print("Creating datasets...\n")

    train_dl, val_base_dl, val_augment_dl, display_dl_iter = create_dataloaders(params)

    # =====================================
    # Loss Function
    # =====================================
    print("Creating loss function...\n")

    criterion = nn.CrossEntropyLoss()

    # =====================================
    # Create Model
    # =====================================
    print("Creating model...\n")

    model = create_model(
        bool(params['use_hidden_layer']),
        params['dropout']
    )

    print_model_params(model)

    # =====================================
    # Move Model to GPU if available
    # =====================================
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing device: {device}\n")

    model = model.to(device)

    # =====================================
    # Optimizer
    # =====================================
    params_to_update = get_trainable_params(model)

    print("Creating optimizer...\n")

    optimizer = optim.Adam(
        params_to_update,
        lr=params['max_lr'],
        weight_decay=params['weight_decay']
    )

    # =====================================
    # Learning Rate Scheduler
    # =====================================
    if params['use_lr_scheduler']:

        scheduler = lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=params['max_lr'],
            epochs=params['n_epochs'],
            steps_per_epoch=len(train_dl),
            pct_start=params['pct_start'],
            anneal_strategy='cos',
            cycle_momentum=False
        )

    else:
        scheduler = None

    # =====================================
    # Start Training
    # =====================================
    print("===================================")
    print("Training started...")
    print("===================================\n")

    train(
        train_dl,
        val_base_dl,
        val_augment_dl,
        display_dl_iter,
        model,
        optimizer,
        params['n_epochs'],
        params['max_lr'],
        scheduler,
        criterion,
        train_source=params["train_data"]
    )

    print("\n===================================")
    print("Training finished")
    print("Best model saved in /checkpoints")
    print("===================================")


# =====================================
# Required for Windows multiprocessing
# =====================================
if __name__ == "__main__":
    main()