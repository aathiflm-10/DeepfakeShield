import torch
from tqdm import tqdm
import numpy as np
import os

from utils import visualize_metrics
from sklearn.metrics import roc_auc_score as extra_metric


# =====================================================
# Records Class
# =====================================================
class Records:
    def __init__(self):
        self.train_losses, self.train_losses_wo_dropout = [], []
        self.base_val_losses, self.augment_val_losses = [], []

        self.train_accs, self.train_accs_wo_dropout = [], []
        self.base_val_accs, self.augment_val_accs = [], []

        self.train_custom_metrics, self.train_custom_metrics_wo_dropout = [], []
        self.base_val_custom_metrics, self.augment_val_custom_metrics = [], []

        self.lrs = []


# =====================================================
# Train One Epoch
# =====================================================
def train_one_epoch(epoch, model, train_dl, max_lr,
                    optimizer, criterion, scheduler, records):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()

    train_loss = 0
    train_loss_eval = 0

    total = 0
    correct_count = 0
    correct_count_eval = 0

    train_tk = tqdm(train_dl, total=len(train_dl), desc=f'Epoch {epoch}')

    for step, data in enumerate(train_tk):
        inputs = data['image'].to(device)
        labels = data['label'].view(-1).to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        if scheduler is not None:
            scheduler.step()
            records.lrs.append(scheduler.get_last_lr()[0])
        else:
            records.lrs.append(max_lr)

        _, predicted = torch.max(outputs, 1)

        total += labels.size(0)
        correct_count += (predicted == labels).sum().item()
        train_loss += loss.item()

        train_tk.set_postfix(
            loss=train_loss / (step + 1),
            acc=correct_count / total
        )

        # Evaluation without dropout
        model.eval()
        with torch.no_grad():
            outputs_eval = model(inputs)
            loss_eval = criterion(outputs_eval, labels)
            _, predicted_eval = torch.max(outputs_eval, 1)
            correct_count_eval += (predicted_eval == labels).sum().item()

        train_loss_eval += loss_eval.item()
        model.train()

    records.train_losses.append(train_loss / len(train_dl))
    records.train_accs.append(correct_count / total)

    records.train_losses_wo_dropout.append(train_loss_eval / len(train_dl))
    records.train_accs_wo_dropout.append(correct_count_eval / total)

    print(f"Epoch {epoch}: "
          f"Train Loss={records.train_losses[-1]:.4f}, "
          f"Train Acc={records.train_accs[-1]:.4f}")


# =====================================================
# Validation
# =====================================================
def validate(model, val_dl, criterion, records, data_name):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    val_loss = 0
    correct_count = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for data in val_dl:
            inputs = data['image'].to(device)
            labels = data['label'].view(-1).to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct_count += (predicted == labels).sum().item()
            val_loss += loss.item()

            all_labels.append(labels.cpu().numpy())
            all_predictions.append(predicted.cpu().numpy())

    all_labels = np.concatenate(all_labels)
    all_predictions = np.concatenate(all_predictions)

    extra_score = extra_metric(all_labels, all_predictions)

    if data_name == 'base':
        records.base_val_losses.append(val_loss / len(val_dl))
        records.base_val_accs.append(correct_count / total)
        records.base_val_custom_metrics.append(extra_score)
    else:
        records.augment_val_losses.append(val_loss / len(val_dl))
        records.augment_val_accs.append(correct_count / total)
        records.augment_val_custom_metrics.append(extra_score)

    print(f"{data_name.upper()} Val: "
          f"Acc={correct_count / total:.4f}, "
          f"{extra_metric.__name__}={extra_score:.4f}")


# =====================================================
# Main Train Loop
# =====================================================
def train(train_dl, val_base_dl, val_augment_dl,
          display_dl_iter, model, optimizer,
          n_epochs, max_lr, scheduler,
          criterion, train_source):

    records = Records()
    best_metric = 0.0

    os.makedirs("checkpoints", exist_ok=True)

    for epoch in range(n_epochs):

        train_one_epoch(epoch, model, train_dl,
                        max_lr, optimizer,
                        criterion, scheduler, records)

        validate(model, val_base_dl, criterion, records, 'base')
        validate(model, val_augment_dl, criterion, records, 'augment')

        # Metric selection logic
        if train_source == "both":
            selection_metric = np.mean([
                records.base_val_accs[-1],
                records.augment_val_accs[-1]
            ])
        elif train_source == "augment":
            selection_metric = records.augment_val_accs[-1]
        else:
            selection_metric = records.base_val_accs[-1]

        if selection_metric > best_metric:
            best_metric = selection_metric
            print(f">>> Saving best model (Metric={best_metric:.4f})")

            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict()
            }, "checkpoints/best_model.pth")

        visualize_metrics(records, extra_metric, "metrics.png")

    print("\nTraining completed successfully.")
