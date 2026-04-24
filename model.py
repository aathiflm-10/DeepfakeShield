import torchvision.models as models
import torch.nn as nn


# =====================================================
# Print model structure helper
# =====================================================
def check_model_block(model):
    for name, child in model.named_children():
        print(name)


# =====================================================
# Count model parameters
# =====================================================
def print_model_params(model):
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    print(f"total number of params: {pytorch_total_params:,}")
    return pytorch_total_params


# =====================================================
# Trainable parameters
# =====================================================
def get_trainable_params(model):

    print("Params to learn:")

    params_to_update = []

    for name, param in model.named_parameters():

        if param.requires_grad:
            print("\t", repr(name))
            params_to_update.append(param)

    return params_to_update


# =====================================================
# Model creation
# =====================================================
def create_model(use_hidden_layer=True, dropout=0.3, pretrained=True):

    # ----------------------------------------
    # Load ResNet18 backbone
    # ----------------------------------------
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    else:
        model = models.resnet18(weights=None)

    in_features = model.fc.in_features

    print(f"Input feature dim: {in_features}")

    # ----------------------------------------
    # Custom classifier
    # ----------------------------------------
    if use_hidden_layer:

        model.fc = nn.Sequential(
            nn.Dropout(dropout),

            nn.Linear(in_features, 256),
            nn.ReLU(),

            nn.BatchNorm1d(256),

            nn.Dropout(dropout),

            nn.Linear(256, 2)
        )

    else:

        model.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 2)
        )

    print(model)

    return model