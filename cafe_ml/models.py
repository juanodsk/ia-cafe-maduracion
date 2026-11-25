"""The three CNNs and two Transformers used in the coffee maturity comparison.

The explicit weight versions match TorchVision's current DEFAULT aliases, while
remaining stable if those aliases change. Evaluation always uses the selected
weights' official resize, central crop, interpolation and normalization preset.
Training adds only a mild geometric perturbation, preserving maturity colours.

Official model and preprocessing references:
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.resnet50.html
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.efficientnet_b0.html
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.densenet121.html
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.vit_b_16.html
https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.swin_t.html

ViT-B/16 has about 86.6 million parameters and 330 MB of pretrained weights;
load models sequentially rather than keeping all five on a small GPU together.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torchvision import models as tv_models
from torchvision import transforms


MODEL_NAMES = (
    "resnet50",
    "efficientnet_b0",
    "densenet121",
    "vit_b_16",
    "swin_t",
)
CNN_NAMES = MODEL_NAMES[:3]
TRANSFORMER_NAMES = MODEL_NAMES[3:]

# Pin enum members rather than the mutable DEFAULT aliases.
_WEIGHT_ENUM_NAMES = {
    "resnet50": ("ResNet50_Weights", "IMAGENET1K_V2"),
    "efficientnet_b0": ("EfficientNet_B0_Weights", "IMAGENET1K_V1"),
    "densenet121": ("DenseNet121_Weights", "IMAGENET1K_V1"),
    "vit_b_16": ("ViT_B_16_Weights", "IMAGENET1K_V1"),
    "swin_t": ("Swin_T_Weights", "IMAGENET1K_V1"),
}
_HEAD_ATTRIBUTES = {
    "resnet50": "fc",
    "efficientnet_b0": "classifier",
    "densenet121": "classifier",
    "vit_b_16": "heads",
    "swin_t": "head",
}
_FEATURE_DIMS = {
    "resnet50": 2048,
    "efficientnet_b0": 1280,
    "densenet121": 1024,
    "vit_b_16": 768,
    "swin_t": 768,
}


def _validate_name(name: str) -> str:
    if name not in MODEL_NAMES:
        raise ValueError(f"Unknown model {name!r}; choose one of {MODEL_NAMES}.")
    return name


def get_weights(name: str) -> Any:
    """Return the pinned weight enum without downloading its checkpoint."""
    enum_name, member = _WEIGHT_ENUM_NAMES[_validate_name(name)]
    return getattr(getattr(tv_models, enum_name), member)


def get_feature_dim(name: str) -> int:
    """Number of pooled backbone features before the classification head."""
    return _FEATURE_DIMS[_validate_name(name)]


def get_model_spec(name: str) -> dict[str, Any]:
    """Return JSON-serializable provenance and preprocessing metadata."""
    weights = get_weights(name)
    preset = weights.transforms()
    enum_name, member = _WEIGHT_ENUM_NAMES[name]
    return {
        "name": name,
        "family": "cnn" if name in CNN_NAMES else "transformer",
        "weights": f"{enum_name}.{member}",
        "weights_url": weights.url,
        "imagenet_parameters_before_head_replacement": weights.meta["num_params"],
        "checkpoint_size_mb": weights.meta.get("_file_size"),
        "feature_dim": get_feature_dim(name),
        "input_size": list(preset.crop_size),
        "resize_size": list(preset.resize_size),
        "interpolation": preset.interpolation.value,
        "normalization_mean": list(preset.mean),
        "normalization_std": list(preset.std),
        "reference": (
            "https://docs.pytorch.org/vision/stable/models/generated/"
            f"torchvision.models.{name}.html"
        ),
    }


def get_transforms(name: str, train: bool = False) -> Any:
    """Get a PIL-RGB-to-tensor transform with the official weight preset.

    Validation/test are deterministic. Colour jitter, hue shifts, mixup and
    cutmix are intentionally absent because fruit colour defines maturity.
    The training script can also select ``train=False`` for cached frozen
    features so that a saved feature row always represents the same crop.
    """
    preset = get_weights(name).transforms()
    if not train:
        return preset
    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(
                degrees=10,
                interpolation=preset.interpolation,
                fill=0,
            ),
            preset,
        ]
    )


def _model_name(model: nn.Module, name: str | None = None) -> str:
    resolved = name if name is not None else getattr(model, "_cafe_model_name", None)
    return _validate_name(resolved)


def get_head(model: nn.Module, name: str | None = None) -> nn.Module:
    """Return the classifier module replaced for the five maturity classes."""
    return getattr(model, _HEAD_ATTRIBUTES[_model_name(model, name)])


def set_head(model: nn.Module, name: str, head: nn.Module) -> nn.Module:
    """Replace a classifier, e.g. with ``nn.Identity()`` to cache embeddings.

    Replacing the *whole* EfficientNet classifier also removes its dropout, so
    deterministic frozen features match evaluation rather than a random mask.
    ``get_feature_dim(name)`` specifies the input size of a replacement head.
    """
    _validate_name(name)
    if not isinstance(head, nn.Module):
        raise TypeError("head must be a torch.nn.Module.")
    setattr(model, _HEAD_ATTRIBUTES[name], head)
    head.requires_grad_(True)
    model._cafe_model_name = name
    return model


def head_parameters(
    model: nn.Module, name: str | None = None
) -> Iterator[nn.Parameter]:
    return get_head(model, name).parameters()


def set_backbone_trainable(
    model: nn.Module,
    name: str | None = None,
    trainable: bool = False,
) -> nn.Module:
    """Freeze/unfreeze the backbone while always leaving the head trainable.

    Rebuild the optimizer after changing this flag to include newly trainable
    parameters. Call :func:`configure_training_mode` at each train/eval phase.
    """
    resolved = _model_name(model, name)
    for parameter in model.parameters():
        parameter.requires_grad_(trainable)
    for parameter in get_head(model, resolved).parameters():
        parameter.requires_grad_(True)
    model._cafe_model_name = resolved
    model._cafe_backbone_trainable = bool(trainable)
    return model


def configure_training_mode(
    model: nn.Module,
    name: str | None = None,
    training: bool = True,
) -> nn.Module:
    """Set train/eval modes without updating a frozen backbone's statistics.

    ``requires_grad=False`` alone does not freeze BatchNorm running statistics
    or disable stochastic depth/dropout. A frozen backbone therefore stays in
    eval mode, with only the classifier switched back to training mode.
    """
    resolved = _model_name(model, name)
    if not training:
        return model.eval()
    if getattr(model, "_cafe_backbone_trainable", False):
        return model.train()
    model.eval()
    get_head(model, resolved).train()
    return model


def build_model(
    name: str,
    num_classes: int = 5,
    pretrained: bool = True,
    trainable_backbone: bool = False,
    weights_dir: str | Path | None = None,
) -> nn.Module:
    """Construct a selected architecture and replace its ImageNet classifier.

    Only this function can trigger a checkpoint download. ``pretrained=False``
    is useful for offline tests and restoring a complete project checkpoint;
    it must not silently replace pretrained initialization in an experiment.
    ``weights_dir`` is a Torch Hub cache directory, containing ``checkpoints/``.
    A caller can instead set ``TORCH_HOME`` before using this module.
    """
    _validate_name(name)
    if not isinstance(num_classes, int) or isinstance(num_classes, bool) or num_classes < 2:
        raise ValueError("num_classes must be an integer of at least two.")
    if weights_dir is not None:
        cache = Path(weights_dir).expanduser().resolve()
        cache.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache))
    weights = get_weights(name) if pretrained else None
    model = getattr(tv_models, name)(weights=weights)
    if name == "resnet50":
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name == "efficientnet_b0":
        old_linear = model.classifier[-1]
        model.classifier[-1] = nn.Linear(old_linear.in_features, num_classes)
    elif name == "densenet121":
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif name == "vit_b_16":
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    elif name == "swin_t":
        model.head = nn.Linear(model.head.in_features, num_classes)
    model._cafe_model_name = name
    model._cafe_weight_name = (
        f"{_WEIGHT_ENUM_NAMES[name][0]}.{_WEIGHT_ENUM_NAMES[name][1]}"
        if pretrained
        else None
    )
    set_backbone_trainable(model, trainable=trainable_backbone)
    configure_training_mode(model, training=False)
    return model
