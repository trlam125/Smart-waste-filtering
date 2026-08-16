from __future__ import annotations

from collections.abc import Sequence

import torch.nn as nn
from torchvision import models, transforms

SUPPORTED_ARCHITECTURES: tuple[str, ...] = (
    "efficientnet_b0",
    "mobilenet_v3_large",
    "resnet18",
)

IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)
TRAIN_AUGMENTATION_PROFILE = "camera_realworld_v2"


def create_model(architecture: str, num_classes: int, *, pretrained: bool) -> nn.Module:
    architecture = architecture.strip().lower()
    if architecture == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        return model

    if architecture == "mobilenet_v3_large":
        weights = models.MobileNet_V3_Large_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v3_large(weights=weights)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
        return model

    if architecture == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    raise ValueError(
        f"Unsupported architecture {architecture!r}. "
        f"Choose one of: {', '.join(SUPPORTED_ARCHITECTURES)}"
    )


def get_final_classifier_layer(model: nn.Module, architecture: str) -> nn.Module:
    """Return the final classification layer whose input is used as feedback embedding.

    Capturing the input to this layer gives a semantic feature vector from the trained
    network instead of reusing the 11-class softmax scores.
    """
    architecture = architecture.strip().lower()
    if architecture == "efficientnet_b0":
        layer = model.classifier[1]
    elif architecture == "mobilenet_v3_large":
        layer = model.classifier[3]
    elif architecture == "resnet18":
        layer = model.fc
    else:
        raise ValueError(
            f"Unsupported architecture {architecture!r}. "
            f"Choose one of: {', '.join(SUPPORTED_ARCHITECTURES)}"
        )
    if not isinstance(layer, nn.Linear):
        raise TypeError(f"Expected final classifier to be nn.Linear, got {type(layer).__name__}")
    return layer



def build_eval_transform(
    image_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    resize_size = max(image_size, int(round(image_size * 256 / 224)))
    return transforms.Compose(
        [
            transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
        ]
    )


def build_train_transform(
    image_size: int,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> transforms.Compose:
    """Camera-like augmentation for real-world single-object waste photos.

    The previous RandomResizedCrop pipeline tended to enlarge the foreground
    object. Deployment photos often contain a smaller, off-centre object with
    more surrounding context, so this profile keeps a normal resize/crop and
    then randomly translates and scales the whole view.
    """
    resize_size = max(image_size, int(round(image_size * 256 / 224)))
    fill = tuple(int(round(float(value) * 255.0)) for value in mean)
    return transforms.Compose(
        [
            transforms.Resize(
                resize_size,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomCrop(image_size),
            transforms.RandomAffine(
                degrees=12,
                translate=(0.12, 0.12),
                scale=(0.72, 1.08),
                shear=(-5.0, 5.0),
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=fill,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [
                    transforms.ColorJitter(
                        brightness=0.25,
                        contrast=0.25,
                        saturation=0.20,
                        hue=0.03,
                    )
                ],
                p=0.80,
            ),
            transforms.RandomPerspective(
                distortion_scale=0.15,
                p=0.15,
                interpolation=transforms.InterpolationMode.BILINEAR,
                fill=fill,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=tuple(mean), std=tuple(std)),
            transforms.RandomErasing(p=0.10, scale=(0.02, 0.08), ratio=(0.5, 2.0)),
        ]
    )
