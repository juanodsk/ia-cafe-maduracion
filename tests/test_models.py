"""Offline regression checks for model selection and frozen-backbone training."""

import gc
import unittest

import torch
from PIL import Image
from torch import nn

from cafe_ml.models import (
    MODEL_NAMES,
    build_model,
    configure_training_mode,
    get_feature_dim,
    get_head,
    get_model_spec,
    get_transforms,
    set_backbone_trainable,
    set_head,
)


class ToyResNet(nn.Module):
    """Expose the same head boundary with an inexpensive BatchNorm backbone."""

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5))
        self.fc = nn.Linear(4, 5)
        self._cafe_model_name = "resnet50"

    def forward(self, inputs):
        return self.fc(self.backbone(inputs))


class ModelTests(unittest.TestCase):
    def test_exact_architectures_and_five_output_heads_offline(self):
        expected_inputs = {
            "resnet50": 2048,
            "efficientnet_b0": 1280,
            "densenet121": 1024,
            "vit_b_16": 768,
            "swin_t": 768,
        }
        self.assertEqual(tuple(expected_inputs), MODEL_NAMES)
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                model = build_model(name, pretrained=False)
                linear = [module for module in get_head(model).modules() if isinstance(module, nn.Linear)][-1]
                self.assertEqual(linear.in_features, expected_inputs[name])
                self.assertEqual(get_feature_dim(name), expected_inputs[name])
                self.assertEqual(linear.out_features, 5)
                head_ids = {id(parameter) for parameter in get_head(model).parameters()}
                self.assertTrue(all(parameter.requires_grad == (id(parameter) in head_ids) for parameter in model.parameters()))
                del model
                gc.collect()

    def test_identity_head_exposes_backbone_features(self):
        model = ToyResNet().eval()
        set_backbone_trainable(model, trainable=False)
        set_head(model, "resnet50", nn.Identity())
        inputs = torch.rand(3, 4)
        with torch.inference_mode():
            self.assertTrue(torch.equal(model(inputs), model.backbone(inputs)))
        self.assertIsInstance(get_head(model), nn.Identity)

    def test_frozen_training_changes_head_but_not_backbone_or_batchnorm(self):
        torch.manual_seed(42)
        model = ToyResNet()
        set_backbone_trainable(model, trainable=False)
        before = {key: value.clone() for key, value in model.state_dict().items()}
        configure_training_mode(model, training=True)
        self.assertFalse(model.backbone.training)
        self.assertFalse(model.backbone[1].training)
        self.assertTrue(model.fc.training)
        optimizer = torch.optim.SGD(model.fc.parameters(), lr=0.2)
        logits = model(torch.rand(8, 4))
        loss = nn.CrossEntropyLoss()(logits, torch.arange(8) % 5)
        loss.backward()
        optimizer.step()
        for key, value in model.state_dict().items():
            if key.startswith("backbone."):
                self.assertTrue(torch.equal(value, before[key]), key)
        self.assertFalse(torch.equal(model.fc.weight, before["fc.weight"]))
        set_backbone_trainable(model, trainable=True)
        configure_training_mode(model, training=True)
        self.assertTrue(model.backbone[1].training)
        self.assertTrue(all(parameter.requires_grad for parameter in model.parameters()))
        configure_training_mode(model, training=False)
        self.assertFalse(any(module.training for module in model.modules()))

    def test_official_eval_transforms_are_deterministic_and_224(self):
        image = Image.new("RGB", (83, 65), (20, 145, 30))
        for name in MODEL_NAMES:
            with self.subTest(model=name):
                transform = get_transforms(name)
                first = transform(image)
                second = transform(image)
                self.assertEqual(tuple(first.shape), (3, 224, 224))
                self.assertTrue(torch.equal(first, second))
                self.assertEqual(get_model_spec(name)["input_size"], [224])

    def test_invalid_model_and_class_count_are_rejected_before_download(self):
        with self.assertRaises(ValueError):
            build_model("resnet18", pretrained=True)
        with self.assertRaises(ValueError):
            build_model("resnet50", num_classes=1, pretrained=True)


if __name__ == "__main__":
    unittest.main()
