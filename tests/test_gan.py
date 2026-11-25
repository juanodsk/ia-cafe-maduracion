"""CPU checks for conditioning, real-training-only guards, and saved GAN output."""
import csv
import json
from pathlib import Path
import tempfile
import unittest

from PIL import Image
import torch
from torch import nn

from cafe_ml.data import CLASS_NAMES, CoffeeFruitDataset
from cafe_ml.gan import (
    ConditionalDiscriminator, ConditionalGenerator, load_synthetic_records,
    train_gan, train_step, validate_training_records,
)


class ConditionalGanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_conditional_shapes_ranges_and_label_validation(self):
        generator = ConditionalGenerator(latent_dim=8, base_channels=4, embedding_dim=4).eval()
        discriminator = ConditionalDiscriminator(base_channels=4).eval()
        labels = torch.tensor([0, 1, 2, 3, 4])
        with torch.no_grad():
            images = generator(torch.randn(5, 8), labels)
            logits = discriminator(images, labels)
        self.assertEqual(images.shape, (5, 3, 64, 64))
        self.assertEqual(logits.shape, (5,))
        self.assertTrue(torch.isfinite(images).all())
        self.assertGreaterEqual(float(images.min()), -1)
        self.assertLessEqual(float(images.max()), 1)
        self.assertFalse(any(isinstance(layer, nn.Sigmoid) for layer in discriminator.modules()))
        with self.assertRaisesRegex(ValueError, "long tensor"):
            generator(torch.randn(5, 8), labels.float())
        with self.assertRaisesRegex(ValueError, "vocabulary"):
            generator(torch.randn(5, 8), torch.tensor([0, 1, 2, 3, 9]))

    def test_one_alternating_step_updates_both_networks(self):
        torch.manual_seed(42)
        generator = ConditionalGenerator(8, 4, 4)
        discriminator = ConditionalDiscriminator(4)
        before_g = generator.layers[0].weight.detach().clone()
        before_d = discriminator.layers[0].weight.detach().clone()
        optim_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optim_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        modes = []
        hook = discriminator.register_forward_pre_hook(
            lambda module, inputs: modes.append((module.training, next(module.parameters()).requires_grad))
        )
        metrics = train_step(generator, discriminator, torch.rand(4, 3, 64, 64) * 2 - 1,
                             torch.tensor([0, 1, 2, 3]), optim_g, optim_d)
        hook.remove()
        self.assertEqual(modes, [(True, True), (True, True), (True, False)])
        self.assertGreater(metrics["loss_d"], 0)
        self.assertGreater(metrics["loss_g"], 0)
        self.assertFalse(torch.equal(generator.layers[0].weight, before_g))
        self.assertFalse(torch.equal(discriminator.layers[0].weight, before_d))
        self.assertTrue(all(torch.isfinite(p).all() for p in generator.parameters()))
        self.assertTrue(all(torch.isfinite(p).all() for p in discriminator.parameters()))

    def test_guard_rejects_validation_test_and_generated_records(self):
        records = [dict(split="train", label=label, class_index=i) for i, label in enumerate(CLASS_NAMES)]
        validate_training_records(records)
        for forbidden in ("valid", "test"):
            changed = [dict(row, split=forbidden) for row in records]
            with self.assertRaisesRegex(ValueError, "real train"):
                validate_training_records(changed)
        with self.assertRaisesRegex(ValueError, "previously generated"):
            validate_training_records([dict(row, synthetic=True) for row in records])

    def test_cpu_pilot_saves_loadable_training_only_images_and_checkpoints(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            dataset, output = root / "dataset", root / "gan"
            (dataset / "csv").mkdir(parents=True)
            images, annotations = [], []
            for split, colour in (("train", "red"), ("valid", "green"), ("test", "blue")):
                (dataset / split).mkdir()
                Image.new("RGB", (64, 64), colour).save(dataset / split / "image.png")
                images.append(dict(split=split, image_id=1, relative_path=f"{split}/image.png",
                    group_id=f"group_{split}", sha256=f"sha_{split}", source_name=f"source_{split}",
                    width=64, height=64, annotation_count=5, capture_key=f"capture_{split}"))
                for index, label in enumerate(CLASS_NAMES):
                    annotations.append(dict(split=split, image_id=1, annotation_id=index + 1,
                        relative_image_path=f"{split}/image.png", group_id=f"group_{split}", label=label,
                        category_id=index + 1, bbox_x=0, bbox_y=0, bbox_width=64, bbox_height=64,
                        source_split=split, source_image_id=1, source_annotation_id=index + 1))
            for name, rows in (("imagenes", images), ("anotaciones", annotations)):
                with (dataset / "csv" / f"{name}.csv").open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            result = train_gan(dataset, output, device="cpu", epochs=1, steps_per_epoch=1,
                batch_size=4, workers=0, latent_dim=8, base_channels=4, synthetic_per_class=2)
            self.assertEqual(result["total_steps"], 1)
            self.assertEqual(result["synthetic_images"], 8)
            self.assertFalse(result["test_used"])
            generated = load_synthetic_records(output)
            self.assertEqual({row["label"] for row in generated}, set(CLASS_NAMES[:-1]))
            self.assertEqual({row["split"] for row in generated}, {"train"})
            self.assertEqual(len({row["group_id"] for row in generated}), 1)
            crop, target = CoffeeFruitDataset(generated, output)[0]
            self.assertEqual(crop.size, (64, 64))
            self.assertEqual(crop.mode, "RGB")
            self.assertEqual(target, 0)
            for file in ("history.csv", "losses.png", "generated_grid.png", "resultado.json"):
                self.assertTrue((output / file).exists())
            checkpoint = torch.load(result["training_checkpoint"], map_location="cpu", weights_only=True)
            self.assertEqual(checkpoint["completed_epochs"], 1)
            # Resuming a completed pilot preserves the existing sampled output.
            resumed = train_gan(dataset, output, device="cpu", epochs=1, steps_per_epoch=1,
                batch_size=4, workers=0, latent_dim=8, base_channels=4, synthetic_per_class=2)
            self.assertEqual(resumed["generator_run_id"], result["generator_run_id"])
            with self.assertRaisesRegex(ValueError, "configuration differs"):
                train_gan(dataset, output, device="cpu", epochs=2, steps_per_epoch=1,
                    batch_size=4, workers=0, latent_dim=8, base_channels=4, synthetic_per_class=2)


if __name__ == "__main__":
    unittest.main()
