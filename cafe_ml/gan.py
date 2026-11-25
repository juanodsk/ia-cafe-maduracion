"""Conditional 64x64 DCGAN pilot fitted exclusively on real training fruit crops.

The convolutional architecture and initialization follow the official PyTorch
DCGAN tutorial, extended with class embeddings. The discriminator emits logits
for BCEWithLogitsLoss. Synthetic labels are requested conditions, not expert
verified maturity labels. This small pilot does not establish image quality or
improved classification performance; evaluate the separate augmentation ablation.

Primary references:
https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
https://docs.pytorch.org/docs/2.9/generated/torch.nn.BCEWithLogitsLoss.html
https://docs.pytorch.org/docs/2.9/data.html#torch.utils.data.WeightedRandomSampler
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache" / "matplotlib"))
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from .data import CLASS_NAMES, CLASS_TO_INDEX, CoffeeFruitDataset, load_records

MINORITY_CLASSES = CLASS_NAMES[:-1]


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _worker_init(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)


def _labels(labels: torch.Tensor, size: int) -> None:
    if labels.ndim != 1 or labels.shape[0] != size or labels.dtype != torch.long:
        raise ValueError("Conditional class labels must be a long tensor of shape (batch,)")
    if labels.numel() and (int(labels.min()) < 0 or int(labels.max()) >= len(CLASS_NAMES)):
        raise ValueError("Conditional class label outside the fixed maturity vocabulary")


def validate_training_records(records: Sequence[dict]) -> None:
    if not records or any(r.get("split") != "train" for r in records):
        raise ValueError("GAN training accepts only the fixed real train partition")
    if any(r.get("synthetic") in (True, "True", "true", "1") for r in records):
        raise ValueError("GAN training requires real crops, not previously generated images")
    counts = Counter(r.get("label") for r in records)
    if set(counts) != set(CLASS_NAMES):
        raise ValueError("All five real maturity classes are required for conditional GAN training")
    if any(int(r["class_index"]) != CLASS_TO_INDEX[r["label"]] for r in records):
        raise ValueError("GAN records contain inconsistent class indices")


def initialize_weights(module: nn.Module) -> None:
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, mean=1.0, std=0.02)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, mean=0.0, std=0.02)


class ConditionalGenerator(nn.Module):
    def __init__(self, latent_dim: int = 64, base_channels: int = 32, embedding_dim: int = 16):
        super().__init__()
        if min(latent_dim, base_channels, embedding_dim) < 1:
            raise ValueError("Generator dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.base_channels = int(base_channels)
        self.embedding_dim = int(embedding_dim)
        self.embedding = nn.Embedding(len(CLASS_NAMES), embedding_dim)
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(latent_dim + embedding_dim, base_channels * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(base_channels * 8), nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4), nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2), nn.ReLU(True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels), nn.ReLU(True),
            nn.ConvTranspose2d(base_channels, 3, 4, 2, 1, bias=False), nn.Tanh(),
        )
        self.apply(initialize_weights)

    def forward(self, noise: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if noise.ndim != 2 or noise.shape[1] != self.latent_dim:
            raise ValueError(f"Noise must have shape (batch, {self.latent_dim})")
        _labels(labels, len(noise))
        condition = self.embedding(labels)
        return self.layers(torch.cat((noise, condition), dim=1)[:, :, None, None])


class ConditionalDiscriminator(nn.Module):
    def __init__(self, base_channels: int = 32):
        super().__init__()
        if base_channels < 1:
            raise ValueError("Discriminator width must be positive")
        self.embedding = nn.Embedding(len(CLASS_NAMES), 64 * 64)
        self.layers = nn.Sequential(
            nn.Conv2d(4, base_channels, 4, 2, 1, bias=False), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 2), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 4), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(base_channels * 8), nn.LeakyReLU(0.2, True),
            nn.Conv2d(base_channels * 8, 1, 4, 1, 0, bias=False),
        )
        self.apply(initialize_weights)

    def forward(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[1:]) != (3, 64, 64):
            raise ValueError("Discriminator requires RGB tensors of shape (batch, 3, 64, 64)")
        _labels(labels, len(images))
        condition = self.embedding(labels).view(-1, 1, 64, 64)
        return self.layers(torch.cat((images, condition), dim=1)).flatten()


def train_step(generator: ConditionalGenerator, discriminator: ConditionalDiscriminator,
               real: torch.Tensor, labels: torch.Tensor,
               optimizer_g: torch.optim.Optimizer, optimizer_d: torch.optim.Optimizer) -> dict[str, float]:
    """One alternating update with matching real/generated maturity conditions."""
    if len(real) < 2 or not torch.isfinite(real).all() or real.min() < -1.0001 or real.max() > 1.0001:
        raise ValueError("GAN real batches require at least two finite images normalized to [-1,1]")
    _labels(labels, len(real))
    generator.train()
    discriminator.train()
    discriminator.requires_grad_(True)
    loss_fn = nn.BCEWithLogitsLoss()
    noise = torch.randn(len(real), generator.latent_dim, device=real.device)
    fake = generator(noise, labels)
    optimizer_d.zero_grad(set_to_none=True)
    real_logits = discriminator(real, labels)
    fake_logits = discriminator(fake.detach(), labels)
    loss_d = loss_fn(real_logits, torch.ones_like(real_logits)) + loss_fn(fake_logits, torch.zeros_like(fake_logits))
    if not torch.isfinite(loss_d):
        raise RuntimeError("Nonfinite discriminator loss")
    loss_d.backward()
    optimizer_d.step()
    optimizer_g.zero_grad(set_to_none=True)
    # Keep D parameters fixed but retain the same BatchNorm batch-statistic
    # function used to discriminate fake images above (standard DCGAN).
    # Switching D to eval here lets G exploit a different discriminator.
    discriminator.requires_grad_(False)
    discriminator.train()
    generated_logits = discriminator(fake, labels)
    loss_g = loss_fn(generated_logits, torch.ones_like(generated_logits))
    if not torch.isfinite(loss_g):
        raise RuntimeError("Nonfinite generator loss")
    loss_g.backward()
    optimizer_g.step()
    discriminator.requires_grad_(True)
    discriminator.train()
    return {"loss_d": float(loss_d.detach()), "loss_g": float(loss_g.detach()),
            "real_score": float(real_logits.detach().sigmoid().mean()),
            "fake_score": float(fake_logits.detach().sigmoid().mean())}


def get_gan_transform():
    # Keep colour/hue unchanged: maturity depends on the fruit's colour.
    return transforms.Compose([
        transforms.Resize((64, 64), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        transforms.RandomHorizontalFlip(p=0.5), transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def _save_grid(generator: ConditionalGenerator, path: Path, device: torch.device, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = torch.Generator(device=device).manual_seed(seed)
    labels = torch.arange(len(CLASS_NAMES), device=device).repeat_interleave(8)
    generator.eval()
    with torch.inference_mode():
        noise = torch.randn(len(labels), generator.latent_dim, device=device, generator=rng)
        images = generator(noise, labels).cpu().add(1).div(2).clamp(0, 1)
    grid = make_grid(images, nrow=8, padding=2).permute(1, 2, 0).numpy()
    figure, axis = plt.subplots(figsize=(10, 7))
    axis.imshow(grid)
    axis.set_title("Conditional DCGAN pilot: requested class per row")
    axis.set_yticks([34 + index * 66 for index in range(len(CLASS_NAMES))], labels=CLASS_NAMES)
    axis.set_xticks([])
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _save_losses(history: list[dict], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(9, 4))
    axis.plot([r["global_step"] for r in history], [r["loss_d"] for r in history], label="Discriminator")
    axis.plot([r["global_step"] for r in history], [r["loss_g"] for r in history], label="Generator")
    axis.set(xlabel="Training step", ylabel="BCE loss", title="Conditional DCGAN training losses")
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def generate_synthetic(generator: ConditionalGenerator, output_dir: Path, device: torch.device,
                       per_class: int, seed: int, run_id: str) -> list[dict[str, Any]]:
    if per_class < 1:
        raise ValueError("Synthetic images per minority class must be positive")
    generator.eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    records = []
    for label in MINORITY_CLASSES:
        directory = output_dir / "synthetic" / label
        directory.mkdir(parents=True, exist_ok=True)
        for start in range(0, per_class, 64):
            count = min(64, per_class - start)
            labels = torch.full((count,), CLASS_TO_INDEX[label], dtype=torch.long, device=device)
            with torch.inference_mode():
                noise = torch.randn(count, generator.latent_dim, generator=rng, device=device)
                images = generator(noise, labels).cpu().add(1).div(2).clamp(0, 1)
            for offset, image in enumerate(images):
                index = start + offset
                file = directory / f"{label}_{index:05d}.png"
                save_image(image, file)
                relative = file.relative_to(output_dir).as_posix()
                identifier = len(records) + 1
                records.append({"split": "train", "image_id": identifier, "annotation_id": identifier,
                    "crop_id": f"gan:{run_id}:{label}:{index}", "relative_image_path": relative,
                    "group_id": f"gan_{run_id}", "label": label, "class_index": CLASS_TO_INDEX[label],
                    "category_id": CLASS_TO_INDEX[label] + 1,
                    "bbox_x": 0, "bbox_y": 0, "bbox_width": 64, "bbox_height": 64,
                    "image_width": 64, "image_height": 64, "crop_left": 0, "crop_top": 0, "crop_right": 64, "crop_bottom": 64,
                    "parent_sha256": _hash_file(file), "synthetic": True,
                    "generator_fitted_on_split": "train", "allowed_use": "train_only",
                    "source_split": "gan_train", "source_image_id": identifier, "source_annotation_id": identifier,
                    "generation_seed": seed, "generator_run_id": run_id})
    with (output_dir / "synthetic_train.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    return records


def load_synthetic_records(gan_dir: str | Path) -> list[dict[str, Any]]:
    """Load generated training-only crops for an external fixed-holdout ablation.

    Use ``CoffeeFruitDataset(records, gan_dir, transform)``. A GAN fitted on all
    train images must not augment a held-out CV fold; that would leak its images.
    """
    gan_dir = Path(gan_dir).resolve()
    with (gan_dir / "synthetic_train.csv").open("r", encoding="utf-8", newline="") as handle:
        records = list(csv.DictReader(handle))
    if not records:
        raise ValueError("No generated training images")
    for record in records:
        if record["split"] != "train" or record["allowed_use"] != "train_only" or record["generator_fitted_on_split"] != "train":
            raise ValueError("Synthetic record is not restricted to training")
        if record["label"] not in MINORITY_CLASSES:
            raise ValueError("Unexpected synthetic class")
        for name in ("image_id", "annotation_id", "class_index", "category_id", "image_width", "image_height", "crop_left", "crop_top", "crop_right", "crop_bottom", "source_image_id", "source_annotation_id", "generation_seed"):
            record[name] = int(record[name])
        for name in ("bbox_x", "bbox_y", "bbox_width", "bbox_height"):
            record[name] = float(record[name])
        record["synthetic"] = True
        if record["class_index"] != CLASS_TO_INDEX[record["label"]]:
            raise ValueError("Synthetic label and class index disagree")
        file = (gan_dir / record["relative_image_path"]).resolve()
        if not file.is_relative_to(gan_dir) or _hash_file(file) != record["parent_sha256"]:
            raise ValueError("Synthetic image path/hash invalid")
    return records


def train_gan(dataset_root: str | Path, output_dir: str | Path, device: str = "cuda",
              epochs: int = 5, steps_per_epoch: int = 80, batch_size: int = 64, seed: int = 42,
              synthetic_per_class: int = 256, workers: int = 2,
              latent_dim: int = 64, base_channels: int = 32) -> dict[str, Any]:
    """Train a bounded conditional GAN pilot and export minority training crops.

    One epoch here is a stated number of balanced, sampled batches, rather than
    a complete pass through the imbalanced real dataset. No validation or test
    fruit records are loaded, and GAN losses are not classification metrics.
    """
    if min(epochs, steps_per_epoch, synthetic_per_class, latent_dim, base_channels) < 1 or batch_size < 2 or workers < 0:
        raise ValueError("Positive GAN budgets and batch_size >= 2 are required")
    torch_device = torch.device(device)
    if torch_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; explicitly choose device='cpu' for a CPU pilot")
    dataset_root, output_dir = Path(dataset_root).resolve(), Path(output_dir).resolve()
    records = load_records(dataset_root, "train")
    validate_training_records(records)
    counts = Counter(r["label"] for r in records)
    identity = json.dumps([[r["crop_id"], r["group_id"], r["parent_sha256"], r["label"],
        r["crop_left"], r["crop_top"], r["crop_right"], r["crop_bottom"]] for r in records], separators=(",", ":"))
    config = {"architecture": "conditional_DCGAN_64", "class_labels": list(CLASS_NAMES),
        "epochs": epochs, "steps_per_epoch": steps_per_epoch, "batch_size": batch_size,
        "seed": seed, "synthetic_per_minority_class": synthetic_per_class,
        "latent_dim": latent_dim, "base_channels": base_channels, "embedding_dim": 16,
        "learning_rate": 0.0002, "adam_betas": [0.5, 0.999], "real_range": [-1, 1],
        "training_records_sha256": hashlib.sha256(identity.encode()).hexdigest(),
        "train_crops": len(records), "train_class_counts": dict(counts),
        "sampled_images_per_epoch": batch_size * steps_per_epoch,
        "train_sampler": "WeightedRandomSampler_inverse_class_frequency_with_replacement",
        "workers": workers, "device": str(torch_device), "torch": str(torch.__version__),
        "new_augmentation": "horizontal_flip_only; no colour changes", "test_used": False,
        "discriminator_generator_update_mode": "train_batchnorm_frozen_parameters",
        "pilot": True}
    config_path = output_dir / "configuracion.json"
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != config:
        raise ValueError("GAN configuration differs from saved run; use another output directory")
    result_path = output_dir / "resultado.json"
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        load_synthetic_records(output_dir)
        return result
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = output_dir / "checkpoints"
    checkpoints.mkdir(exist_ok=True)
    _json(config_path, config)
    run_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    _seed(seed)
    generator = ConditionalGenerator(latent_dim, base_channels).to(torch_device)
    discriminator = ConditionalDiscriminator(base_channels).to(torch_device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    history: list[dict] = []
    start_epoch = 0
    last_checkpoint = checkpoints / "training_last.pt"
    if last_checkpoint.exists():
        saved = torch.load(last_checkpoint, map_location=torch_device, weights_only=True)
        if saved["configuration"] != config:
            raise ValueError("GAN checkpoint configuration mismatch")
        generator.load_state_dict(saved["generator"])
        discriminator.load_state_dict(saved["discriminator"])
        optimizer_g.load_state_dict(saved["optimizer_g"])
        optimizer_d.load_state_dict(saved["optimizer_d"])
        history = saved["history"]
        start_epoch = saved["completed_epochs"]
    dataset = CoffeeFruitDataset(records, dataset_root, get_gan_transform(), cache_size=8)
    weights = torch.tensor([1.0 / counts[r["label"]] for r in records], dtype=torch.double)
    started = time.monotonic()
    for epoch in range(start_epoch, epochs):
        _seed(seed + epoch)
        sampler = WeightedRandomSampler(weights, batch_size * steps_per_epoch, replacement=True,
            generator=torch.Generator().manual_seed(seed + 1000 + epoch))
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=workers,
            drop_last=True, pin_memory=torch_device.type == "cuda", worker_init_fn=_worker_init,
            generator=torch.Generator().manual_seed(seed + 2000 + epoch), persistent_workers=False)
        sampled_counts = Counter()
        for step, (real, labels) in enumerate(loader, start=1):
            sampled_counts.update(labels.tolist())
            values = train_step(generator, discriminator, real.to(torch_device), labels.to(torch_device), optimizer_g, optimizer_d)
            history.append({"epoch": epoch + 1, "step": step, "global_step": epoch * steps_per_epoch + step, **values})
            if step == 1 or step % 20 == 0 or step == steps_per_epoch:
                print(f"[GAN] epoch {epoch + 1}/{epochs}, step {step}/{steps_per_epoch}: D={values['loss_d']:.4f}, G={values['loss_g']:.4f}", flush=True)
        del loader
        checkpoint = {"configuration": config, "completed_epochs": epoch + 1,
            "generator": generator.state_dict(), "discriminator": discriminator.state_dict(),
            "optimizer_g": optimizer_g.state_dict(), "optimizer_d": optimizer_d.state_dict(), "history": history}
        temporary = checkpoints / "training_last.tmp"
        torch.save(checkpoint, temporary)
        temporary.replace(last_checkpoint)
        torch.save({"configuration": config, "state_dict": generator.state_dict(), "completed_epochs": epoch + 1},
                   checkpoints / f"generator_epoch_{epoch + 1:03d}.pt")
        with (output_dir / "history.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(history)
        _json(output_dir / "estado.json", {"stage": "training", "completed_epochs": epoch + 1,
            "total_epochs": epochs, "sampled_class_counts_last_epoch": {CLASS_NAMES[i]: sampled_counts[i] for i in range(len(CLASS_NAMES))},
            "test_used": False})
        _save_grid(generator, output_dir / f"generated_epoch_{epoch + 1:03d}.png", torch_device, seed + 10000)
    generator_checkpoint = checkpoints / "generator.pt"
    discriminator_checkpoint = checkpoints / "discriminator.pt"
    torch.save({"configuration": config, "state_dict": generator.state_dict()}, generator_checkpoint)
    torch.save({"configuration": config, "state_dict": discriminator.state_dict()}, discriminator_checkpoint)
    _save_grid(generator, output_dir / "generated_grid.png", torch_device, seed + 10000)
    _save_losses(history, output_dir / "losses.png")
    synthetic = generate_synthetic(generator, output_dir, torch_device, synthetic_per_class, seed + 20000, run_id)
    result = {"stage": "complete", "configuration": config, "generator_run_id": run_id,
        "completed_epochs": epochs, "total_steps": len(history), "training_seconds_this_invocation": time.monotonic() - started,
        "generator_checkpoint": str(generator_checkpoint), "discriminator_checkpoint": str(discriminator_checkpoint),
        "training_checkpoint": str(last_checkpoint), "history_path": str(output_dir / "history.csv"),
        "synthetic_manifest": str(output_dir / "synthetic_train.csv"), "synthetic_images": len(synthetic),
        "synthetic_class_counts": dict(Counter(r["label"] for r in synthetic)),
        "test_used": False, "final_losses": {key: history[-1][key] for key in ("loss_d", "loss_g")},
        "limitations": [
            "Small conditional GAN pilot; generated images and requested labels require visual/expert review.",
            "GAN losses are not image quality or classification accuracy metrics.",
            "Synthetic images may only augment training; validation and test remain real and unchanged.",
            "This generator saw all fixed training images. Do not use its synthetic images in held-out CV folds; train a fold-specific GAN for that.",
            "Source v5 training crops include inherited shear/exposure augmentations."
        ]}
    _json(result_path, result)
    _json(output_dir / "estado.json", {"stage": "complete", "completed_epochs": epochs, "synthetic_images": len(synthetic), "test_used": False})
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("train",))
    parser.add_argument("--dataset", type=Path, default=ROOT / "dataset_preparado")
    parser.add_argument("--output", type=Path, default=ROOT / "experimentos" / "transfer_01" / "gan")
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--steps-per-epoch", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--synthetic-per-class", type=int, default=256)
    args = parser.parse_args()
    print(json.dumps(train_gan(args.dataset, args.output, args.device, args.epochs,
        args.steps_per_epoch, args.batch_size, workers=args.workers, synthetic_per_class=args.synthetic_per_class), indent=2), flush=True)


if __name__ == "__main__":
    main()
