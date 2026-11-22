"""GPU-augmented training loop shared by every model condition.

Images are decoded once into uint8 tensors; augmentation (random resized
crop, flip, brightness/contrast jitter) runs on the GPU per batch via
torchvision v2 transforms. Optimization is AdamW with a cosine schedule
and one warmup epoch, label smoothing 0.1, optional inverse-frequency
class weights, AMP fp16 on CUDA, and early stopping on validation
macro-F1 keeping the best checkpoint.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import v2

from vehicle_cnn.models.resnet import NUM_CLASSES

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

AUGMENT = v2.Compose(
    [
        v2.RandomResizedCrop(64, scale=(0.8, 1.0), antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2),
    ]
)


def seed_everything(seed: int) -> None:
    import random  # noqa: PLC0415

    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_model_input(
    x_uint8: torch.Tensor,
    device: torch.device,
    input_size: int = 64,
    augment: bool = False,
) -> torch.Tensor:
    """uint8 NCHW tensor -> normalized float model input on ``device``."""
    x = x_uint8.to(device, non_blocking=True).float().div_(255.0)
    if augment:
        x = AUGMENT(x)
    if input_size != 64:
        x = F.interpolate(
            x, size=(input_size, input_size), mode="bilinear",
            align_corners=False,
        )
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    return (x - mean) / std


def class_weights(labels: np.ndarray, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse train class frequency normalized to mean 1."""
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


@dataclass
class TrainConfig:
    kind: str = "custom"
    seed: int = 0
    weighted: bool = False
    max_epochs: int = 40
    patience: int = 8
    batch_size: int = 256
    lr: float | None = None
    input_size: int = 64
    label_smoothing: float = 0.1
    weight_decay: float = 0.05
    num_classes: int = NUM_CLASSES

    def resolved_lr(self) -> float:
        if self.lr is not None:
            return self.lr
        return 3e-4 if self.kind == "resnet_pretrained" else 1e-3

    def resolved_epochs_patience(self) -> tuple[int, int]:
        if self.kind == "resnet_pretrained":
            return min(self.max_epochs, 15), min(self.patience, 4)
        return self.max_epochs, self.patience


@dataclass
class TrainResult:
    config: TrainConfig
    best_epoch: int
    best_val_macro_f1: float
    epochs_run: int
    train_seconds: float
    state_dict: dict[str, torch.Tensor]
    history: list[dict[str, float]] = field(default_factory=list)
    weight_sha256: str = ""


def _macro_f1(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int
) -> float:
    f1s = []
    for c in range(num_classes):
        tp = int(((y_pred == c) & (y_true == c)).sum())
        fp = int(((y_pred == c) & (y_true != c)).sum())
        fn = int(((y_pred != c) & (y_true == c)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom else 0.0)
    return float(np.mean(f1s))


@torch.no_grad()
def predict_labels(
    model: nn.Module,
    x_uint8: torch.Tensor,
    batch_size: int,
    device: torch.device,
    input_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Argmax labels and softmax probabilities for a uint8 tensor set."""
    model.eval()
    preds = np.empty(len(x_uint8), dtype=np.int64)
    probs = np.empty((len(x_uint8), NUM_CLASSES), dtype=np.float64)
    for start in range(0, len(x_uint8), batch_size):
        xb = to_model_input(
            x_uint8[start : start + batch_size], device, input_size
        )
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=device.type == "cuda"
        ):
            logits = model(xb)
        p = torch.softmax(logits.float(), dim=1).cpu().numpy()
        probs[start : start + len(p)] = p
        preds[start : start + len(p)] = p.argmax(axis=1)
    return preds, probs


def train_model(
    model: nn.Module,
    x_train: torch.Tensor,
    y_train: np.ndarray,
    x_val: torch.Tensor,
    y_val: np.ndarray,
    config: TrainConfig,
    device: torch.device,
) -> TrainResult:
    """Train ``model`` with early stopping on val macro-F1."""
    from vehicle_cnn.evaluation.metrics import macro_f1_score  # noqa: PLC0415

    seed_everything(config.seed)
    model.to(device)
    max_epochs, patience = config.resolved_epochs_patience()
    n = len(x_train)
    steps_per_epoch = (n + config.batch_size - 1) // config.batch_size
    total_steps = max_epochs * steps_per_epoch
    warmup_steps = steps_per_epoch

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.resolved_lr(),
        weight_decay=config.weight_decay,
    )

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return float(0.5 * (1.0 + np.cos(np.pi * min(progress, 1.0))))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    weights = (
        class_weights(y_train, config.num_classes).to(device)
        if config.weighted
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=weights, label_smoothing=config.label_smoothing
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    y_train_t = torch.from_numpy(y_train.astype(np.int64))

    best_f1 = -1.0
    best_epoch = 0
    best_state: dict[str, torch.Tensor] = {}
    bad_epochs = 0
    history: list[dict[str, float]] = []
    t0 = time.time()
    gen = torch.Generator().manual_seed(config.seed)

    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=gen)
        total_loss = 0.0
        for start in range(0, n, config.batch_size):
            idx = perm[start : start + config.batch_size]
            xb = to_model_input(
                x_train[idx], device, config.input_size, augment=True
            )
            yb = y_train_t[idx].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda", dtype=torch.float16, enabled=use_amp
            ):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_loss += float(loss.detach()) * len(idx)

        val_pred, _ = predict_labels(
            model, x_val, config.batch_size, device, config.input_size
        )
        f1 = macro_f1_score(y_val, val_pred, config.num_classes)
        acc = float((val_pred == y_val).mean())
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / n,
                "val_macro_f1": f1,
                "val_accuracy": acc,
                "lr": float(sched.get_last_lr()[0]),
            }
        )
        if f1 > best_f1:
            best_f1 = f1
            best_epoch = epoch
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    train_seconds = time.time() - t0
    if not best_state:
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }
        best_epoch = len(history)
        best_f1 = history[-1]["val_macro_f1"] if history else 0.0

    return TrainResult(
        config=config,
        best_epoch=best_epoch,
        best_val_macro_f1=best_f1,
        epochs_run=len(history),
        train_seconds=train_seconds,
        state_dict=best_state,
        history=history,
    )
