#!/usr/bin/env python3
"""
Low-resource training and inference script for Indian cattle/buffalo breed detection.

Implements the pipeline described in the paper:
1) Custom 4-block CNN
2) VGG16 transfer learning (2-stage training)
3) Weighted probability ensemble (default alpha=0.38)

Design goal: run on modest laptops with practical defaults.
Use `--lite` for faster smoke training.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms as T

try:
    import cv2  # type: ignore

    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Metrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float


class CattleDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        transform: T.Compose,
        img_size: int = 224,
        use_clahe: bool = True,
        use_bilateral: bool = True,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self.img_size = img_size
        self.use_clahe = use_clahe
        self.use_bilateral = use_bilateral

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = load_preprocessed_image(
            img_path,
            img_size=self.img_size,
            use_clahe=self.use_clahe,
            use_bilateral=self.use_bilateral,
        )
        tensor = self.transform(image)
        return tensor, label


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transforms(train: bool) -> T.Compose:
    if train:
        return T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.RandomRotation(degrees=20),
                T.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.1,
                    hue=0.05,
                ),
                T.ToTensor(),
                T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
                T.RandomErasing(p=0.2, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
            ]
        )
    return T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def _pad_to_square_np(arr: np.ndarray) -> np.ndarray:
    h, w = arr.shape[:2]
    if h == w:
        return arr
    size = max(h, w)
    top = (size - h) // 2
    bottom = size - h - top
    left = (size - w) // 2
    right = size - w - left
    return np.pad(arr, ((top, bottom), (left, right), (0, 0)), mode="edge")


def load_preprocessed_image(
    img_path: Path,
    img_size: int = 224,
    use_clahe: bool = True,
    use_bilateral: bool = True,
) -> Image.Image:
    if HAS_CV2:
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            raise RuntimeError(f"Could not read image: {img_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = _pad_to_square_np(rgb)
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_CUBIC)

        if use_clahe:
            ycrcb = cv2.cvtColor(rgb, cv2.COLOR_RGB2YCrCb)
            y, cr, cb = cv2.split(ycrcb)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            y = clahe.apply(y)
            ycrcb = cv2.merge((y, cr, cb))
            rgb = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)

        if use_bilateral:
            rgb = cv2.bilateralFilter(rgb, d=9, sigmaColor=75, sigmaSpace=75)

        return Image.fromarray(rgb)

    # Fallback without OpenCV: padding + resize only.
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    arr = _pad_to_square_np(arr)
    img = Image.fromarray(arr).resize((img_size, img_size), Image.BICUBIC)
    return img


def find_classes(folder: Path) -> List[str]:
    classes = sorted([p.name for p in folder.iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"No class folders found in {folder}")
    return classes


def collect_samples_from_root(
    root: Path,
    class_to_idx: Dict[str, int],
    max_images_per_class: int | None,
    seed: int,
) -> List[Tuple[Path, int]]:
    rng = random.Random(seed)
    samples: List[Tuple[Path, int]] = []
    for class_name, class_idx in class_to_idx.items():
        class_dir = root / class_name
        if not class_dir.exists():
            continue
        files = [p for p in class_dir.rglob("*") if p.suffix.lower() in IMG_EXTS]
        files.sort()
        if max_images_per_class is not None and len(files) > max_images_per_class:
            files = rng.sample(files, max_images_per_class)
        samples.extend([(p, class_idx) for p in files])
    if not samples:
        raise RuntimeError(f"No images found in {root}")
    return samples


def stratified_split(
    samples: Sequence[Tuple[Path, int]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    per_class: Dict[int, List[Tuple[Path, int]]] = {}
    for item in samples:
        per_class.setdefault(item[1], []).append(item)

    rng = random.Random(seed)
    train, val, test = [], [], []
    for items in per_class.values():
        rng.shuffle(items)
        n = len(items)
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        if n_train + n_val >= n:
            n_val = max(1, n - n_train - 1)
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val:
                n_train -= 1
            else:
                n_val -= 1
        train.extend(items[:n_train])
        val.extend(items[n_train : n_train + n_val])
        test.extend(items[n_train + n_val :])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def build_splits(
    data_dir: Path, max_images_per_class: int | None, seed: int
) -> Tuple[List[str], List[Tuple[Path, int]], List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    pre_split = all((data_dir / s).is_dir() for s in ("train", "val", "test"))
    if pre_split:
        classes = find_classes(data_dir / "train")
        class_to_idx = {c: i for i, c in enumerate(classes)}
        train_samples = collect_samples_from_root(
            data_dir / "train", class_to_idx, max_images_per_class, seed
        )
        val_samples = collect_samples_from_root(
            data_dir / "val", class_to_idx, max_images_per_class, seed + 1
        )
        test_samples = collect_samples_from_root(
            data_dir / "test", class_to_idx, max_images_per_class, seed + 2
        )
        return classes, train_samples, val_samples, test_samples

    classes = find_classes(data_dir)
    class_to_idx = {c: i for i, c in enumerate(classes)}
    all_samples = collect_samples_from_root(data_dir, class_to_idx, max_images_per_class, seed)
    train_samples, val_samples, test_samples = stratified_split(
        all_samples, train_ratio=0.7, val_ratio=0.15, seed=seed
    )
    return classes, train_samples, val_samples, test_samples


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


def build_vgg16(num_classes: int) -> nn.Module:
    try:
        weights = models.VGG16_Weights.IMAGENET1K_V1
        model = models.vgg16(weights=weights)
    except Exception:
        # Offline fallback
        model = models.vgg16(weights=None)
        print("[WARN] Could not load ImageNet weights. Using randomly initialized VGG16.")

    for p in model.features.parameters():
        p.requires_grad = False

    model.classifier = nn.Sequential(
        nn.Linear(25088, 4096),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(1024, num_classes),
    )
    return model


def unfreeze_vgg_blocks_3_to_5(model: nn.Module) -> None:
    for p in model.features.parameters():
        p.requires_grad = False
    # VGG16: block3 starts around index 10 in features.
    for idx, layer in enumerate(model.features):
        if idx >= 10:
            for p in layer.parameters():
                p.requires_grad = True


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
) -> Tuple[float, float]:
    train = optimizer is not None
    model.train(mode=train)

    total_loss = 0.0
    total_correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=(scaler is not None)):
                logits = model(x)
                loss = criterion(logits, y)

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total += y.size(0)
        total_loss += loss.item() * y.size(0)

    return total_loss / max(1, total), total_correct / max(1, total)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    patience: int,
    ckpt_path: Path,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
) -> None:
    best_val_loss = float("inf")
    wait = 0
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            scaler=scaler,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            optimizer=None,
            scaler=None,
        )

        if scheduler is not None:
            scheduler.step(epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={tr_loss:.4f}, train_acc={tr_acc:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            wait = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    model.load_state_dict(torch.load(ckpt_path, map_location=device))


@torch.no_grad()
def predict_probs(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_probs: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu()
        all_probs.append(probs)
        all_labels.append(y.cpu())
    probs_np = torch.cat(all_probs, dim=0).numpy()
    labels_np = torch.cat(all_labels, dim=0).numpy()
    return probs_np, labels_np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Metrics:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) != 0)
    recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) != 0)
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) != 0,
    )

    accuracy = float((y_true == y_pred).mean())
    return Metrics(
        accuracy=accuracy,
        macro_precision=float(precision.mean()),
        macro_recall=float(recall.mean()),
        macro_f1=float(f1.mean()),
    )


def print_metrics(title: str, metrics: Metrics) -> None:
    print(
        f"{title}: "
        f"accuracy={metrics.accuracy:.4f}, "
        f"macro_precision={metrics.macro_precision:.4f}, "
        f"macro_recall={metrics.macro_recall:.4f}, "
        f"macro_f1={metrics.macro_f1:.4f}"
    )


def measure_ensemble_latency(
    cnn: nn.Module,
    vgg: nn.Module,
    device: torch.device,
    img_size: int,
    runs: int = 100,
) -> float:
    cnn.eval()
    vgg.eval()
    x = torch.randn(1, 3, img_size, img_size, device=device)

    # Warm-up
    for _ in range(10):
        _ = cnn(x)
        _ = vgg(x)
    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(runs):
        _ = cnn(x)
        _ = vgg(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) * 1000.0 / runs


def train_pipeline(args: argparse.Namespace, device: torch.device) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.lite:
        if args.max_images_per_class is None:
            args.max_images_per_class = 150
        args.epochs_cnn = min(args.epochs_cnn, 5)
        args.epochs_vgg_stage1 = min(args.epochs_vgg_stage1, 2)
        args.epochs_vgg_stage2 = min(args.epochs_vgg_stage2, 2)
        args.batch_size = min(args.batch_size, 6)

    classes, train_samples, val_samples, test_samples = build_splits(
        data_dir=data_dir,
        max_images_per_class=args.max_images_per_class,
        seed=args.seed,
    )
    print(f"Classes ({len(classes)}): {classes}")
    print(
        f"Samples -> train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )

    if not HAS_CV2:
        print("[WARN] OpenCV not available. CLAHE and bilateral filtering are skipped.")

    train_ds = CattleDataset(
        train_samples,
        transform=build_transforms(train=True),
        img_size=args.img_size,
        use_clahe=args.use_clahe,
        use_bilateral=args.use_bilateral,
    )
    val_ds = CattleDataset(
        val_samples,
        transform=build_transforms(train=False),
        img_size=args.img_size,
        use_clahe=args.use_clahe,
        use_bilateral=args.use_bilateral,
    )
    test_ds = CattleDataset(
        test_samples,
        transform=build_transforms(train=False),
        img_size=args.img_size,
        use_clahe=args.use_clahe,
        use_bilateral=args.use_bilateral,
    )

    pin_mem = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_mem,
    )

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # 1) Train custom CNN
    cnn = CustomCNN(num_classes=len(classes)).to(device)
    cnn_ckpt = output_dir / "custom_cnn_best.pt"
    cnn_opt = torch.optim.AdamW(
        cnn.parameters(), lr=args.lr_head, weight_decay=args.weight_decay
    )
    cnn_sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        cnn_opt, T_0=10, T_mult=2
    )
    print("\n[Training] Custom CNN")
    fit_model(
        model=cnn,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=cnn_opt,
        device=device,
        epochs=args.epochs_cnn,
        patience=args.patience,
        ckpt_path=cnn_ckpt,
        scheduler=cnn_sched,
    )

    # 2) Train VGG16 stage-1
    vgg = build_vgg16(num_classes=len(classes)).to(device)
    vgg_stage1_ckpt = output_dir / "vgg16_stage1_best.pt"
    vgg_stage2_ckpt = output_dir / "vgg16_best.pt"

    stage1_params = [p for p in vgg.parameters() if p.requires_grad]
    vgg_opt_stage1 = torch.optim.AdamW(
        stage1_params, lr=args.lr_head, weight_decay=args.weight_decay
    )
    vgg_sched_stage1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        vgg_opt_stage1, T_0=10, T_mult=2
    )
    print("\n[Training] VGG16 Stage-1 (frozen backbone)")
    fit_model(
        model=vgg,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=vgg_opt_stage1,
        device=device,
        epochs=args.epochs_vgg_stage1,
        patience=args.patience,
        ckpt_path=vgg_stage1_ckpt,
        scheduler=vgg_sched_stage1,
    )

    # 3) Train VGG16 stage-2
    unfreeze_vgg_blocks_3_to_5(vgg)
    backbone_params = [p for p in vgg.features.parameters() if p.requires_grad]
    head_params = list(vgg.classifier.parameters())
    vgg_opt_stage2 = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr_backbone},
            {"params": head_params, "lr": args.lr_head},
        ],
        weight_decay=args.weight_decay,
    )
    vgg_sched_stage2 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        vgg_opt_stage2, T_0=10, T_mult=2
    )
    print("\n[Training] VGG16 Stage-2 (fine-tune blocks 3-5)")
    fit_model(
        model=vgg,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=vgg_opt_stage2,
        device=device,
        epochs=args.epochs_vgg_stage2,
        patience=args.patience,
        ckpt_path=vgg_stage2_ckpt,
        scheduler=vgg_sched_stage2,
    )

    # 4) Evaluate single models + ensemble
    print("\n[Evaluation] Test set")
    cnn_probs, y_true = predict_probs(cnn, test_loader, device)
    vgg_probs, y_true_2 = predict_probs(vgg, test_loader, device)
    if not np.array_equal(y_true, y_true_2):
        raise RuntimeError("Label mismatch between model predictions. Check data loader order.")

    cnn_pred = cnn_probs.argmax(axis=1)
    vgg_pred = vgg_probs.argmax(axis=1)
    ens_probs = args.alpha * cnn_probs + (1.0 - args.alpha) * vgg_probs
    ens_pred = ens_probs.argmax(axis=1)

    m_cnn = compute_metrics(y_true, cnn_pred, num_classes=len(classes))
    m_vgg = compute_metrics(y_true, vgg_pred, num_classes=len(classes))
    m_ens = compute_metrics(y_true, ens_pred, num_classes=len(classes))
    print_metrics("Custom CNN", m_cnn)
    print_metrics("VGG16 FT", m_vgg)
    print_metrics("Ensemble", m_ens)

    metrics_out = {
        "custom_cnn": m_cnn.__dict__,
        "vgg16_finetuned": m_vgg.__dict__,
        "ensemble": m_ens.__dict__,
        "alpha": args.alpha,
        "classes": classes,
        "img_size": args.img_size,
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
    }

    if args.benchmark_latency:
        latency_ms = measure_ensemble_latency(
            cnn=cnn, vgg=vgg, device=device, img_size=args.img_size, runs=args.latency_runs
        )
        metrics_out["latency_ms_per_image"] = latency_ms
        print(f"Approx ensemble latency: {latency_ms:.2f} ms/image ({device.type})")

    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)

    metadata = {
        "classes": classes,
        "alpha": args.alpha,
        "img_size": args.img_size,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
        "use_clahe": args.use_clahe,
        "use_bilateral": args.use_bilateral,
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved outputs in: {output_dir.resolve()}")
    print(f"- {cnn_ckpt.name}")
    print(f"- {vgg_stage2_ckpt.name}")
    print("- metadata.json")
    print("- metrics.json")


@torch.no_grad()
def predict_pipeline(args: argparse.Namespace, device: torch.device) -> None:
    weights_dir = Path(args.weights_dir if args.weights_dir else args.output_dir)
    image_path = Path(args.image)

    meta_path = weights_dir / "metadata.json"
    cnn_path = weights_dir / "custom_cnn_best.pt"
    vgg_path = weights_dir / "vgg16_best.pt"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata: {meta_path}")
    if not cnn_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {cnn_path}")
    if not vgg_path.exists():
        raise FileNotFoundError(f"Missing model checkpoint: {vgg_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    classes = meta["classes"]
    alpha = float(meta.get("alpha", 0.38))
    img_size = int(meta.get("img_size", 224))
    use_clahe = bool(meta.get("use_clahe", True))
    use_bilateral = bool(meta.get("use_bilateral", True))

    cnn = CustomCNN(num_classes=len(classes)).to(device)
    vgg = build_vgg16(num_classes=len(classes)).to(device)
    cnn.load_state_dict(torch.load(cnn_path, map_location=device))
    vgg.load_state_dict(torch.load(vgg_path, map_location=device))
    cnn.eval()
    vgg.eval()

    img = load_preprocessed_image(
        image_path, img_size=img_size, use_clahe=use_clahe, use_bilateral=use_bilateral
    )
    transform = build_transforms(train=False)
    x = transform(img).unsqueeze(0).to(device)

    p_cnn = torch.softmax(cnn(x), dim=1)
    p_vgg = torch.softmax(vgg(x), dim=1)
    p_ens = alpha * p_cnn + (1.0 - alpha) * p_vgg

    topk = min(args.topk, len(classes))
    conf, idx = torch.topk(p_ens[0], k=topk)
    print(f"Image: {image_path}")
    print(f"Top-{topk} predictions:")
    for rank, (score, i) in enumerate(zip(conf.tolist(), idx.tolist()), start=1):
        print(f"{rank}. {classes[i]} -> {score * 100:.2f}%")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Custom CNN + VGG16 weighted ensemble for Indian cattle breed detection."
    )
    parser.add_argument("--mode", choices=["train", "predict"], default="train")
    parser.add_argument("--data-dir", type=str, default="", help="Dataset root path")
    parser.add_argument("--output-dir", type=str, default="runs_cattle_ensemble")
    parser.add_argument("--weights-dir", type=str, default="", help="Folder with saved models")
    parser.add_argument("--image", type=str, default="", help="Image path for prediction")
    parser.add_argument("--topk", type=int, default=3)

    # Training + system settings
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-images-per-class", type=int, default=None)
    parser.add_argument("--lite", action="store_true", help="Use faster low-resource settings")

    # Optimization settings
    parser.add_argument("--epochs-cnn", type=int, default=10)
    parser.add_argument("--epochs-vgg-stage1", type=int, default=4)
    parser.add_argument("--epochs-vgg-stage2", type=int, default=4)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr-head", type=float, default=1e-4)
    parser.add_argument("--lr-backbone", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.38)

    # Preprocessing and benchmark
    parser.add_argument(
        "--use-clahe", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--use-bilateral", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--benchmark-latency", action="store_true")
    parser.add_argument("--latency-runs", type=int, default=100)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)

    if args.mode == "train" and not args.data_dir:
        raise ValueError("--data-dir is required in train mode")
    if args.mode == "predict" and not args.image:
        raise ValueError("--image is required in predict mode")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.mode == "train":
        train_pipeline(args, device)
    else:
        predict_pipeline(args, device)


if __name__ == "__main__":
    main()
