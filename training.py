"""Training & evaluasi untuk klasifikasi rempah Indonesia (Tahap 3).

Modul ini sengaja dibuat **arsitektur-agnostik**: `build_model("efficientnet_b0")`
dan `build_model("resnet18")` melewati jalur kode yang persis sama. Tugas ini
dikerjakan berkelompok — tiap anggota melatih arsitektur berbeda lalu hasilnya
dibandingkan — dan perbandingan itu hanya sahih kalau seluruh pipeline, split,
hyperparameter, dan metriknya identik. Kalau tiap orang menulis loop training
sendiri, perbedaan yang terukur bisa jadi berasal dari kode, bukan arsitektur.

Pemakaian:

    import data_pipeline as dp, training as tr

    loaders, ds = dp.build_dataloaders(batch_size=32, num_workers=4)
    model = tr.build_model("efficientnet_b0", num_classes=len(ds["train"].classes))
    hasil = tr.fit(model, loaders, ds, model_name="efficientnet_b0")
"""

from __future__ import annotations

import json
import platform
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torchvision import models
from tqdm import tqdm

import data_pipeline as dp

PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"
RESULT_DIR = PROJECT_DIR / "results"

# Registry arsitektur. Tiap entri: (konstruktor, bobot pretrained, nama atribut
# layer klasifikasi terakhir). Anggota kelompok lain cukup memakai kunci yang
# berbeda di sini -- tidak perlu menyentuh loop training.
ARCHITECTURES = {
    "alexnet": (models.alexnet, models.AlexNet_Weights.IMAGENET1K_V1, "classifier"),
    "vgg16": (models.vgg16, models.VGG16_Weights.IMAGENET1K_V1, "classifier"),
    "resnet18": (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1, "fc"),
    "resnet50": (models.resnet50, models.ResNet50_Weights.IMAGENET1K_V2, "fc"),
    "mobilenet_v2": (models.mobilenet_v2, models.MobileNet_V2_Weights.IMAGENET1K_V2, "classifier"),
    "efficientnet_b0": (models.efficientnet_b0, models.EfficientNet_B0_Weights.IMAGENET1K_V1, "classifier"),
}


def build_model(name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """Bangun model dengan layer klasifikasi diganti sesuai jumlah kelas."""
    if name not in ARCHITECTURES:
        raise ValueError(f"Arsitektur '{name}' belum terdaftar. Pilihan: {list(ARCHITECTURES)}")
    ctor, weights, head_attr = ARCHITECTURES[name]
    model = ctor(weights=weights if pretrained else None)

    head = getattr(model, head_attr)
    if isinstance(head, nn.Linear):
        setattr(model, head_attr, nn.Linear(head.in_features, num_classes))
    else:
        # classifier berupa Sequential (AlexNet/VGG/MobileNet/EfficientNet):
        # ganti hanya Linear terakhir, sisanya (Dropout dll) dipertahankan
        idx = max(i for i, m in enumerate(head) if isinstance(m, nn.Linear))
        head[idx] = nn.Linear(head[idx].in_features, num_classes)
    return model


def count_parameters(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e6
    return {"total_params": total, "trainable_params": trainable, "size_mb": size_mb}


@dataclass
class TrainConfig:
    """Hyperparameter. Harus SAMA untuk semua arsitektur yang dibandingkan."""
    epochs: int = 8
    lr: float = 1e-4
    weight_decay: float = 1e-4
    batch_size: int = 32
    label_smoothing: float = 0.0
    patience: int = 5              # early stopping pada val macro-F1
    seed: int = dp.SEED
    augment: str = "basic"
    use_class_weights: bool = True
    extra: dict = field(default_factory=dict)


def _sync(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def run_epoch(model, loader, criterion, device, optimizer=None, desc="", log_every=0,
              progress=False):
    """Satu epoch. optimizer=None berarti mode evaluasi.

    log_every > 0 mencetak progres tiap sekian batch, supaya training yang
    berjalan lama bisa dipantau dari file log tanpa menunggu epoch selesai.
    progress=True menampilkan progress bar tqdm (untuk notebook/terminal
    interaktif; jangan dipakai bersama file log karena bar ditulis berulang).
    """
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss, n_seen = 0.0, 0
    preds_all, labels_all = [], []
    n_batch = len(loader)
    t_mulai = time.perf_counter()

    # tqdm teks biasa (bukan tqdm.auto) agar tidak butuh ipywidgets; tetap
    # ter-update di tempat pada output sel Jupyter/PyCharm.
    bar = tqdm(loader, desc=desc, total=n_batch, unit="batch", leave=train_mode,
               file=sys.stdout, dynamic_ncols=True, disable=not progress)

    with torch.set_grad_enabled(train_mode):
        for i, (x, y) in enumerate(bar, start=1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if train_mode:
                optimizer.zero_grad(set_to_none=True)
            out = model(x)
            loss = criterion(out, y)
            if train_mode:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
            preds_all.append(out.argmax(1).detach().cpu())
            labels_all.append(y.detach().cpu())

            if progress:
                bar.set_postfix(loss=f"{total_loss/n_seen:.4f}")

            if log_every and (i % log_every == 0 or i == n_batch):
                berlalu = time.perf_counter() - t_mulai
                sisa = berlalu / i * (n_batch - i)
                print(f"    {desc} batch {i:4d}/{n_batch}  loss {total_loss/n_seen:.4f}  "
                      f"[{berlalu:.0f}s berlalu, ~{sisa:.0f}s lagi]", flush=True)

    bar.close()
    preds = torch.cat(preds_all).numpy()
    labels = torch.cat(labels_all).numpy()
    return total_loss / n_seen, float((preds == labels).mean()), preds, labels


def fit(model, loaders, datasets, model_name: str, config: TrainConfig | None = None,
        device: torch.device | None = None, verbose: bool = True, log_every: int = 0,
        progress: bool = True):
    """Latih model, simpan checkpoint terbaik berdasarkan val macro-F1.

    progress=True menampilkan progress bar per epoch (train & validation).
    """
    from sklearn.metrics import f1_score

    config = config or TrainConfig()
    device = device or dp.get_device()
    dp.seed_everything(config.seed)

    classes = datasets["train"].classes
    model = model.to(device)

    weight = dp.class_weights(datasets["train"]).to(device) if config.use_class_weights else None
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)

    MODEL_DIR.mkdir(exist_ok=True)
    ckpt_path = MODEL_DIR / f"{model_name}_best.pt"

    history = []
    best_f1, best_epoch, sejak_membaik = -1.0, -1, 0
    t_mulai = time.perf_counter()

    for epoch in range(1, config.epochs + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc, _, _ = run_epoch(
            model, loaders["train"], criterion, device, optimizer,
            desc=f"epoch {epoch}/{config.epochs} train", log_every=log_every,
            progress=progress)
        _sync(device)
        t_train = time.perf_counter() - t0

        t1 = time.perf_counter()
        va_loss, va_acc, va_pred, va_true = run_epoch(
            model, loaders["val"], criterion, device,
            desc=f"epoch {epoch}/{config.epochs} val  ", progress=progress)
        _sync(device)
        t_val = time.perf_counter() - t1

        va_f1 = f1_score(va_true, va_pred, average="macro", zero_division=0)
        scheduler.step()

        history.append({
            "epoch": epoch,
            "train_loss": tr_loss, "train_acc": tr_acc,
            "val_loss": va_loss, "val_acc": va_acc, "val_macro_f1": va_f1,
            "lr": optimizer.param_groups[0]["lr"],
            "train_time_s": t_train, "val_time_s": t_val,
        })

        membaik = va_f1 > best_f1
        if membaik:
            best_f1, best_epoch, sejak_membaik = va_f1, epoch, 0
            torch.save({"model_state": model.state_dict(), "classes": classes,
                        "epoch": epoch, "val_macro_f1": va_f1,
                        "model_name": model_name, "config": asdict(config)}, ckpt_path)
        else:
            sejak_membaik += 1

        if verbose:
            print(f"epoch {epoch:2d}/{config.epochs}  "
                  f"train loss {tr_loss:.4f} acc {tr_acc:.4f}  |  "
                  f"val loss {va_loss:.4f} acc {va_acc:.4f} F1 {va_f1:.4f}  "
                  f"({t_train:.0f}s+{t_val:.0f}s){'  *best' if membaik else ''}", flush=True)

        if sejak_membaik >= config.patience:
            if verbose:
                print(f"Early stopping: {config.patience} epoch tanpa perbaikan val macro-F1.")
            break

    total_time = time.perf_counter() - t_mulai

    # Kembalikan bobot terbaik, bukan bobot epoch terakhir
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    return {
        "model": model, "history": history, "best_epoch": best_epoch,
        "best_val_macro_f1": best_f1, "total_train_time_s": total_time,
        "config": config, "device": str(device), "checkpoint": str(ckpt_path),
    }


@torch.no_grad()
def predict(model, loader, device):
    """Kembalikan (probabilitas, prediksi, label sebenarnya) untuk satu split."""
    model.eval().to(device)
    probs, preds, trues = [], [], []
    for x, y in loader:
        out = model(x.to(device))
        p = torch.softmax(out.float(), dim=1).cpu()
        probs.append(p)
        preds.append(p.argmax(1))
        trues.append(y)
    return torch.cat(probs).numpy(), torch.cat(preds).numpy(), torch.cat(trues).numpy()


def measure_inference(model, device, batch_size=32, img_size=dp.IMG_SIZE, n=20):
    """Latensi inferensi setelah warmup.

    Panggilan pertama di MPS/CUDA memuat biaya kompilasi kernel, jadi tanpa
    warmup angkanya bisa puluhan kali lebih lambat dari kondisi sebenarnya.
    """
    model.eval().to(device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    with torch.no_grad():
        for _ in range(5):
            model(x)
        _sync(device)
        waktu = []
        for _ in range(n):
            t0 = time.perf_counter()
            model(x)
            _sync(device)
            waktu.append(time.perf_counter() - t0)
    median = float(np.median(waktu))
    return {
        "batch_size": batch_size,
        "latency_ms_per_batch": median * 1000,
        "latency_ms_per_image": median * 1000 / batch_size,
        "throughput_img_per_s": batch_size / median,
    }


def evaluate(model, loaders, datasets, model_name: str, hasil_fit: dict,
             device: torch.device | None = None, split: str = "test") -> dict:
    """Hitung seluruh metrik evaluasi + efisiensi, lalu simpan ke results/."""
    from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                                 confusion_matrix, classification_report, top_k_accuracy_score)

    device = device or dp.get_device()
    classes = datasets["train"].classes
    probs, preds, trues = predict(model, loaders[split], device)

    acc = accuracy_score(trues, preds)
    p_macro, r_macro, f_macro, _ = precision_recall_fscore_support(
        trues, preds, average="macro", zero_division=0)
    p_w, r_w, f_w, _ = precision_recall_fscore_support(
        trues, preds, average="weighted", zero_division=0)
    p_cls, r_cls, f_cls, sup_cls = precision_recall_fscore_support(
        trues, preds, average=None, labels=range(len(classes)), zero_division=0)

    cm = confusion_matrix(trues, preds, labels=range(len(classes)))
    infer = measure_inference(model, device, batch_size=hasil_fit["config"].batch_size)
    params = count_parameters(model)

    hist = hasil_fit["history"]
    metrics = {
        "model_name": model_name,
        "split": split,
        "accuracy": float(acc),
        "top5_accuracy": float(top_k_accuracy_score(trues, probs, k=5, labels=range(len(classes)))),
        "precision_macro": float(p_macro), "recall_macro": float(r_macro), "f1_macro": float(f_macro),
        "precision_weighted": float(p_w), "recall_weighted": float(r_w), "f1_weighted": float(f_w),
        **params,
        "best_epoch": hasil_fit["best_epoch"],
        "epochs_run": len(hist),
        "best_val_macro_f1": float(hasil_fit["best_val_macro_f1"]),
        "total_train_time_s": float(hasil_fit["total_train_time_s"]),
        "mean_epoch_time_s": float(np.mean([h["train_time_s"] for h in hist])),
        **{f"infer_{k}": v for k, v in infer.items()},
        "device": str(device),
        "platform": f"{platform.machine()} / torch {torch.__version__}",
        "config": asdict(hasil_fit["config"]),
        "n_train": len(datasets["train"]), "n_val": len(datasets["val"]), "n_test": len(datasets["test"]),
        "num_classes": len(classes),
    }

    per_class = {
        "class": classes,
        "support": sup_cls.tolist(),
        "precision": p_cls.tolist(),
        "recall": r_cls.tolist(),
        "f1": f_cls.tolist(),
    }

    out_dir = RESULT_DIR / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "per_class.json").write_text(json.dumps(per_class, indent=2))
    (out_dir / "history.json").write_text(json.dumps(hist, indent=2))
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",",
               header=",".join(classes), comments="")
    np.save(out_dir / "test_probs.npy", probs)
    np.save(out_dir / "test_preds.npy", preds)
    np.save(out_dir / "test_true.npy", trues)
    (out_dir / "classification_report.txt").write_text(
        classification_report(trues, preds, target_names=classes, zero_division=0, digits=4))

    return {"metrics": metrics, "per_class": per_class, "confusion_matrix": cm,
            "probs": probs, "preds": preds, "trues": trues, "classes": classes,
            "output_dir": str(out_dir)}


def load_run(model_name: str, model: nn.Module, device: torch.device | None = None) -> dict:
    """Muat checkpoint + history hasil training yang sudah dijalankan sebelumnya.

    Dipakai notebook agar tidak perlu melatih ulang hanya untuk membuat grafik:
    training dijalankan sekali lewat `train_efficientnet.py`, notebook memakai
    hasilnya. Bentuk kembaliannya sama dengan `fit()`.
    """
    device = device or dp.get_device()
    ckpt_path = MODEL_DIR / f"{model_name}_best.pt"
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state"])

    history = json.loads((RESULT_DIR / model_name / "history.json").read_text())
    cfg = TrainConfig(**{k: v for k, v in ckpt["config"].items()})
    best = max(history, key=lambda h: h["val_macro_f1"])
    return {
        "model": model.to(device), "history": history,
        "best_epoch": best["epoch"], "best_val_macro_f1": best["val_macro_f1"],
        "total_train_time_s": sum(h["train_time_s"] + h["val_time_s"] for h in history),
        "config": cfg, "device": str(device), "checkpoint": str(ckpt_path),
    }


def load_results(model_name: str) -> dict:
    """Muat hasil tersimpan -- dipakai notebook perbandingan antar anggota kelompok."""
    out_dir = RESULT_DIR / model_name
    return {
        "metrics": json.loads((out_dir / "metrics.json").read_text()),
        "per_class": json.loads((out_dir / "per_class.json").read_text()),
        "history": json.loads((out_dir / "history.json").read_text()),
    }
