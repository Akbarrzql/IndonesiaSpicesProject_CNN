"""Jalankan training EfficientNet-B0 dari terminal, dengan progres live.

    python train_efficientnet.py

Progres ditulis ke stdout dan ke logs/<model>.log, jadi bisa dipantau dengan:

    tail -f logs/efficientnet_b0.log

Notebook `train_efficientnet.ipynb` memakai hasil yang disimpan skrip ini,
sehingga tidak perlu melatih ulang hanya untuk membuat grafik.
"""

import sys
import time
from pathlib import Path

import data_pipeline as dp
import training as tr

MODEL_NAME = "efficientnet_b0"


class Tee:
    """Tulis ke stdout sekaligus ke file log."""

    def __init__(self, path):
        self.file = open(path, "a", buffering=1)

    def write(self, s):
        sys.__stdout__.write(s)
        sys.__stdout__.flush()
        self.file.write(s)

    def flush(self):
        sys.__stdout__.flush()
        self.file.flush()


def main():
    Path("logs").mkdir(exist_ok=True)
    sys.stdout = Tee(f"logs/{MODEL_NAME}.log")

    config = tr.TrainConfig(
        epochs=5, lr=1e-4, weight_decay=1e-4, batch_size=32,
        patience=5, augment="basic", use_class_weights=True,
    )

    dp.seed_everything(config.seed)
    device = dp.get_device()

    # num_workers rendah dan tanpa persistent workers: mesin ini 8 GB dan
    # worker tambahan justru memicu swap, yang jauh lebih mahal daripada
    # keuntungan paralelisasi data loading.
    loaders, ds = dp.build_dataloaders(
        batch_size=config.batch_size, num_workers=2, augment=config.augment
    )

    print("=" * 70, flush=True)
    print(f"Mulai   : {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"Model   : {MODEL_NAME}   device: {device}", flush=True)
    print(f"Data    : train {len(ds['train'])} / val {len(ds['val'])} / test {len(ds['test'])}", flush=True)
    print(f"Config  : epochs={config.epochs} lr={config.lr} batch={config.batch_size}", flush=True)
    print(f"Batch   : {len(loaders['train'])} per epoch", flush=True)
    print("=" * 70, flush=True)

    model = tr.build_model(MODEL_NAME, num_classes=len(ds["train"].classes))
    hasil = tr.fit(model, loaders, ds, model_name=MODEL_NAME, config=config,
                   device=device, log_every=20, progress=False)

    print("\nMenghitung metrik pada test set...", flush=True)
    evalres = tr.evaluate(model, loaders, ds, model_name=MODEL_NAME,
                          hasil_fit=hasil, device=device)
    m = evalres["metrics"]

    print("\n" + "=" * 70, flush=True)
    print(f"SELESAI {time.strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"  test accuracy : {m['accuracy']:.4f}", flush=True)
    print(f"  test macro-F1 : {m['f1_macro']:.4f}", flush=True)
    print(f"  epoch terbaik : {m['best_epoch']} dari {m['epochs_run']}", flush=True)
    print(f"  waktu training: {m['total_train_time_s']/60:.1f} menit", flush=True)
    print(f"  hasil         : {evalres['output_dir']}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
