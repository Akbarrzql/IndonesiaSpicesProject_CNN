"""Buat folder data_split/ (train / val / test) dari dataset Kaggle Indonesian Spices.

Isinya SAMA dengan prepare_data.ipynb: unduh -> buang duplikat -> split stratified
70/15/15 (seed 42). Bedanya, path otomatis mengikuti lokasi file ini, jadi tidak ada
yang perlu diedit.

Cara pakai (dari folder repo, venv aktif):
    python buat_data_split.py

Di akhir, skrip memeriksa sendiri apakah hasilnya cocok dengan data yang dipakai
EfficientNet-B0 (4221 / 905 / 905, plus jumlah test per kelas). Kalau tidak cocok,
JANGAN training.
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "data_split"

KAGGLE_DATASET = "albertnathaniel12/indonesian-spices-dataset"
DATASET_FOLDER = "Indonesian Spices Dataset"

TRAIN_RATIO, VAL_RATIO, TEST_RATIO = 0.70, 0.15, 0.15
SEED = 42
IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DHASH_SIZE = 8
THUMB_SIZE = 32
RMSE_THRESHOLD = 5.0   # di bawah ini = benar-benar duplikat

# Hasil yang tercatat saat EfficientNet-B0 dilatih (results/efficientnet_b0/metrics.json)
EXPECTED = {"train": 4221, "val": 905, "test": 905}


def cari_sumber() -> Path:
    """Unduh dataset dari Kaggle (kagglehub memakai cache bila sudah pernah diunduh)."""
    import kagglehub

    print("1/5 Mengunduh dataset (memakai cache kalau sudah ada)...", flush=True)
    path = Path(kagglehub.dataset_download(KAGGLE_DATASET))
    print(f"    Lokasi cache : {path}")
    src = path / DATASET_FOLDER
    if not src.is_dir():
        isi = sorted(p.name for p in path.iterdir())
        sys.exit(f"STOP: folder '{DATASET_FOLDER}' tidak ada di {path}. Isinya: {isi}")
    return src


def kumpulkan(src: Path) -> pd.DataFrame:
    records = []
    for class_dir in sorted(p for p in src.iterdir() if p.is_dir()):
        for img_path in sorted(class_dir.rglob("*")):
            if img_path.is_file() and img_path.suffix.lower() in IMG_EXTENSIONS:
                records.append({"filepath": img_path, "label": class_dir.name})
    df = pd.DataFrame(records)
    print(f"2/5 Dataset asli: {df['label'].nunique()} kelas, {len(df)} gambar")
    return df


def image_signature(path: Path):
    """(md5, dhash, thumbnail 32x32 grayscale) untuk satu file."""
    raw = path.read_bytes()
    with Image.open(path) as im:
        im.load()
        g = im.convert("L")
        small = g.resize((DHASH_SIZE + 1, DHASH_SIZE), Image.LANCZOS)
        a = np.asarray(small, dtype=np.int16)
        dh = np.packbits((a[:, 1:] > a[:, :-1]).flatten()).tobytes().hex()
        thumb = np.asarray(g.resize((THUMB_SIZE, THUMB_SIZE), Image.BILINEAR), dtype=np.float32)
    return hashlib.md5(raw).hexdigest(), dh, thumb


def buang_duplikat(df: pd.DataFrame) -> pd.DataFrame:
    print("3/5 Mendeteksi duplikat (beberapa menit)...", flush=True)
    sigs = [image_signature(p) for p in tqdm(df["filepath"], desc="hash citra")]
    df = df.copy()
    df["md5"] = [s[0] for s in sigs]
    df["dhash"] = [s[1] for s in sigs]
    thumbs = {p: s[2] for p, s in zip(df["filepath"], sigs)}

    def max_rmse(paths):
        t = [thumbs[p] for p in paths]
        return max(
            (float(np.sqrt(((t[i] - t[j]) ** 2).mean()))
             for i in range(len(t)) for j in range(i + 1, len(t))),
            default=0.0,
        )

    drop_duplicate, drop_conflict, false_positive = [], [], 0
    for _, grp in df[df.duplicated("dhash", keep=False)].groupby("dhash"):
        paths = list(grp["filepath"])
        if max_rmse(paths) >= RMSE_THRESHOLD:
            false_positive += 1
            continue
        if grp["label"].nunique() > 1:
            drop_conflict.extend(paths)          # gambar sama, label beda -> buang semua
        else:
            drop_duplicate.extend(sorted(paths)[1:])   # simpan satu wakil

    to_drop = set(drop_duplicate) | set(drop_conflict)
    bersih = df[~df["filepath"].isin(to_drop)].reset_index(drop=True)

    sisa = [
        dh for dh, grp in bersih[bersih.duplicated("dhash", keep=False)].groupby("dhash")
        if max_rmse(list(grp["filepath"])) < RMSE_THRESHOLD
    ]
    assert not sisa, "Masih ada duplikat tersisa"

    print(f"    Duplikat dibuang : {len(drop_duplicate)} | label konflik dibuang: {len(drop_conflict)} "
          f"| grup ditolak verifikasi piksel: {false_positive}")
    print(f"    Dibuang {len(df) - len(bersih)} citra, tersisa {len(bersih)}")
    assert bersih["label"].nunique() >= 5 and len(bersih) >= 1000
    return bersih


def bagi_dan_salin(df: pd.DataFrame) -> None:
    print("4/5 Membagi 70/15/15 (stratified, seed 42) dan menyalin file...", flush=True)
    random.seed(SEED)
    train_df, temp_df = train_test_split(
        df, test_size=(VAL_RATIO + TEST_RATIO), stratify=df["label"],
        random_state=SEED, shuffle=True,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=TEST_RATIO / (VAL_RATIO + TEST_RATIO), stratify=temp_df["label"],
        random_state=SEED, shuffle=True,
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}
    assert len(train_df) + len(val_df) + len(test_df) == len(df)

    # Hanya menghapus data_split/ milik repo ini
    assert OUTPUT_DIR.parent == PROJECT_DIR and OUTPUT_DIR.name == "data_split"
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    manifest_rows = []
    for split_name, part in splits.items():
        for label in df["label"].unique():
            (OUTPUT_DIR / split_name / label).mkdir(parents=True, exist_ok=True)
        for row in tqdm(part.itertuples(index=False), total=len(part), desc=f"copy {split_name}"):
            dest = OUTPUT_DIR / split_name / row.label / row.filepath.name
            shutil.copy2(row.filepath, dest)
            manifest_rows.append({
                "split": split_name, "label": row.label, "filename": row.filepath.name,
                "filepath": str(dest.relative_to(PROJECT_DIR)),
            })
    pd.DataFrame(manifest_rows).to_csv(OUTPUT_DIR / "manifest.csv", index=False)


def verifikasi() -> bool:
    print("5/5 Verifikasi hasil di disk...")
    counts = {
        s: sum(1 for f in (OUTPUT_DIR / s).rglob("*") if f.is_file())
        for s in ("train", "val", "test")
    }
    n_kelas = len([d for d in (OUTPUT_DIR / "train").iterdir() if d.is_dir()])
    print(f"    Kelas di train : {n_kelas}")
    print(f"    Jumlah gambar  : {counts}  (diharapkan {EXPECTED})")
    ok = counts == EXPECTED

    pc = PROJECT_DIR / "results" / "efficientnet_b0" / "per_class.json"
    if pc.exists():
        p = json.loads(pc.read_text())
        beda = []
        for kelas, support in zip(p["class"], p["support"]):
            d = OUTPUT_DIR / "test" / kelas
            n = sum(1 for f in d.iterdir() if f.is_file()) if d.is_dir() else -1
            if n != support:
                beda.append((kelas, support, n))
        print("    Jumlah test per kelas vs EfficientNet:",
              "COCOK semua" if not beda else f"BEDA {beda}")
        ok = ok and not beda
    else:
        print("    (results/efficientnet_b0/per_class.json tidak ada, cek per kelas dilewati)")
    return ok


def main() -> None:
    if not (PROJECT_DIR / "data_pipeline.py").exists():
        sys.exit("STOP: taruh file ini di folder repo (sejajar dengan data_pipeline.py).")
    src = cari_sumber()
    df = kumpulkan(src)
    df = buang_duplikat(df)
    bagi_dan_salin(df)
    ok = verifikasi()
    print()
    if ok:
        print("LOLOS: data_split/ sama dengan yang dipakai EfficientNet-B0. Boleh lanjut.")
    else:
        print("GAGAL: hasil tidak cocok. JANGAN training; kirim output ini untuk dicek.")
        sys.exit(1)


if __name__ == "__main__":
    main()
