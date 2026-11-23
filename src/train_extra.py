"""
Cosmic Forecast - Ek Sinir Ağı Modellerinin Eğitimi

Bu dosya, 24 saatlik CNN ile 48 saatlik GRU ve CNN modellerini eğitir.
Daha önce eğitilmiş 24 saatlik GRU'nun ağırlıkları değiştirilmez.

Proje ana klasöründen kullanım:
    python src/train_extra.py --all
    python src/train_extra.py --models cnn24
    python src/train_extra.py --models gru48 cnn48

Modellerin bağımsız test kümesindeki değerlendirmesi evaluate_all.py
üzerinden yapılır; bu dosyada test kümesiyle model seçimi yapılmaz.
"""

import argparse
import json
import random

import matplotlib

# Grafikler ekran açılmadan doğrudan dosyaya kaydedilir.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    RESULTS,
    load_photometry,
    make_windows,
    split_by_object,
    scaling_from_history,
)
from models_extra import CNNForecast, GRUForecastFlexible


# Aynı rastgele başlangıç koşullarını tekrar oluşturmak için sabit tohum.
SEED = 42

# Modellerin mimarisi, tahmin ufku ve ağırlık dosyası adları.
# Bir zaman adımı 6 saattir: 4 adım = 24 saat, 8 adım = 48 saat.
CONFIG = {
    "cnn24": {
        "architecture": "CNN",
        "horizon": 4,
        "filename": "cnn_24h_model.pt",
    },
    "gru48": {
        "architecture": "GRU",
        "horizon": 8,
        "filename": "gru_48h_model.pt",
    },
    "cnn48": {
        "architecture": "CNN",
        "horizon": 8,
        "filename": "cnn_48h_model.pt",
    },
}


def set_seed(seed=SEED):
    """Python, NumPy ve PyTorch için rastgelelik tohumunu ayarlar."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # CPU üzerinde kullanılan PyTorch iş parçacığı sayısını sınırla.
    if hasattr(torch, "set_num_threads"):
        torch.set_num_threads(min(torch.get_num_threads(), 4))


def make_model(architecture, horizon):
    """İstenen mimariyi belirtilen tahmin adımı sayısıyla oluşturur."""
    if architecture == "GRU":
        return GRUForecastFlexible(horizon=horizon, hidden=32)
    if architecture == "CNN":
        return CNNForecast(horizon=horizon, lookback=12)
    raise ValueError(architecture)


def prepare(df, horizon):
    """Pencereleri hazırlar, geçmişe göre ölçekler ve nesne bazlı ayırır."""
    # Her örnek için 12 adet 6 saatlik geçmiş gözlem (72 saat) kullanılır.
    raw_x, raw_y, names, _ = make_windows(
        df, lookback=12, horizon=horizon
    )

    # Aynı süpernovaya ait pencereler farklı kümelere dağıtılmaz.
    partitions = split_by_object(names, seed=SEED)

    # Ortalama ve ölçek yalnızca geçmiş 72 saatlik girdiden hesaplanır.
    # Gelecekteki hedef değerler normalizasyon parametrelerine katılmaz.
    mu, scale = scaling_from_history(raw_x)
    x = ((raw_x - mu) / scale).astype("float32")
    y = ((raw_y - mu) / scale).astype("float32")

    # Eğitim, doğrulama ve test kümeleri süpernova kimliğine göre ayrık olmalı.
    sets = [
        set(names[partitions == group])
        for group in ["train", "val", "test"]
    ]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    if any(not s for s in sets):
        raise ValueError("Train/validation/test groups must be nonempty.")

    return x, y, names, partitions


def loaders(x, y, partitions, batch_size):
    """Yalnızca eğitim ve doğrulama kümeleri için mini-batch yükleyicileri kurar."""
    result = {}
    for phase in ["train", "val"]:
        ix = partitions == phase

        # GRU ve CNN için giriş boyutu: (örnek, zaman, özellik).
        ds = TensorDataset(
            torch.from_numpy(x[ix]).unsqueeze(-1),
            torch.from_numpy(y[ix]),
        )

        # Eğitim verisi karıştırılır; doğrulama verisi karıştırılmaz.
        result[phase] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(phase == "train"),
        )
    return result


def fit_one(task, df, args):
    """Tek bir modeli eğitir; en iyi doğrulama ağırlıklarını ve kayıtları saklar."""
    cfg = CONFIG[task]

    # Her model aynı tohumla bağımsız olarak başlatılır.
    set_seed(SEED)
    x, y, names, partitions = prepare(df, cfg["horizon"])
    batches = loaders(x, y, partitions, args.batch_size)
    model = make_model(cfg["architecture"], cfg["horizon"])

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()
    history = []
    best_val = float("inf")
    best_weights = None
    stalled = 0

    print(
        f'\n[{task}] {cfg["architecture"]}, 72h -> {cfg["horizon"] * 6}h;'
        f" windows={len(names)}; independent objects={len(set(names))}",
        flush=True,
    )

    for epoch in range(1, args.epochs + 1):
        # --------------------------------------------------
        # Eğitim: ileri yayılım, MSE, geri yayılım, ağırlık güncelleme.
        # --------------------------------------------------
        model.train()
        train_weighted, n_train = 0.0, 0

        for xb, yb in batches["train"]:
            optimizer.zero_grad()
            loss = mse(model(xb), yb)
            loss.backward()

            # Aşırı büyüyen gradyanları sınırlayarak eğitimi kararlı tut.
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            # Son mini-batch küçük olabileceği için örnek sayısıyla ağırlıklandır.
            train_weighted += loss.item() * len(xb)
            n_train += len(xb)

        # --------------------------------------------------
        # Doğrulama: ağırlıkları değiştirmeden hata hesapla.
        # --------------------------------------------------
        model.eval()
        val_weighted, n_val = 0.0, 0

        with torch.no_grad():
            for xb, yb in batches["val"]:
                val_weighted += mse(model(xb), yb).item() * len(xb)
                n_val += len(xb)

        train_loss = train_weighted / n_train
        val_loss = val_weighted / n_val

        # Normalize edilmiş MSE değerlerini epoch bazında kaydet.
        history.append({
            "epoch": epoch,
            "train_normalized_MSE": train_loss,
            "val_normalized_MSE": val_loss,
        })
        print(
            f"  epoch {epoch:02d} train={train_loss:.5f} val={val_loss:.5f}",
            flush=True,
        )

        # Early stopping: doğrulama kaybı yeterince iyileşmiyorsa dur.
        # En iyi dönemin ağırlıklarının kopyası ayrıca saklanır.
        if val_loss < best_val - 1e-5:
            best_val, stalled = val_loss, 0
            best_weights = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            best_epoch = epoch
        else:
            stalled += 1
            if stalled >= args.patience:
                print("  early stopping")
                break

    if best_weights is None:
        raise RuntimeError(f"No valid checkpoint for {task}.")

    # En iyi doğrulama döneminin ağırlıklarını kaydet.
    # Orijinal 24 saatlik GRU dosyası bu CONFIG içinde bulunmaz.
    RESULTS.mkdir(exist_ok=True, parents=True)
    torch.save(best_weights, RESULTS / cfg["filename"])

    # Eğitim ve doğrulama kayıplarını CSV dosyasına yaz.
    pd.DataFrame(history).to_csv(
        RESULTS / f"{task}_training_history.csv",
        index=False,
    )

    # Kayıp değişimini gösteren eğitim eğrisini oluştur.
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        [h["epoch"] for h in history],
        [h["train_normalized_MSE"] for h in history],
        label="Train",
    )
    ax.plot(
        [h["epoch"] for h in history],
        [h["val_normalized_MSE"] for h in history],
        label="Validation",
    )
    ax.set(
        xlabel="Epoch",
        ylabel="Normalized MSE",
        title=f"{task}: training history",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(RESULTS / f"{task}_training_curve.png", dpi=160)
    plt.close(fig)

    # Deneyin tekrar edilebilmesi için veri kapsamı, eğitim parametreleri
    # ve en iyi doğrulama dönemi JSON dosyasında saklanır.
    manifest = {
        "task": task,
        "architecture": cfg["architecture"],
        "lookback_bins": 12,
        "horizon_bins": cfg["horizon"],
        "bin_width_hours": 6,
        "seed": SEED,
        "max_per_object": 60,
        "n_train_windows": int((partitions == "train").sum()),
        "n_val_windows": int((partitions == "val").sum()),
        "n_test_windows": int((partitions == "test").sum()),
        "n_train_supernova": int(len(set(names[partitions == "train"]))),
        "n_val_supernova": int(len(set(names[partitions == "val"]))),
        "n_test_supernova": int(len(set(names[partitions == "test"]))),
        "best_epoch": best_epoch,
        "best_val_normalized_MSE": float(best_val),
        "epochs_requested": args.epochs,
        "patience": args.patience,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "note": "Test-set not inspected for hyperparameter selection; split by SN Name.",
    }
    (RESULTS / f"{task}_training_config.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f'  saved {cfg["filename"]}; best validation epoch={best_epoch}')


def main():
    """Komut satırından seçilen model veya modellerin eğitimini başlatır."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="Train cnn24, gru48 and cnn48."
    )
    parser.add_argument(
        "--models", nargs="+", choices=list(CONFIG), help="Choose tasks."
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    # Eğitim parametrelerinin pozitif olduğundan emin ol.
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1 or args.lr <= 0:
        parser.error("epochs, patience, batch-size, and lr must be positive.")

    # --all veya seçenek verilmemesi durumunda üç modelin tamamı eğitilir.
    tasks = list(CONFIG) if args.all or args.models is None else args.models

    # 24 saatlik GRU bağımsız dosyada eğitilir; burada üzerine yazılmaz.
    if not (RESULTS / "gru_model.pt").exists():
        print(
            "NOTE: Existing gru_model.pt not found. To compare 24h models, "
            "run python src/train_gru.py."
        )

    print("Loading TESS light curves...")
    df = load_photometry()

    for task in tasks:
        fit_one(task, df, args)

    print("\nFinished training. Next command: python src/evaluate_all.py")


if __name__ == "__main__":
    main()
