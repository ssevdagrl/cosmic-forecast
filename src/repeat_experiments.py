
"""
Cosmic Forecast - Tekrarlı Model Eğitimleri

Amaç:
    GRU ve CNN modellerinin farklı rastgele başlangıçlar
    ve süpernova veri bölmeleri altında performansını ölçmek.

Kullanım:
    python src/repeat_experiments.py --mode init --seeds 41 42 43

    python src/repeat_experiments.py --mode split --seeds 21 22 23

Yeni modeller results/repeat_runs klasörüne kaydedilir.
Mevcut eğitilmiş modellerin üzerine yazılmaz.
"""

import argparse
import json
import random
import time

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    RESULTS,
    load_photometry,
    make_windows,
    scaling_from_history,
    split_by_object,
)
from models_extra import CNNForecast, GRUForecastFlexible
from train_gru import GRUForecast


# Mevcut deneylerde kullanılan model mimarileri.
CONFIG = {
    "gru24": (
        4,
        lambda: GRUForecast(hidden=32)
    ),
    "cnn24": (
        4,
        lambda: CNNForecast(horizon=4, lookback=12)
    ),
    "gru48": (
        8,
        lambda: GRUForecastFlexible(horizon=8, hidden=32)
    ),
    "cnn48": (
        8,
        lambda: CNNForecast(horizon=8, lookback=12)
    ),
}


def set_seed(seed):
    """Eğitim rastgeleliğini kontrol eder."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare(df, horizon, split_seed):
    """Veri pencerelerini ve kimlik bazlı ayrımı hazırlar."""

    raw_x, raw_y, names, _ = make_windows(
        df,
        lookback=12,
        horizon=horizon
    )

    parts = split_by_object(
        names,
        seed=split_seed
    )

    # Aynı süpernova farklı kümelerde bulunmamalıdır.
    groups = [
        set(names[parts == part])
        for part in ("train", "val", "test")
    ]

    if any(not group for group in groups):
        raise ValueError("Bos veri kumesi bulundu.")

    if any(
        groups[i] & groups[j]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        raise ValueError("Supernova kimligi veri sizintisi.")

    # Normalizasyon parametreleri yalnızca geçmişten gelir.
    mu, scale = scaling_from_history(raw_x)

    x = torch.from_numpy(
        ((raw_x - mu) / scale).astype(np.float32)
    ).unsqueeze(-1)

    y = torch.from_numpy(
        ((raw_y - mu) / scale).astype(np.float32)
    )

    return x, y, raw_x, raw_y, names, parts, mu, scale


def run_one(task, split_seed, init_seed, prepared, output_dir):
    """Tek bir modelin eğitim ve test işlemini yürütür."""

    horizon, build_model = CONFIG[task]

    x, y, raw_x, raw_y, names, parts, mu, scale = prepared

    # Her eğitimin başlangıcında rastgeleliği sıfırla.
    set_seed(init_seed)

    train = parts == "train"
    val = parts == "val"
    test = parts == "test"

    train_loader = DataLoader(
        TensorDataset(x[train], y[train]),
        batch_size=128,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(x[val], y[val]),
        batch_size=128,
        shuffle=False
    )

    model = build_model()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    mse = nn.MSELoss()

    best_val = float("inf")
    best_epoch = None
    best_weights = None
    stagnant = 0

    # Eğitim süresini ölç.
    start = time.perf_counter()

    for epoch in range(1, 21):

        # Eğitim aşaması.
        model.train()

        for xb, yb in train_loader:
            optimizer.zero_grad()

            loss = mse(model(xb), yb)
            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

        # Doğrulama aşaması.
        # Test kümesi model seçiminde kullanılmaz.
        model.eval()

        weighted_loss = 0.0
        val_count = 0

        with torch.inference_mode():
            for xb, yb in val_loader:
                batch_loss = mse(model(xb), yb).item()

                weighted_loss += batch_loss * len(xb)
                val_count += len(xb)

        val_loss = weighted_loss / val_count

        if not np.isfinite(val_loss):
            raise RuntimeError("Gecersiz validation loss.")

        # En iyi doğrulama sonucunun ağırlıklarını sakla.
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            best_epoch = epoch

            best_weights = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

            stagnant = 0

        else:
            stagnant += 1

            if stagnant >= 5:
                break

    training_seconds = time.perf_counter() - start

    if best_weights is None:
        raise RuntimeError("Gecerli model agirligi bulunamadi.")

    # En iyi epoch'un ağırlıklarını geri yükle.
    model.load_state_dict(best_weights)
    model.eval()

    # Test tahminini yalnızca eğitim tamamlandıktan sonra yap.
    with torch.inference_mode():
        prediction_norm = model(x[test]).numpy()

    prediction = (
        prediction_norm * scale[test] + mu[test]
    )

    truth = raw_y[test]
    err = truth - prediction

    # Her deney için eşleştirilmiş Persistence referansı.
    baseline = np.repeat(
        raw_x[test, -1:],
        horizon,
        axis=1
    )

    base_err = truth - baseline

    # Model ağırlıklarını ayrı klasöre kaydet.
    model_path = (
        output_dir
        / f"{task}_split{split_seed}_init{init_seed}.pt"
    )

    torch.save(best_weights, model_path)

    summary = {
        "task": task,
        "horizon_h": horizon * 6,
        "split_seed": split_seed,
        "init_seed": init_seed,
        "n_test_supernova": int(len(set(names[test]))),
        "n_test_windows": int(test.sum()),
        "best_epoch": best_epoch,
        "best_val_normalized_MSE": float(best_val),
        "training_seconds": round(training_seconds, 3),
        "MAE_counts_per_s": float(np.mean(np.abs(err))),
        "RMSE_counts_per_s": float(
            np.sqrt(np.mean(err ** 2))
        ),
        "baseline_MAE_counts_per_s": float(
            np.mean(np.abs(base_err))
        ),
        "checkpoint": model_path.name,
    }

    # Nesne bazlı istatistikler sonraki bootstrap
    # ve eşleştirilmiş karşılaştırmalar için saklanır.
    per_object = []
    test_names = names[test]

    for name in np.unique(test_names):

        mask = test_names == name

        per_object.append({
            "task": task,
            "horizon_h": horizon * 6,
            "split_seed": split_seed,
            "init_seed": init_seed,
            "Name": str(name),
            "n_windows": int(mask.sum()),
            "model_MAE_counts_per_s": float(
                np.mean(np.abs(err[mask]))
            ),
            "baseline_MAE_counts_per_s": float(
                np.mean(np.abs(base_err[mask]))
            ),
        })

    print(
        f"{task:5s} split={split_seed} init={init_seed} "
        f"epoch={best_epoch:2d} "
        f"MAE={summary['MAE_counts_per_s']:.4f} "
        f"RMSE={summary['RMSE_counts_per_s']:.4f} "
        f"egitim={training_seconds:.1f}s",
        flush=True
    )

    return summary, per_object


def main():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--mode",
        choices=("init", "split"),
        default="init"
    )

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[41, 42, 43]
    )

    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(CONFIG),
        default=list(CONFIG)
    )

    args = parser.parse_args()

    if (
        len(set(args.seeds)) != len(args.seeds)
        or any(seed < 0 for seed in args.seeds)
    ):
        parser.error(
            "Seed degerleri negatif olmamali ve tekrar etmemelidir."
        )

    # CPU eğitimlerinde iş parçacığı sayısını sabitle.
    torch.set_num_threads(
        min(4, torch.get_num_threads())
    )

    # Ana model klasörüne dokunmadan ayrı sonuç klasörü aç.
    output_dir = RESULTS / "repeat_runs" / args.mode

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    df = load_photometry()

    # Aynı veri bölmesini gereksiz yere yeniden üretme.
    cache = {}

    summaries = []
    all_objects = []

    for seed in args.seeds:

        # init: sabit split, farklı model başlangıcı
        # split: farklı split, sabit model başlangıcı
        split_seed = 42 if args.mode == "init" else seed
        init_seed = seed if args.mode == "init" else 42

        for task in args.models:

            horizon, _ = CONFIG[task]
            key = (horizon, split_seed)

            if key not in cache:
                cache[key] = prepare(
                    df,
                    horizon,
                    split_seed
                )

            summary, objects = run_one(
                task,
                split_seed,
                init_seed,
                cache[key],
                output_dir
            )

            summaries.append(summary)
            all_objects.extend(objects)

    summary_frame = pd.DataFrame(summaries)

    summary_frame.to_csv(
        output_dir / "metrics.csv",
        index=False
    )

    pd.DataFrame(all_objects).to_csv(
        output_dir / "object_metrics.csv",
        index=False
    )

    # Sabit veri bölmesindeki farklı başlangıçları özetle.
    if args.mode == "init":

        aggregate = summary_frame.groupby("task").agg(
            MAE_mean=("MAE_counts_per_s", "mean"),
            MAE_std=("MAE_counts_per_s", "std"),
            RMSE_mean=("RMSE_counts_per_s", "mean"),
            RMSE_std=("RMSE_counts_per_s", "std"),
        )

        aggregate.to_csv(
            output_dir / "aggregate.csv"
        )

        print(
            "\nSeed bazli ozet:\n",
            aggregate.to_string()
        )

    # Deneyin yeniden üretilebilmesi için ayarları sakla.
    protocol = {
        "mode": args.mode,
        "seeds": args.seeds,
        "models": args.models,
        "base_split_seed": 42,
        "base_init_seed": 42,
        "epochs_max": 20,
        "patience": 5,
        "batch_size": 128,
        "learning_rate": 0.001,
        "torch_version": torch.__version__,
        "test_selection": (
            "Only once after validation-based checkpoint selection"
        ),
    }

    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8"
    )

    print(f"\nCiktilar: {output_dir}")

    print(
        "Mevcut gru_model.pt ve diger ana modeller degistirilmedi."
    )


if __name__ == "__main__":
    main()
