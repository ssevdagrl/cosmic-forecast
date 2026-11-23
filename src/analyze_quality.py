
"""
Cosmic Forecast - Fotometrik Belirsizlik Analizi

Amac:
    72 saatlik gecmis olcumlerin belirsizligine gore
    Persistence, Ridge, GRU ve CNN modellerini
    24 ve 48 saatlik tahminlerde karsilastirmak.

Kalite gruplari:
    Dusuk belirsizlik
    Orta belirsizlik
    Yuksek belirsizlik

Esikler yalnizca egitim pencerelerinden hesaplanir.
Gelecek gozlemlerin belirsizlikleri kullanilmaz.

Kullanim:
    python src/analyze_quality.py
"""

import json

import numpy as np
import pandas as pd
import torch

from dataset import (
    RESULTS,
    load_photometry,
    make_windows,
    scaling_from_history,
    split_by_object,
)
from repeat_experiments import CONFIG


LOOKBACK = 12
SPLIT_SEED = 42
INIT_SEED = 42

GROUP_NAMES = ("Dusuk", "Orta", "Yuksek")


def bin_photometry(df):
    """Orijinal veri hazirlama koduyla ayni 6 saatlik binleri olustur."""

    bins = np.floor(
        df["BJD"].to_numpy() * 4.0 + 1e-7
    ).astype(np.int64)

    frame = pd.DataFrame({
        "Name": df["Name"].astype(str).to_numpy(),
        "bin": bins,
        "rate": df["CRate"].to_numpy(),
        "unc": df["e_CRate"].to_numpy(),
    })

    # Ayni bin icindeki akinin ve belirsizligin medyani.
    # Belirsizlik medyani bir kalite gostergesidir;
    # binlenmis akinin yayilmis hata tahmini degildir.
    return frame.groupby(
        ["Name", "bin"],
        sort=False
    ).agg(
        rate=("rate", "median"),
        unc=("unc", "median"),
    )


def historical_uncertainties(binned, names, t0, raw_x):
    """
    Her tahmin penceresinin 12 gecmis binine
    ait belirsizlikleri eslestirir.

    Gelecek zaman dilimleri kullanilmaz.
    """

    starts = np.rint(
        np.asarray(t0, dtype=np.float64) * 4.0
    ).astype(np.int64)

    history_bins = (
        starts[:, None]
        - np.arange(LOOKBACK, 0, -1)[None, :]
    )

    keys = pd.MultiIndex.from_arrays(
        [
            np.repeat(names.astype(str), LOOKBACK),
            history_bins.ravel(),
        ],
        names=["Name", "bin"]
    )

    aligned = binned.reindex(keys)

    matched_rates = aligned["rate"].to_numpy().reshape(
        -1, LOOKBACK
    )

    errors = aligned["unc"].to_numpy().reshape(
        -1, LOOKBACK
    )

    if (
        not np.isfinite(matched_rates).all()
        or not np.isfinite(errors).all()
    ):
        raise RuntimeError(
            "Pencerelerde eksik 6 saatlik bin eslesmesi var."
        )

    if not (errors > 0).all():
        raise RuntimeError(
            "Pozitif olmayan olcum belirsizligi bulundu."
        )

    # Yeni kalite verileri ile orijinal pencerelerin
    # ayni SN ve zaman dilimlerine ait oldugunu denetle.
    if not np.allclose(
        matched_rates,
        raw_x,
        rtol=1e-6,
        atol=2e-5
    ):
        maximum = np.max(
            np.abs(matched_rates - raw_x)
        )

        raise RuntimeError(
            f"CRate bin eslesmesi yanlis. "
            f"En buyuk fark: {maximum}"
        )

    # Her pencere icin gecmis belirsizliklerinin medyani.
    return np.median(errors, axis=1)


def neural_prediction(task, x_norm, mu, scale):
    """Daha once egitilmis sinir agi modelini kullan."""

    checkpoint = (
        RESULTS
        / "repeat_runs"
        / "init"
        / f"{task}_split{SPLIT_SEED}_init{INIT_SEED}.pt"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Model kaydi bulunamadi: {checkpoint}"
        )

    model = CONFIG[task][1]()

    weights = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(weights, strict=True)
    model.eval()

    with torch.inference_mode():

        inputs = torch.from_numpy(
            x_norm.astype(np.float32)
        ).unsqueeze(-1)

        predicted = model(inputs).cpu().numpy()

    return predicted * scale + mu


def analyze_horizon(df, binned, horizon):

    hours = horizon * 6

    # Mevcut deneylerdeki pencereleri yeniden olustur.
    x, y, names, t0 = make_windows(
        df,
        lookback=LOOKBACK,
        horizon=horizon
    )

    parts = split_by_object(
        names,
        seed=SPLIT_SEED
    )

    train = parts == "train"
    test = parts == "test"

    # Gelecek gozlemleri kullanmadan kaliteyi hesapla.
    quality = historical_uncertainties(
        binned,
        names,
        t0,
        x
    )

    # Kalite sinirlari SADECE egitim kumesinden alinir.
    low, high = np.quantile(
        quality[train],
        [1 / 3, 2 / 3]
    )

    if not low < high:
        raise RuntimeError(
            "Kalite esikleri esit cikti. "
            "Belirsizlik dagilimini inceleyin."
        )

    groups = np.where(
        quality <= low,
        GROUP_NAMES[0],
        np.where(
            quality <= high,
            GROUP_NAMES[1],
            GROUP_NAMES[2]
        )
    )

    # Modellerin orijinal gecmis temelli normalizasyonu.
    mu, scale = scaling_from_history(x)

    x_norm = (
        (x[test] - mu[test]) / scale[test]
    ).astype(np.float64)

    truth = y[test]
    test_x = x[test]

    test_mu = mu[test]
    test_scale = scale[test]

    test_names = names[test]
    test_groups = groups[test]
    test_quality = quality[test]

    # Persistence tahmini.
    predictions = {
        "Persistence": np.repeat(
            test_x[:, -1:],
            horizon,
            axis=1
        )
    }

    # Daha once egitilen Ridge modelini yukle.
    ridge_path = RESULTS / f"ridge_{hours}h.npz"

    if not ridge_path.is_file():
        raise FileNotFoundError(
            f"Ridge kaydi bulunamadi: {ridge_path}"
        )

    with np.load(ridge_path) as ridge:
        weights = ridge["weights"]
        bias = ridge["bias"]

    predictions["Ridge"] = (
        (x_norm @ weights + bias)
        * test_scale + test_mu
    )

    # Seed=42 ile egitilmis GRU ve CNN modelleri.
    for kind in ("GRU", "CNN"):

        task = f"{kind.lower()}{hours}"

        predictions[kind] = neural_prediction(
            task,
            x_norm,
            test_mu,
            test_scale
        )

    # Onceki deneylerle test kumesi tutarliligi.
    expected = {
        24: (2393, 1.647022),
        48: (2113, 2.050194),
    }

    expected_n, expected_mae = expected[hours]

    actual_mae = np.mean(
        np.abs(
            truth - predictions["Persistence"]
        )
    )

    if len(truth) != expected_n:
        raise RuntimeError(
            "Test pencere sayisi eski deneyle uyusmuyor."
        )

    if not np.isclose(
        actual_mae,
        expected_mae,
        atol=1e-4
    ):
        raise RuntimeError(
            "Persistence MAE eski deneyle uyusmuyor."
        )

    for model, forecast in predictions.items():

        if (
            forecast.shape != truth.shape
            or not np.isfinite(forecast).all()
        ):
            raise RuntimeError(
                f"Gecersiz tahmin: {model}"
            )

    summaries = []
    objects = []

    # Uc belirsizlik grubunu ayri ayri degerlendir.
    for group in GROUP_NAMES:

        mask = test_groups == group

        if not mask.any():
            continue

        group_names = test_names[mask]

        baseline_error = np.abs(
            truth[mask]
            - predictions["Persistence"][mask]
        )

        baseline_mae = float(
            baseline_error.mean()
        )

        for model, forecast in predictions.items():

            delta = truth[mask] - forecast[mask]

            mae = float(np.abs(delta).mean())

            rmse = float(
                np.sqrt(np.square(delta).mean())
            )

            # Her supernovanin grup icindeki hatasini
            # ayri kaydet; sonraki istatistiksel analizde
            # pencereleri bagimsiz kabul etmeyecegiz.
            for sn in np.unique(group_names):

                object_mask = (
                    mask & (test_names == sn)
                )

                object_error = (
                    truth[object_mask]
                    - forecast[object_mask]
                )

                objects.append({
                    "horizon_h": hours,
                    "quality_group": group,
                    "Name": str(sn),
                    "model": model,
                    "n_windows": int(object_mask.sum()),
                    "MAE_counts_per_s": float(
                        np.abs(object_error).mean()
                    ),
                })

            improvement = (
                100.0 * (baseline_mae - mae)
                / baseline_mae
                if baseline_mae > 0
                else np.nan
            )

            summaries.append({
                "horizon_h": hours,
                "quality_group": group,
                "model": model,
                "n_windows": int(mask.sum()),
                "n_supernovae": int(
                    np.unique(group_names).size
                ),
                "median_history_e_CRate": float(
                    np.median(test_quality[mask])
                ),
                "MAE_counts_per_s": mae,
                "RMSE_counts_per_s": rmse,
                "improvement_vs_persistence_pct": improvement,
            })

    summary = pd.DataFrame(summaries)

    print(
        f"\n=== {hours} SAAT - BELIRSIZLIK ANALIZI ==="
    )

    columns = [
        "quality_group",
        "model",
        "n_windows",
        "n_supernovae",
        "median_history_e_CRate",
        "MAE_counts_per_s",
        "improvement_vs_persistence_pct",
    ]

    print(
        summary[columns].round(4).to_string(index=False)
    )

    print(
        f"\nEgitimden belirlenen esikler: "
        f"Dusuk <= {low:.4f}, "
        f"Orta <= {high:.4f}, "
        f"Yuksek > {high:.4f} counts/s"
    )

    return (
        summary,
        pd.DataFrame(objects),
        {"low": float(low), "high": float(high)}
    )


def main():

    torch.set_num_threads(1)

    df = load_photometry()

    binned = bin_photometry(df)

    all_summaries = []
    all_objects = []
    thresholds = {}

    for horizon in (4, 8):

        summary, objects, limits = analyze_horizon(
            df,
            binned,
            horizon
        )

        all_summaries.append(summary)
        all_objects.append(objects)

        thresholds[f"{horizon * 6}h"] = limits

    output = RESULTS / "quality_analysis"

    output.mkdir(
        parents=True,
        exist_ok=True
    )

    pd.concat(
        all_summaries,
        ignore_index=True
    ).to_csv(
        output / "quality_summary_all.csv",
        index=False,
        encoding="utf-8-sig"
    )

    pd.concat(
        all_objects,
        ignore_index=True
    ).to_csv(
        output / "quality_objects_all.csv",
        index=False,
        encoding="utf-8-sig"
    )

    protocol = {
        "split_seed": SPLIT_SEED,
        "model_init_seed": INIT_SEED,
        "lookback_bins": LOOKBACK,
        "bin_duration_h": 6,
        "quality_measure": (
            "Median of 12 historical bins' median e_CRate"
        ),
        "group_thresholds": thresholds,
        "thresholds_fit_on": (
            "Training windows only, separately per horizon"
        ),
        "limitations": [
            "The bin median uncertainty is a quality proxy.",
            "It is not the propagated uncertainty of the flux median.",
            "One supernova may contribute to several quality groups.",
            "The groups do not describe physical explosion stages."
        ],
    }

    (
        output / "quality_protocol.json"
    ).write_text(
        json.dumps(
            protocol,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print(f"\nKaydedildi: {output}")
    print(
        "Mevcut veri hazirlama ve model dosyalari degistirilmedi."
    )


if __name__ == "__main__":
    main()
