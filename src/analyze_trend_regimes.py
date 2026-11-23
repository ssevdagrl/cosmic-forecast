
"""
Cosmic Forecast - Gozlenen Isik Egrisi Egilim Analizi

72 saatlik gecmis pencereleri artan, azalan ve
yatay egilim olarak siniflandirir.

Persistence, Ridge, GRU ve CNN modellerini
bu gruplarda karsilastirir.

Gruplama yalnizca gecmis gozlemlerle yapilir.
Gelecekteki gercek olcumler grup seciminde kullanilmaz.

Bu siniflar fiziksel supernova evreleri degildir.

Kullanim:
    python src/analyze_trend_regimes.py
"""

import numpy as np
import pandas as pd
import torch

from dataset import RESULTS, load_photometry
from repeat_experiments import CONFIG, prepare


SPLIT_SEED = 42
INIT_SEED = 42

# Gecmisteki dogrusal trendin normalize edilmis buyuklugu.
# Bu esik test sonucuna gore ayarlanmamistir.
TREND_THRESHOLD = 0.5


def classify_trend(x_normalized):
    """
    12 gecmis noktaya dogrusal egilim uydurur.

    Elde edilen egim, toplam 11 adimlik degisime
    donusturulur. Normalizasyon yalnizca gecmise
    dayandigi icin gelecek bilgisi kullanilmaz.
    """

    length = x_normalized.shape[1]

    t = np.arange(length, dtype=np.float64)
    t = t - t.mean()

    slopes = (x_normalized @ t) / np.dot(t, t)

    total_change = slopes * (length - 1)

    return np.where(
        total_change > TREND_THRESHOLD,
        "Artan",
        np.where(
            total_change < -TREND_THRESHOLD,
            "Azalan",
            "Yatay"
        )
    )


def load_neural_predictions(
    task,
    x_test,
    mu_test,
    scale_test
):
    """
    Tekrarli deneylerde saklanan seed=42 agirliklarini
    yukler. Model yeniden egitilmez.
    """

    checkpoint = (
        RESULTS
        / "repeat_runs"
        / "init"
        / f"{task}_split{SPLIT_SEED}_init{INIT_SEED}.pt"
    )

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)

    model = CONFIG[task][1]()

    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(state)
    model.eval()

    with torch.inference_mode():
        normalized_prediction = (
            model(x_test).cpu().numpy()
        )

    return (
        normalized_prediction * scale_test
        + mu_test
    )


def evaluate_horizon(df, horizon):
    """24 veya 48 saatlik test pencerelerini analiz eder."""

    hours = horizon * 6

    prepared = prepare(
        df,
        horizon,
        SPLIT_SEED
    )

    (
        x_normalized,
        _,
        raw_x,
        raw_y,
        names,
        parts,
        mu,
        scale
    ) = prepared

    test = parts == "test"

    history = raw_x[test].astype(np.float64)
    truth = raw_y[test].astype(np.float64)

    test_names = names[test]

    mu_test = mu[test]
    scale_test = scale[test]

    # Gecmise dayali normalize edilmis degerler.
    # Ridge egitimiyle ayni normalizasyon kullanilir.
    nx = (history - mu_test) / scale_test

    regimes = classify_trend(nx)

    predictions = {
        "Persistence": np.repeat(
            history[:, -1:],
            horizon,
            axis=1
        )
    }

    # Kaydedilmis Ridge modelini yukle.
    ridge_path = RESULTS / f"ridge_{hours}h.npz"

    with np.load(ridge_path) as ridge:
        w = ridge["weights"]
        b = ridge["bias"]

    predictions["Ridge"] = (
        (nx @ w + b) * scale_test + mu_test
    )

    # Mevcut sinir agi agirliklarini kullan.
    x_test = x_normalized[test]

    for model_name in ("GRU", "CNN"):

        task = f"{model_name.lower()}{hours}"

        predictions[model_name] = load_neural_predictions(
            task,
            x_test,
            mu_test,
            scale_test
        )

    # Tum tahminlerin dogru boyutta oldugunu denetle.
    for model_name, prediction in predictions.items():

        if (
            prediction.shape != truth.shape
            or not np.isfinite(prediction).all()
        ):
            raise RuntimeError(
                f"{model_name}: tahmin boyutu/verisi hatali."
            )

    # Eski sonuclarla pencere tutarliligini dogrula.
    expected = {
        24: (2393, 1.647022),
        48: (2113, 2.050194)
    }

    expected_n, expected_mae = expected[hours]

    actual_mae = np.abs(
        truth - predictions["Persistence"]
    ).mean()

    if (
        len(truth) != expected_n
        or not np.isclose(
            actual_mae,
            expected_mae,
            atol=1e-5,
            rtol=0
        )
    ):
        raise RuntimeError(
            "Test pencereleri onceki deneylerle eslesmiyor."
        )

    summary = []
    object_rows = []

    for regime in ("Artan", "Azalan", "Yatay"):

        mask = regimes == regime

        if not mask.any():
            print(f"{hours}h: {regime} grubu bos.")
            continue

        group_truth = truth[mask]
        group_names = test_names[mask]

        baseline_mae = float(
            np.mean(
                np.abs(
                    group_truth
                    - predictions["Persistence"][mask]
                )
            )
        )

        for model_name, prediction in predictions.items():

            group_prediction = prediction[mask]

            err = group_truth - group_prediction

            mae = float(np.mean(np.abs(err)))
            rmse = float(
                np.sqrt(np.mean(err ** 2))
            )

            improvement = (
                100.0 * (baseline_mae - mae)
                / baseline_mae
                if baseline_mae > 0
                else np.nan
            )

            # Her SN icin hata ayri hesaplanir.
            object_maes = []

            for name in np.unique(group_names):

                object_mask = group_names == name

                object_error = (
                    group_truth[object_mask]
                    - group_prediction[object_mask]
                )

                object_mae = float(
                    np.mean(np.abs(object_error))
                )

                object_maes.append(object_mae)

                object_rows.append({
                    "horizon_h": hours,
                    "regime": regime,
                    "Name": str(name),
                    "model": model_name,
                    "n_windows": int(object_mask.sum()),
                    "MAE_counts_per_s": object_mae
                })

            summary.append({
                "horizon_h": hours,
                "regime": regime,
                "model": model_name,
                "n_windows": int(mask.sum()),
                "n_supernovae": int(
                    len(np.unique(group_names))
                ),
                "MAE_counts_per_s": mae,
                "RMSE_counts_per_s": rmse,
                "object_mean_MAE": float(
                    np.mean(object_maes)
                ),
                "improvement_vs_persistence_pct": improvement
            })

    output_dir = RESULTS / "trend_regimes"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.DataFrame(summary)

    summary_df.to_csv(
        output_dir / f"summary_{hours}h.csv",
        index=False
    )

    pd.DataFrame(object_rows).to_csv(
        output_dir / f"objects_{hours}h.csv",
        index=False
    )

    print(f"\n===== {hours} SAAT - EGILIM ANALIZI =====")

    print(
        summary_df[
            [
                "regime",
                "model",
                "n_windows",
                "n_supernovae",
                "MAE_counts_per_s",
                "object_mean_MAE"
            ]
        ].round(4).to_string(index=False)
    )

    return summary_df


def main():

    torch.set_num_threads(1)

    df = load_photometry()

    summaries = [
        evaluate_horizon(df, horizon)
        for horizon in (4, 8)
    ]

    output_dir = RESULTS / "trend_regimes"

    pd.concat(
        summaries,
        ignore_index=True
    ).to_csv(
        output_dir / "summary_all.csv",
        index=False
    )

    print(
        "\nAnaliz tamamlandi. "
        "Ciktilar: results/trend_regimes/"
    )


if __name__ == "__main__":
    main()
