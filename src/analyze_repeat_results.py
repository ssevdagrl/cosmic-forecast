
"""
Cosmic Forecast - Tekrarlı Deneylerin Istatistiksel Analizi

Mevcut 24 egitimin sonuclarini okur.
Model egitimi veya agirlik dosyasi degisikligi yapmaz.

Analizler:
- GRU ve CNN karsilastirmasi
- Pencere bazli MAE
- Supernova bazli eslestirilmis MAE farki
- 3000 tekrarlı object-level bootstrap
- Farkli seed deneylerinin ozetlenmesi
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPEATS = ROOT / "results" / "repeat_runs"

BOOTSTRAP_ITERATIONS = 3000


def bootstrap_object_difference(
    differences,
    seed
):
    """
    differences = GRU MAE - CNN MAE

    Pozitif:
        CNN daha iyi.

    Negatif:
        GRU daha iyi.

    Bootstrap, pencereleri degil
    supernova nesnelerini yeniden ornekler.
    """

    differences = np.asarray(
        differences,
        dtype=np.float64
    )

    if len(differences) < 2:
        raise ValueError(
            "Bootstrap icin en az iki nesne gerekli."
        )

    rng = np.random.default_rng(seed)

    samples = rng.choice(
        differences,
        size=(BOOTSTRAP_ITERATIONS, len(differences)),
        replace=True
    )

    bootstrap_means = samples.mean(axis=1)

    ci_low, ci_high = np.quantile(
        bootstrap_means,
        [0.025, 0.975]
    )

    return {
        "object_mean_difference": float(
            differences.mean()
        ),
        "CI95_low": float(ci_low),
        "CI95_high": float(ci_high),
        "CNN_better_objects": int(
            (differences > 0).sum()
        ),
        "GRU_better_objects": int(
            (differences < 0).sum()
        ),
        "ties": int(
            (differences == 0).sum()
        ),
        "n_objects": len(differences),
    }


def analyze_mode(mode):

    folder = REPEATS / mode

    metrics_path = folder / "metrics.csv"
    object_path = folder / "object_metrics.csv"

    metrics = pd.read_csv(metrics_path)
    objects = pd.read_csv(object_path)

    results = []

    keys = [
        "split_seed",
        "init_seed",
        "horizon_h"
    ]

    for key, group in objects.groupby(
        keys,
        sort=True
    ):

        split_seed, init_seed, horizon = (
            int(value)
            for value in key
        )

        gru_name = f"gru{horizon}"
        cnn_name = f"cnn{horizon}"

        # Her SN icin iki modelin
        # hatalarini yan yana getir.
        paired = group.pivot(
            index="Name",
            columns="task",
            values="model_MAE_counts_per_s"
        )

        paired = paired[
            [gru_name, cnn_name]
        ].dropna()

        # Her nesneye esit agirlik ver.
        differences = (
            paired[gru_name] - paired[cnn_name]
        ).to_numpy()

        stats = bootstrap_object_difference(
            differences,
            seed=split_seed * 1000
            + init_seed * 10
            + horizon
        )

        # Pencere bazli genel MAE'leri de al.
        selected = metrics.loc[
            (metrics["split_seed"] == split_seed)
            & (metrics["init_seed"] == init_seed)
            & (metrics["horizon_h"] == horizon)
        ].set_index("task")

        gru_mae = float(
            selected.loc[
                gru_name,
                "MAE_counts_per_s"
            ]
        )

        cnn_mae = float(
            selected.loc[
                cnn_name,
                "MAE_counts_per_s"
            ]
        )

        baseline_gru = float(
            selected.loc[
                gru_name,
                "baseline_MAE_counts_per_s"
            ]
        )

        baseline_cnn = float(
            selected.loc[
                cnn_name,
                "baseline_MAE_counts_per_s"
            ]
        )

        if not np.isclose(
            baseline_gru,
            baseline_cnn,
            atol=1e-9,
            rtol=0
        ):
            raise RuntimeError(
                "Modeller farkli test pencerelerinde!"
            )

        record = {
            "mode": mode,
            "split_seed": split_seed,
            "init_seed": init_seed,
            "horizon_h": horizon,
            "GRU_MAE": gru_mae,
            "CNN_MAE": cnn_mae,
            "Baseline_MAE": baseline_gru,
            "window_MAE_difference": (
                gru_mae - cnn_mae
            ),
            **stats
        }

        results.append(record)

    return pd.DataFrame(results)


def main():

    frames = [
        analyze_mode("init"),
        analyze_mode("split")
    ]

    comparison = pd.concat(
        frames,
        ignore_index=True
    )

    output = (
        REPEATS
        / "paired_model_comparison.csv"
    )

    comparison.to_csv(
        output,
        index=False
    )

    print("\n=== ESLESTIRILMIS MODEL KARSILASTIRMASI ===")

    columns = [
        "mode",
        "split_seed",
        "init_seed",
        "horizon_h",
        "window_MAE_difference",
        "object_mean_difference",
        "CI95_low",
        "CI95_high",
        "CNN_better_objects",
        "GRU_better_objects",
        "n_objects",
    ]

    print(
        comparison[columns].round(5).to_string(
            index=False
        )
    )

    print("\n=== TEKRARLI DENEY OZETI ===")

    summary = comparison.groupby(
        ["mode", "horizon_h"]
    ).agg(
        GRU_mean=("GRU_MAE", "mean"),
        GRU_std=("GRU_MAE", "std"),
        CNN_mean=("CNN_MAE", "mean"),
        CNN_std=("CNN_MAE", "std"),
        Baseline_mean=("Baseline_MAE", "mean"),
    )

    print(
        summary.round(5).to_string()
    )

    summary.to_csv(
        REPEATS / "experiment_summary.csv"
    )

    print(
        "\nPozitif GRU-CNN farki: CNN daha basarili."
    )

    print(
        "Negatif GRU-CNN farki: GRU daha basarili."
    )

    print(
        "\nNot: Bootstrap araliklari tek tek deney "
        "kosullarina baglidir. Farkli veri bolmeleri "
        "ve egitimler bagimsizmis gibi "
        "birlikte bootstrap edilmemistir."
    )

    print(f"\nKaydedildi: {output}")


if __name__ == "__main__":
    main()
