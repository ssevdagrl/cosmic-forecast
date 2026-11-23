
"""
Cosmic Forecast - Kalite Gruplari Bootstrap Analizi

Amac:
    Olcum belirsizligine gore olusturulan gruplarda
    modeller arasindaki nesne bazli MAE farklarini
    eslestirilmis bootstrap ile degerlendirmek.

Kullanilan dosya:
    results/quality_analysis/quality_objects_all.csv

Yeni model egitilmez.
Mevcut agirlik dosyalari degistirilmez.

Bootstrap birimi: Supernova kimligi.
Tekrar sayisi: 5000.

Kullanim:
    python src/analyze_quality_bootstrap.py
"""

import numpy as np
import pandas as pd

from dataset import RESULTS


BOOTSTRAP_REPEATS = 5000
SEED = 2026

MODELS = ("Persistence", "Ridge", "GRU", "CNN")

# Pozitif fark, ikinci model lehine bir sonuc demektir.
COMPARISONS = (
    ("Persistence", "GRU"),
    ("Persistence", "CNN"),
    ("Ridge", "GRU"),
    ("Ridge", "CNN"),
    ("GRU", "CNN"),
)


def paired_bootstrap(differences, rng):
    """Supernova bazli eslestirilmis bootstrap."""

    differences = np.asarray(
        differences,
        dtype=np.float64
    )

    if len(differences) < 2:
        raise ValueError(
            "Bootstrap icin yeterli sayida nesne yok."
        )

    n = len(differences)

    # Nesneleri yerine koyarak yeniden ornekle.
    indices = rng.integers(
        low=0,
        high=n,
        size=(BOOTSTRAP_REPEATS, n)
    )

    simulated_means = (
        differences[indices].mean(axis=1)
    )

    lower, upper = np.quantile(
        simulated_means,
        [0.025, 0.975]
    )

    return (
        float(differences.mean()),
        float(lower),
        float(upper)
    )


def validate_group(group):
    """Ayni nesnenin modeller arasinda eslestigini denetle."""

    if group.duplicated(
        ["Name", "model"]
    ).any():
        raise ValueError(
            "Ayni supernova ve model icin tekrarlanan kayit var."
        )

    errors = group.pivot(
        index="Name",
        columns="model",
        values="MAE_counts_per_s"
    )

    counts = group.pivot(
        index="Name",
        columns="model",
        values="n_windows"
    )

    missing = set(MODELS) - set(errors.columns)

    if missing:
        raise ValueError(
            f"Eksik model sonuclari: {missing}"
        )

    errors = errors.loc[:, MODELS]
    counts = counts.loc[:, MODELS]

    if errors.isna().any().any():
        raise ValueError(
            "Modellerin test ettigi SN kimlikleri eslesmiyor."
        )

    if not counts.eq(
        counts.iloc[:, 0],
        axis=0
    ).all().all():
        raise ValueError(
            "Ayni SN icin modellerin pencere sayilari farkli."
        )

    if not np.isfinite(
        errors.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Gecersiz MAE degerleri bulundu."
        )

    return errors


def main():

    source = (
        RESULTS
        / "quality_analysis"
        / "quality_objects_all.csv"
    )

    if not source.is_file():
        raise FileNotFoundError(
            f"Sonuc dosyasi bulunamadi: {source}"
        )

    data = pd.read_csv(
        source,
        dtype={"Name": "string"}
    )

    required = {
        "horizon_h",
        "quality_group",
        "Name",
        "model",
        "n_windows",
        "MAE_counts_per_s",
    }

    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            f"Eksik CSV sutunlari: {missing}"
        )

    rng = np.random.default_rng(SEED)

    results = []

    for (hours, quality_group), group in data.groupby(
        ["horizon_h", "quality_group"],
        sort=True
    ):

        errors = validate_group(group)

        n_objects = len(errors)

        if n_objects < 5:
            print(
                f"Uyari: {hours}h {quality_group} "
                "grubunda nesne sayisi cok az."
            )

        for first, second in COMPARISONS:

            first_values = errors[
                first
            ].to_numpy(dtype=np.float64)

            second_values = errors[
                second
            ].to_numpy(dtype=np.float64)

            # Pozitif fark = ikinci modelin hatasi daha dusuk.
            differences = (
                first_values - second_values
            )

            mean_diff, ci_low, ci_high = (
                paired_bootstrap(
                    differences,
                    rng
                )
            )

            result = {
                "horizon_h": int(hours),
                "quality_group": quality_group,
                "comparison": (
                    f"{first} - {second}"
                ),
                "n_supernovae": n_objects,
                "mean_MAE_difference": mean_diff,
                "CI95_low": ci_low,
                "CI95_high": ci_high,
                "second_model_better_SN": int(
                    np.sum(differences > 0)
                ),
                "first_model_better_SN": int(
                    np.sum(differences < 0)
                ),
                "ties": int(
                    np.sum(differences == 0)
                ),
                "CI_excludes_zero": bool(
                    ci_low > 0 or ci_high < 0
                ),
            }

            results.append(result)

    output = pd.DataFrame(results)

    output_dir = RESULTS / "quality_analysis"

    output.to_csv(
        output_dir / "quality_bootstrap.csv",
        index=False,
        encoding="utf-8-sig"
    )

    for hours in sorted(output["horizon_h"].unique()):

        subset = output[
            output["horizon_h"] == hours
        ]

        print(
            f"\n===== {hours} SAAT - BOOTSTRAP ====="
        )

        display_columns = [
            "quality_group",
            "comparison",
            "n_supernovae",
            "mean_MAE_difference",
            "CI95_low",
            "CI95_high",
            "second_model_better_SN",
            "first_model_better_SN",
        ]

        print(
            subset[display_columns]
            .round(5)
            .to_string(index=False)
        )

    print(
        "\nPozitif MAE farki ikinci model lehinedir."
    )

    print(
        "Negatif MAE farki birinci model lehinedir."
    )

    print(
        "\nNot: Guven araliklari nesne bazli ve "
        "kesifsel analiz icindir; coklu karsilastirma "
        "duzeltmesi uygulanmamistir."
    )

    print(
        "Kalite gruplari arasindaki SN kimlikleri "
        "kesisebilir; gruplar bagimsiz degildir."
    )

    print(
        f"\nKaydedildi: "
        f"{output_dir / 'quality_bootstrap.csv'}"
    )


if __name__ == "__main__":
    main()
