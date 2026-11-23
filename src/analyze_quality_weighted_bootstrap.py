
"""
Cosmic Forecast - Pencere Agirlikli Cluster Bootstrap

Amac:
    Kalite gruplarindaki pencere bazli MAE farklari icin
    supernova kimligine dayali bootstrap guven
    araliklari hesaplamak.

Her bootstrap tekrarinda:
    1. Supernovalar yerine koyarak secilir.
    2. Secilen SN'nin tum pencereleri korunur.
    3. Pencere sayisiyla agirlikli MAE farki hesaplanir.

Bu analiz onceki nesneye esit agirlikli
bootstrap sonuclarinin yerine gecmez;
onlari tamamlar.

Kullanim:
    python src/analyze_quality_weighted_bootstrap.py
"""

import numpy as np
import pandas as pd

from dataset import RESULTS


N_BOOTSTRAP = 5000
RANDOM_SEED = 2026

MODELS = (
    "Persistence",
    "Ridge",
    "GRU",
    "CNN",
)

COMPARISONS = (
    ("Persistence", "GRU"),
    ("Persistence", "CNN"),
    ("Ridge", "GRU"),
    ("Ridge", "CNN"),
    ("GRU", "CNN"),
)


def prepare_group(group):
    """Modelleri ayni supernovalar uzerinde eslestir."""

    if group.duplicated(["Name", "model"]).any():
        raise ValueError(
            "Tekrarlanan SN-model kaydi bulundu."
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
            f"Eksik modeller: {missing}"
        )

    errors = errors.loc[:, MODELS]
    counts = counts.loc[errors.index, MODELS]

    if errors.isna().any().any():
        raise ValueError(
            "Modellerin SN kimlikleri eslesmiyor."
        )

    if not counts.eq(
        counts["Persistence"],
        axis=0
    ).all().all():
        raise ValueError(
            "Modellerin SN pencere sayilari eslesmiyor."
        )

    weights = counts["Persistence"].to_numpy(
        dtype=np.int64
    )

    if np.any(weights <= 0):
        raise ValueError(
            "Gecersiz pencere sayisi."
        )

    if not np.isfinite(
        errors.to_numpy(dtype=float)
    ).all():
        raise ValueError(
            "Gecersiz MAE degeri."
        )

    return errors, weights


def verify_against_summary(
    errors,
    weights,
    summary,
    hours,
    quality_group
):
    """
    Nesne hatalarinin pencere agirlikli ortalamasi,
    onceki ana kalite MAE tablosuyla eslesmeli.
    """

    for model in MODELS:

        selected = summary.loc[
            (summary["horizon_h"] == hours)
            & (summary["quality_group"] == quality_group)
            & (summary["model"] == model)
        ]

        if len(selected) != 1:
            raise RuntimeError(
                "Ana kalite tablosunda kayit eksik."
            )

        expected = float(
            selected.iloc[0]["MAE_counts_per_s"]
        )

        expected_n = int(
            selected.iloc[0]["n_windows"]
        )

        actual = float(
            np.average(
                errors[model].to_numpy(dtype=float),
                weights=weights
            )
        )

        if (
            not np.isclose(
                actual,
                expected,
                atol=1e-5,
                rtol=0
            )
            or weights.sum() != expected_n
        ):
            raise RuntimeError(
                f"{hours}h {quality_group} {model}: "
                "ana MAE ile eslesmiyor."
            )


def analyze_group(
    errors,
    weights,
    hours,
    quality_group,
    rng
):
    """Butun model ciftleri icin cluster bootstrap uygula."""

    n_objects = len(errors)

    # Her tekrar supernovalari yeniden ornekler.
    sampled_indices = rng.integers(
        0,
        n_objects,
        size=(N_BOOTSTRAP, n_objects)
    )

    # Bir SN yeniden secildiginde butun pencereleri
    # birlikte temsil edilir.
    sampled_weights = weights[
        sampled_indices
    ].astype(np.float64)

    denominators = sampled_weights.sum(axis=1)

    records = []

    for first, second in COMPARISONS:

        first_errors = errors[first].to_numpy(
            dtype=np.float64
        )

        second_errors = errors[second].to_numpy(
            dtype=np.float64
        )

        # Pozitif: ikinci model daha dusuk MAE uretiyor.
        differences = (
            first_errors - second_errors
        )

        observed = float(
            np.average(
                differences,
                weights=weights
            )
        )

        bootstrap_differences = (
            differences[sampled_indices]
            * sampled_weights
        ).sum(axis=1) / denominators

        ci_low, ci_high = np.quantile(
            bootstrap_differences,
            [0.025, 0.975]
        )

        first_mae = float(
            np.average(
                first_errors,
                weights=weights
            )
        )

        improvement_pct = (
            100.0 * observed / first_mae
            if first_mae > 0
            else np.nan
        )

        records.append({
            "horizon_h": int(hours),
            "quality_group": quality_group,
            "comparison": f"{first} - {second}",
            "n_supernovae": n_objects,
            "n_windows": int(weights.sum()),
            "window_weighted_MAE_difference": observed,
            "CI95_low": float(ci_low),
            "CI95_high": float(ci_high),
            "improvement_pct": improvement_pct,
            "second_model_better_SN": int(
                np.sum(differences > 0)
            ),
            "first_model_better_SN": int(
                np.sum(differences < 0)
            ),
            "CI_excludes_zero": bool(
                ci_low > 0 or ci_high < 0
            )
        })

    return records


def main():

    folder = RESULTS / "quality_analysis"

    objects = pd.read_csv(
        folder / "quality_objects_all.csv",
        dtype={"Name": "string"}
    )

    summary = pd.read_csv(
        folder / "quality_summary_all.csv"
    )

    rng = np.random.default_rng(RANDOM_SEED)

    all_results = []

    for (hours, quality_group), group in objects.groupby(
        ["horizon_h", "quality_group"],
        sort=True
    ):

        errors, weights = prepare_group(group)

        verify_against_summary(
            errors,
            weights,
            summary,
            hours,
            quality_group
        )

        records = analyze_group(
            errors,
            weights,
            hours,
            quality_group,
            rng
        )

        all_results.extend(records)

    result = pd.DataFrame(all_results)

    output_path = (
        folder / "quality_weighted_bootstrap.csv"
    )

    result.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig"
    )

    for hours in sorted(result["horizon_h"].unique()):

        subset = result[
            result["horizon_h"] == hours
        ]

        print(
            f"\n===== {hours} SAAT - "
            "PENCERE AGIRLIKLI BOOTSTRAP ====="
        )

        cols = [
            "quality_group",
            "comparison",
            "n_supernovae",
            "window_weighted_MAE_difference",
            "CI95_low",
            "CI95_high",
            "improvement_pct",
        ]

        print(
            subset[cols]
            .round(5)
            .to_string(index=False)
        )

    print(
        "\nPozitif fark: ikinci model daha iyi."
    )

    print(
        "Bu bootstrap supernova kimligine gore "
        "kume yeniden orneklemesi kullanir."
    )

    print(
        "Guven araliklari kesifseldir; "
        "coklu karsilastirma duzeltmesi yapilmamistir."
    )

    print(
        "\nKaydedildi:",
        output_path
    )


if __name__ == "__main__":
    main()
