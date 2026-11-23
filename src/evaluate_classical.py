
"""
Cosmic Forecast - Klasik Tahmin Yöntemlerinin Değerlendirilmesi

Bu dosya, sinir ağı modellerinin başarısını değerlendirmek için
dört ek tahmin yöntemi oluşturur:

1. Historical Mean
2. Historical Median
3. Linear Trend
4. Ridge Regression

Sonuçlar, mevcut Persistence yöntemiyle birlikte raporlanır.

Veri ayrımı mevcut deneylerle aynıdır (seed=42).
Ridge modeli yalnızca eğitim verileriyle öğrenir.
Hiperparametre seçimi doğrulama verileriyle yapılır.

Kullanım:
    python src/evaluate_classical.py
"""

import json

import numpy as np
import pandas as pd

from dataset import (
    RESULTS,
    load_photometry,
    make_windows,
    scaling_from_history,
    split_by_object,
)


SEED = 42
LOOKBACK = 12

# Ridge düzenlileştirme parametresi adayları.
# En uygun değer yalnızca doğrulama kümesinde seçilir.
ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def historical_mean(x, horizon):
    """Geçmiş gözlemlerin ortalamasını gelecekte sabit tutar."""

    value = x.mean(axis=1, keepdims=True)

    return np.repeat(value, horizon, axis=1)


def historical_median(x, horizon):
    """Geçmiş gözlemlerin medyanını gelecekte sabit tutar."""

    value = np.median(x, axis=1, keepdims=True)

    return np.repeat(value, horizon, axis=1)


def linear_trend(x, horizon):
    """
    Her geçmiş pencereye doğrusal regresyon uygular.

    Geçmiş 12 zaman adımından eğim hesaplanır.
    Bu eğim gelecek 4 veya 8 adıma uzatılır.
    """

    length = x.shape[1]

    times = np.arange(length, dtype=np.float64)
    centered = times - times.mean()

    x64 = x.astype(np.float64)

    means = x64.mean(axis=1, keepdims=True)

    # En küçük kareler yöntemiyle eğimleri hesapla.
    slopes = (
        (x64 - means) @ centered
        / np.dot(centered, centered)
    )

    # Gelecekteki zaman noktaları.
    future = (
        np.arange(length, length + horizon)
        - times.mean()
    )

    return means + slopes[:, None] * future[None, :]


def fit_ridge(x, y, alpha):
    """
    Çok çıktılı Ridge Regression modelini eğitir.

    Model, 12 geçmiş gözlemden 4 veya 8 gelecek
    değeri doğrudan tahmin etmeyi öğrenir.

    L2 düzenlileştirmesi kullanılır.
    Sabit terim (bias) cezalandırılmaz.
    """

    x_mean = x.mean(axis=0, keepdims=True)
    y_mean = y.mean(axis=0, keepdims=True)

    xc = x - x_mean
    yc = y - y_mean

    eye = np.eye(x.shape[1], dtype=np.float64)

    # Ridge kapalı form çözümü.
    weights = np.linalg.solve(
        xc.T @ xc + alpha * eye,
        xc.T @ yc
    )

    bias = y_mean - x_mean @ weights

    return weights, bias


def scores(truth, prediction):
    """Tahminlerin MAE ve RMSE değerlerini hesaplar."""

    if (
        truth.shape != prediction.shape
        or not np.isfinite(prediction).all()
    ):
        raise ValueError(
            "Tahminlerde boyut veya sonlu sayi hatasi var."
        )

    error = truth - prediction

    mae = float(np.mean(np.abs(error)))
    rmse = float(np.sqrt(np.mean(error ** 2)))

    return mae, rmse


def evaluate_horizon(df, horizon):
    """Belirli tahmin süresindeki bütün yöntemleri değerlendirir."""

    hours = horizon * 6

    # Mevcut GRU ve CNN deneyleriyle aynı pencereleri oluştur.
    x, y, names, _ = make_windows(
        df,
        lookback=LOOKBACK,
        horizon=horizon
    )

    # Süpernova kimliği bazında aynı veri ayrımını kullan.
    split = split_by_object(names, seed=SEED)

    train, val, test = (
        split == part
        for part in ("train", "val", "test")
    )

    # Veri kümelerinde ortak süpernova bulunmamalıdır.
    groups = [
        set(names[split == part])
        for part in ("train", "val", "test")
    ]

    if any(
        groups[i] & groups[j]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        raise RuntimeError(
            "Supernova kimligine dayali veri sizintisi bulundu."
        )

    # Normalizasyon yalnızca geçmiş gözlemlerden hesaplanır.
    mu, scale = scaling_from_history(x)

    nx = ((x - mu) / scale).astype(np.float64)
    ny = ((y - mu) / scale).astype(np.float64)

    # --------------------------------------------------
    # Ridge modelinin eğitimi ve hiperparametre seçimi
    # --------------------------------------------------

    best = None

    for alpha in ALPHAS:

        # Ağırlıklar yalnızca train kümesinden öğrenilir.
        w, b = fit_ridge(
            nx[train],
            ny[train],
            alpha
        )

        # Alpha seçimi yalnızca validation ile yapılır.
        val_prediction = nx[val] @ w + b

        val_mse = float(
            np.mean((ny[val] - val_prediction) ** 2)
        )

        if best is None or val_mse < best[0]:
            best = (val_mse, alpha, w, b)

    best_val_mse, best_alpha, weights, bias = best

    # Ridge test tahminini özgün counts/s birimine döndür.
    ridge_test = (
        (nx[test] @ weights + bias)
        * scale[test]
        + mu[test]
    )

    test_x = x[test]
    truth = y[test]
    test_names = names[test]

    # --------------------------------------------------
    # Beş klasik yöntemin tahminlerini üret
    # --------------------------------------------------

    predictions = {
        "Baseline": np.repeat(
            test_x[:, -1:],
            horizon,
            axis=1
        ),
        "Mean": historical_mean(test_x, horizon),
        "Median": historical_median(test_x, horizon),
        "LinearTrend": linear_trend(test_x, horizon),
        "Ridge": ridge_test,
    }

    summary = []
    per_object = []

    # --------------------------------------------------
    # Performans metrikleri
    # --------------------------------------------------

    for model, prediction in predictions.items():

        mae, rmse = scores(truth, prediction)

        summary.append({
            "forecast_h": hours,
            "model": model,
            "n_test_supernova": len(groups[2]),
            "n_test_windows": len(truth),
            "MAE_counts_per_s": mae,
            "RMSE_counts_per_s": rmse,
        })

        # Nesne bazlı karşılaştırma için her süpernovanın
        # hata değerlerini ayrı kaydet.
        for name in np.unique(test_names):

            mask = test_names == name

            obj_mae, obj_rmse = scores(
                truth[mask],
                prediction[mask]
            )

            per_object.append({
                "Name": name,
                "model": model,
                "n_windows": int(mask.sum()),
                "MAE_counts_per_s": obj_mae,
                "RMSE_counts_per_s": obj_rmse,
            })

    summary_df = pd.DataFrame(summary)

    # --------------------------------------------------
    # Sonuçları kaydet
    # --------------------------------------------------

    summary_df.to_csv(
        RESULTS / f"classical_summary_{hours}h.csv",
        index=False
    )

    pd.DataFrame(per_object).to_csv(
        RESULTS / f"classical_object_{hours}h.csv",
        index=False
    )

    # Eğitilmiş Ridge katsayılarını sakla.
    np.savez(
        RESULTS / f"ridge_{hours}h.npz",
        weights=weights,
        bias=bias,
        alpha=best_alpha
    )

    # Deney protokolünü ve seçilen alpha değerini kaydet.
    protocol = {
        "split_seed": SEED,
        "lookback_bins": LOOKBACK,
        "horizon_bins": horizon,
        "ridge_alpha_candidates": ALPHAS,
        "best_ridge_alpha": best_alpha,
        "ridge_validation_normalized_MSE": best_val_mse,
        "selection_rule": (
            "Ridge alpha selected on validation, never test"
        ),
    }

    (
        RESULTS / f"classical_protocol_{hours}h.json"
    ).write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # Önceki Baseline sonucu ile tutarlılık kontrolü
    # --------------------------------------------------

    old_summary = RESULTS / "model_summary_all.csv"

    if old_summary.is_file():

        previous = pd.read_csv(old_summary)

        old = previous.loc[
            (previous["forecast_h"] == hours)
            & (previous["model"] == "Baseline")
        ]

        if (
            len(old) != 1
            or not np.isclose(
                float(old.iloc[0]["MAE_counts_per_s"]),
                summary[0]["MAE_counts_per_s"],
                rtol=0,
                atol=1e-6
            )
        ):
            raise RuntimeError(
                f"{hours}h: Eski Baseline MAE ile uyumsuzluk var."
            )

        print(
            f"{hours}h: eski Baseline ile karsilastirma basarili."
        )

    # --------------------------------------------------
    # Terminal çıktıları
    # --------------------------------------------------

    print(
        f"\n===== {hours} saat | "
        f"{len(groups[2])} test SN, "
        f"{len(truth)} pencere ====="
    )

    print(
        summary_df[
            ["model", "MAE_counts_per_s", "RMSE_counts_per_s"]
        ].to_string(index=False)
    )

    print(
        f"Ridge alpha={best_alpha}; "
        f"validation normalized MSE={best_val_mse:.6f}"
    )

    return summary_df


def main():
    RESULTS.mkdir(exist_ok=True, parents=True)

    data = load_photometry()

    frames = [
        evaluate_horizon(data, horizon)
        for horizon in (4, 8)
    ]

    pd.concat(
        frames,
        ignore_index=True
    ).to_csv(
        RESULTS / "classical_summary_all.csv",
        index=False
    )

    print(
        "\nKaydedildi: results/classical_summary_all.csv "
        "ve ufuk bazli cikti dosyalari."
    )


if __name__ == "__main__":
    main()
