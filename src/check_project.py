"""Cosmic Forecast - Veri ve sonuç bütünlüğünü kontrol et.

Önce modeller eğitilmeli ve evaluate_all.py çalıştırılmalıdır.
Kullanım: python src/check_project.py
"""

import sys

import numpy as np
import pandas as pd

from dataset import RESULTS, load_photometry, make_windows, split_by_object


# Eksik ya da boş olmaması gereken model ve çıktı dosyaları.
REQUIRED = [
    "gru_model.pt",
    "cnn_24h_model.pt",
    "gru_48h_model.pt",
    "cnn_48h_model.pt",
    "model_summary_all.csv",
    "model_horizon_24h.csv",
    "model_horizon_48h.csv",
    "model_object_24h.csv",
    "model_object_48h.csv",
    "model_bootstrap_24h.json",
    "model_bootstrap_48h.json",
    "model_horizon_24h.png",
    "model_horizon_48h.png",
    "model_example_24h_1.png",
    "model_example_48h_1.png",
]


def validate_summary(path, errors):
    """Altı beklenen deney satırını, sütunları ve sayısal metrikleri doğrula."""
    if not path.is_file() or path.stat().st_size == 0:
        return

    try:
        frame = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        errors.append(f"Sonuç tablosu okunamadı: {exc}")
        return

    required_columns = {
        "forecast_h", "model", "n_test_supernova", "n_test_windows",
        "MAE_counts_per_s", "RMSE_counts_per_s",
    }
    missing = required_columns - set(frame.columns)
    if missing:
        errors.append(f"Sonuç tablosunda eksik sütunlar: {sorted(missing)}")
        return

    # Küme karşılaştırması tek başına tekrar eden satırları yakalamaz.
    expected = {(hours, model) for hours in (24, 48)
                for model in ("Baseline", "GRU", "CNN")}
    pairs = list(zip(frame["forecast_h"], frame["model"]))
    if len(frame) != len(expected) or set(pairs) != expected:
        errors.append(f"Beklenen altı benzersiz satır bulunamadı: {pairs}")

    numeric = [
        "MAE_counts_per_s", "RMSE_counts_per_s",
        "n_test_supernova", "n_test_windows",
    ]
    values = frame[numeric].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        errors.append("Sonuç tablosunda geçersiz sayısal değer var")
    elif (values[["MAE_counts_per_s", "RMSE_counts_per_s"]] < 0).any().any():
        errors.append("MAE veya RMSE negatif olamaz")
    elif (values[["n_test_supernova", "n_test_windows"]] <= 0).any().any():
        errors.append("Test süpernova veya pencere sayısı sıfır olamaz")

    print("\nFinal comparison:\n", frame.to_string(index=False))


def main():
    errors = []

    # 1. Fotometrik veri okuma kontrolü.
    try:
        df = load_photometry()
        print(f"Data: {len(df):,} measurements, {df.Name.nunique()} supernovae")
    except Exception as exc:
        print("DATA ERROR:", exc)
        sys.exit(1)

    # 2. Her tahmin süresinde kimlik ayrımı ve sonlu veri kontrolü.
    for horizon in (4, 8):
        x, y, names, _ = make_windows(df, lookback=12, horizon=horizon)
        split = split_by_object(names, seed=42)
        groups = [set(names[split == part]) for part in ("train", "val", "test")]
        identity_disjoint = all(
            not (groups[i] & groups[j])
            for i in range(3) for j in range(i + 1, 3)
        )
        print(
            f"{horizon * 6}h: windows={len(x)}, "
            f"test objects={len(groups[2])}; identity disjoint={identity_disjoint}"
        )
        if not identity_disjoint:
            errors.append(f"{horizon * 6}h split leakage")
        if any(not group for group in groups):
            errors.append(f"{horizon * 6}h boş eğitim/doğrulama/test grubu")
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            errors.append(f"{horizon * 6}h geçersiz sayısal değer içeriyor")

    # 3. Beklenen dosyalar mevcut ve boş değil mi?
    for name in REQUIRED:
        path = RESULTS / name
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"Output missing or empty: results/{name}")

    # 4. Sonuç tablosunda altı benzersiz deney ve geçerli metrikler var mı?
    validate_summary(RESULTS / "model_summary_all.csv", errors)

    print("\nCHECK:", "PASS" if not errors else "FAIL")
    for issue in errors:
        print(" -", issue)
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
