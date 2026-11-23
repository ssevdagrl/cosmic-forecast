
"""
Cosmic Forecast - Fotometrik Olcum Kalitesi Incelemesi

TESS veri tablosunun:
- Sutunlarini
- Veri turlerini
- CRate ve e_CRate dagilimlarini
- Eksik ve gecersiz olcumleri
- Supernova bazinda belirsizlik farklarini

inceler.

Veriyi degistirmez.
Model egitmez.
Mevcut deney sonuclarini etkilemez.
"""

import numpy as np
import pandas as pd

from dataset import load_photometry


def describe_values(values, label):

    numeric = pd.to_numeric(
        values,
        errors="coerce"
    )

    valid = numeric[np.isfinite(numeric)]

    print(f"\n--- {label} ---")

    print("Toplam kayit:", len(numeric))
    print("Gecerli sayisal kayit:", len(valid))

    if len(valid) == 0:
        print("Sayisal deger bulunamadi.")
        return

    print(
        valid.quantile(
            [0.00, 0.01, 0.10, 0.25,
             0.50, 0.75, 0.90, 0.99, 1.00]
        ).round(5).to_string()
    )


def main():

    df = load_photometry()

    print("\n=== VERI YAPISI ===")
    print("Tablo boyutu:", df.shape)
    print("Sutunlar:", df.columns.tolist())

    print("\nVeri turleri:")
    print(df.dtypes.to_string())

    expected = [
        "Name",
        "BJD",
        "CRate",
        "e_CRate"
    ]

    missing = [
        column
        for column in expected
        if column not in df.columns
    ]

    if missing:
        print("\nEksik sutunlar:", missing)
        print(
            "Kalite analizi oncesinde "
            "dataset.py kontrol edilmeli."
        )
        return

    data = df[expected].copy()

    for column in ("BJD", "CRate", "e_CRate"):
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    print("\n=== KAYIT SAYILARI ===")
    print("Toplam satir:", len(data))
    print("Benzersiz SN:", data["Name"].nunique())

    print(
        "Eksik veya sayisal olmayan belirsizlik:",
        int(data["e_CRate"].isna().sum())
    )

    print(
        "Sifir veya negatif belirsizlik:",
        int((data["e_CRate"] <= 0).sum())
    )

    print(
        "Negatif CRate:",
        int((data["CRate"] < 0).sum())
    )

    describe_values(
        data["CRate"],
        "CRate (counts/s)"
    )

    describe_values(
        data["e_CRate"],
        "e_CRate (counts/s)"
    )

    # Yalnizca gecerli satirlar.
    valid = data.dropna(
        subset=["Name", "BJD", "CRate", "e_CRate"]
    ).copy()

    valid = valid[
        np.isfinite(valid["BJD"])
        & np.isfinite(valid["CRate"])
        & np.isfinite(valid["e_CRate"])
        & (valid["e_CRate"] > 0)
    ]

    print("\n=== SUPERNOVA BAZINDA BELIRSIZLIK ===")

    by_object = valid.groupby("Name").agg(
        n_measurements=("e_CRate", "size"),
        median_uncertainty=("e_CRate", "median")
    )

    print(
        "\nMedyan belirsizligi en dusuk 5 SN:"
    )

    print(
        by_object.sort_values(
            "median_uncertainty"
        ).head(5).round(5).to_string()
    )

    print(
        "\nMedyan belirsizligi en yuksek 5 SN:"
    )

    print(
        by_object.sort_values(
            "median_uncertainty",
            ascending=False
        ).head(5).round(5).to_string()
    )

    # Gecici kontrol: her nesne icin 6 saatlik
    # binlerin belirsizlik istatistikleri.
    # Bu adim henuz model pencerelerine baglanmaz.
    valid["bin6h"] = np.floor(
        4.0 * valid["BJD"].to_numpy()
        + 1e-7
    ).astype(np.int64)

    binned = valid.groupby(
        ["Name", "bin6h"],
        as_index=False
    ).agg(
        median_rate=("CRate", "median"),
        median_uncertainty=("e_CRate", "median"),
        n_measurements=("CRate", "size")
    )

    print("\n=== 6 SAATLIK BIN ISTATISTIKLERI ===")

    print("Bin sayisi:", len(binned))

    describe_values(
        binned["median_uncertainty"],
        "6 saatlik bin medyan belirsizligi"
    )

    print(
        "\nNot: Buradaki medyan belirsizlik, "
        "binlenmis akinin yayilmis hata tahmini degildir. "
        "Ilk kalite gostergesi olarak incelenmektedir."
    )


if __name__ == "__main__":
    main()
