
"""
Cosmic Forecast - Baseline Tahmin Modeli

Bu dosyada persistence (son değeri koruma) yöntemi kullanılarak
24 saatlik ışık akısı tahmini gerçekleştirilir.

Yöntem, son gözlenen parlaklık değerinin gelecek zaman
adımlarında değişmeyeceğini varsayar.

Elde edilen sonuçlar, GRU ve CNN modellerinin performanslarını
karşılaştırmak için referans olarak kullanılır.
"""

import numpy as np
import pandas as pd

from dataset import RESULTS, load_photometry, make_windows, split_by_object


def main():
    # Modelin kullanacağı geçmiş ve gelecek zaman adımı sayıları.
    # Her adım 6 saati temsil eder.
    lookback = 12  # 72 saatlik geçmiş gözlem
    horizon = 4    # 24 saatlik tahmin

    # NASA TESS fotometrik ölçümlerini yükle.
    df = load_photometry()

    # Işık eğrilerini giriş ve hedef pencerelerine dönüştür.
    # Her pencere, geçmiş 72 saatten sonraki 24 saati tahmin eder.
    x, y, objects, _ = make_windows(
        df,
        lookback=lookback,
        horizon=horizon
    )

    # Verileri süpernova kimliğine göre eğitim, doğrulama ve
    # test kümelerine ayırarak veri sızıntısını önle.
    split = split_by_object(objects)

    rows = []

    # Baseline yöntemini her veri kümesinde ayrı değerlendir.
    for part in ["train", "val", "test"]:
        sel = split == part

        # Gerçek gelecek ışık akısı değerlerini seç.
        truth = y[sel]

        # Persistence yöntemi:
        # Son gözlenen değeri gelecek dört zaman adımına kopyala.
        predict = np.repeat(
            x[sel, -1:],
            horizon,
            axis=1
        )

        # Gerçek değerler ile tahminler arasındaki fark.
        err = truth - predict

        # MAE: Ortalama mutlak hata.
        mae = float(np.mean(np.abs(err)))

        # RMSE: Karekök ortalama karesel hata.
        # Büyük tahmin hatalarını MAE'ye göre daha fazla cezalandırır.
        rmse = float(np.sqrt(np.mean(err**2)))

        # Her veri kümesinin performans sonuçlarını kaydet.
        rows.append({
            "split": part,
            "n_windows": int(sel.sum()),
            "n_supernova": len(set(objects[sel])),
            "MAE_ct_per_sec": mae,
            "RMSE_ct_per_sec": rmse
        })

    # Sonuçları tabloya dönüştür.
    out = pd.DataFrame(rows)

    # Sonuç klasörünü oluştur ve metrikleri CSV olarak kaydet.
    RESULTS.mkdir(exist_ok=True, parents=True)

    out.to_csv(
        RESULTS / "baseline_metrikler.csv",
        index=False
    )

    # Hesaplanan performans sonuçlarını terminalde göster.
    print(out.to_string(index=False))

    print(
        "\nBu sadece son-deger (persistence) karsilastirma "
        "modeli; sinir agi sonucu degildir."
    )


if __name__ == "__main__":
    main()
