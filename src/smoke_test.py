
"""
Cosmic Forecast - Temel Yazılım Testleri

Bu dosya, gerçek NASA TESS verilerine veya internet
bağlantısına ihtiyaç duymadan projenin temel
bileşenlerini test eder.

Gerçekleştirilen kontroller:
    1. Yapay süpernova zaman serilerinin oluşturulması
    2. Geçmiş gözlem ve tahmin pencerelerinin hazırlanması
    3. Eğitim, doğrulama ve test kümelerinin bağımsızlığı
    4. Geçmiş gözlemlere dayalı normalizasyon
    5. CNN ve GRU modellerinin giriş-çıkış boyutları
    6. Kayıp hesaplaması ve geri yayılım işlemi

Bu test modelin tahmin başarısını ölçmez.
Yalnızca temel yazılım bileşenlerinin çalışmasını kontrol eder.

Kullanım:
    python src/smoke_test.py
"""

import numpy as np
import pandas as pd
import torch

from dataset import (
    make_windows,
    scaling_from_history,
    split_by_object,
)
from models_extra import CNNForecast, GRUForecastFlexible


def main():
    # --------------------------------------------------
    # 1. Yapay süpernova verilerinin oluşturulması
    # --------------------------------------------------

    rows = []

    # Test için 18 farklı yapay süpernova oluştur.
    for obj in range(18):

        # Her süpernova için 70 adet gözlem üret.
        for k in range(70):

            rows.append({
                "Name": f"TEST_{obj:02}",

                # Gözlemler 6 saatlik aralıklarla oluşturulur.
                # Süpernovaların zaman aralıkları birbirinden
                # ayrılarak farklı nesneler temsil edilir.
                "BJD": 2458000.0 + obj * 100 + k / 4,

                # Sinüs fonksiyonuyla zamana bağlı değişen
                # yapay ışık akısı değerleri oluşturulur.
                "CRate": float(
                    10 + obj + 2 * np.sin(k / 8)
                ),

                # Her ölçüm için pozitif belirsizlik değeri.
                "e_CRate": 0.1
            })

    # Oluşturulan yapay gözlemleri tabloya dönüştür.
    df = pd.DataFrame(rows)

    # --------------------------------------------------
    # 2. Tahmin pencerelerinin test edilmesi
    # --------------------------------------------------

    # Her zaman adımı 6 saati temsil eder.
    #
    # horizon=4 -> 24 saatlik tahmin
    # horizon=8 -> 48 saatlik tahmin
    for horizon in (4, 8):

        # Geçmiş 72 saatlik gözlemden gelecek
        # zaman adımlarına ait tahmin pencereleri oluştur.
        x, y, names, _ = make_windows(
            df,
            lookback=12,
            horizon=horizon,
            max_per_object=60
        )

        # --------------------------------------------------
        # 3. Veri kümelerinin bağımsızlığını kontrol et
        # --------------------------------------------------

        # Verileri süpernova kimliklerine göre ayır.
        split = split_by_object(
            names,
            seed=42
        )

        # Her veri kümesindeki benzersiz
        # süpernova kimliklerini elde et.
        sets = [
            set(names[split == part])
            for part in ("train", "val", "test")
        ]

        # Aynı süpernovanın farklı veri kümelerinde
        # bulunmadığını doğrula.
        # Böylece kimlik bazlı veri sızıntısı önlenir.
        assert not (
            sets[0] & sets[1]
            or sets[0] & sets[2]
            or sets[1] & sets[2]
        )

        # --------------------------------------------------
        # 4. Normalizasyonun kontrol edilmesi
        # --------------------------------------------------

        # Normalizasyon parametrelerini yalnızca
        # geçmiş gözlemlerden hesapla.
        mu, sd = scaling_from_history(x)

        # Normalize edilmiş değerlerin NaN veya
        # sonsuz sayı içermediğini kontrol et.
        assert np.isfinite(
            (x - mu) / sd
        ).all()

        # --------------------------------------------------
        # 5. Model giriş ve hedeflerinin hazırlanması
        # --------------------------------------------------

        # Testin hızlı çalışması için ilk sekiz
        # zaman penceresi kullanılır.
        #
        # Modellerin beklediği giriş boyutu:
        # (örnek sayısı, zaman adımı, özellik sayısı)
        xb = torch.tensor(
            (
                (x[:8] - mu[:8]) / sd[:8]
            ).astype("float32")
        ).unsqueeze(-1)

        # Gelecek hedef değerlerini aynı geçmiş
        # pencerenin normalizasyon parametreleriyle ölçekle.
        yt = torch.tensor(
            (
                (y[:8] - mu[:8]) / sd[:8]
            ).astype("float32")
        )

        # --------------------------------------------------
        # 6. CNN ve GRU modellerinin test edilmesi
        # --------------------------------------------------

        # Her iki model de aynı giriş ve hedef
        # verileri üzerinde test edilir.
        models = (
            CNNForecast(horizon=horizon),
            GRUForecastFlexible(horizon=horizon)
        )

        for model in models:

            # Modelin ileri yayılım işlemiyle
            # tahmin üretmesini sağla.
            pred = model(xb)

            # Tahmin çıktısı ile hedef verinin
            # boyutlarının aynı olduğunu doğrula.
            assert pred.shape == yt.shape

            # Ortalama karesel hata (MSE) hesapla.
            loss = torch.nn.functional.mse_loss(
                pred,
                yt
            )

            # Geri yayılım (backpropagation) işlemini çalıştır.
            # Böylece modelin türev hesaplama sürecinin
            # hata vermeden tamamlandığı kontrol edilir.
            loss.backward()

            # Kayıp değerinin NaN veya sonsuz
            # olmadığını doğrula.
            assert torch.isfinite(loss)

    # --------------------------------------------------
    # 7. Test sonucunun gösterilmesi
    # --------------------------------------------------

    print(
        "PASS: correct model shapes, backward pass, "
        "history-only normalization, no object leakage."
    )


if __name__ == "__main__":
    main()
