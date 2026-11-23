
"""
Cosmic Forecast - GRU Modeli Performans Değerlendirmesi

Bu dosya, önceden eğitilmiş GRU modelinin NASA TESS
süpernova verileri üzerindeki tahmin başarısını değerlendirir.

Gerçekleştirilen işlemler:
    1. Eğitilmiş GRU modelinin yüklenmesi
    2. Eğitimde kullanılmamış test süpernovalarının seçilmesi
    3. Gelecek 24 saatlik ışık akısı tahminlerinin oluşturulması
    4. GRU ve Baseline yöntemlerinin karşılaştırılması
    5. MAE ve RMSE performans metriklerinin hesaplanması
    6. Süpernova bazında bootstrap güven aralığının hesaplanması
    7. Tahmin grafiklerinin ve sonuç dosyalarının oluşturulması

Model yeniden eğitilmez. Eğitimde kullanılan veri bölmesi
(seed=42) ve geçmiş gözlemlere dayalı normalizasyon korunur.

Kullanım:
    python src/evaluate_gru.py
"""

import json

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
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
from train_gru import GRUForecast


def main():
    # --------------------------------------------------
    # 1. Eğitilmiş GRU modelinin kontrolü
    # --------------------------------------------------

    # Eğitim sonucunda kaydedilen model ağırlıklarının konumu.
    weights = RESULTS / "gru_model.pt"

    # Model dosyası bulunmuyorsa değerlendirmeyi başlatma.
    if not weights.is_file():
        raise FileNotFoundError(
            f"{weights} yok. "
            "Once python src/train_gru.py calistir."
        )

    # --------------------------------------------------
    # 2. Test verilerinin hazırlanması
    # --------------------------------------------------

    # NASA TESS fotometrik ölçümlerini yükle.
    df = load_photometry()

    # 72 saatlik geçmiş gözlemden sonraki 24 saatlik
    # ışık akısını tahmin etmek için pencereler oluştur.
    #
    # Her zaman adımı 6 saati temsil eder.
    # lookback=12 -> 72 saat
    # horizon=4  -> 24 saat
    x, y, names, t0s = make_windows(
        df,
        lookback=12,
        horizon=4
    )

    # Süpernovaları eğitim, doğrulama ve test kümelerine
    # kimliklerine göre ayır.
    split = split_by_object(names)

    # Yalnızca test kümesindeki süpernovaları seç.
    selected = split == "test"

    if not selected.any():
        raise ValueError("Test kumesi bos.")

    # Her pencerenin normalizasyon parametrelerini
    # yalnızca geçmiş gözlemlerden hesapla.
    mu, sd = scaling_from_history(x)

    # Test girdilerini GRU modelinin beklediği
    # normalize edilmiş biçime dönüştür.
    norm_x = (
        (x[selected] - mu[selected]) / sd[selected]
    ).astype(np.float32)

    # --------------------------------------------------
    # 3. GRU modelinin yüklenmesi ve tahmin üretimi
    # --------------------------------------------------

    # Eğitimde kullanılan GRU mimarisini oluştur.
    model = GRUForecast(hidden=32)

    # Kaydedilmiş model ağırlıklarını CPU üzerine yükle.
    model.load_state_dict(
        torch.load(
            weights,
            map_location="cpu",
            weights_only=True
        )
    )

    # Modeli değerlendirme moduna geçir.
    model.eval()

    # Gradyan hesaplaması olmadan tahmin üret.
    # Model girdisi: örnek sayısı x 12 zaman adımı x 1 özellik
    with torch.no_grad():
        pred_n = model(
            torch.from_numpy(norm_x).unsqueeze(-1)
        ).numpy()

    # Gerçek gelecek ışık akısı değerlerini al.
    truth = y[selected]

    # Normalize edilmiş tahminleri tekrar orijinal
    # ışık akısı birimine (counts/s) dönüştür.
    pred = (
        pred_n * sd[selected] + mu[selected]
    )

    # Baseline yöntemi:
    # Son gözlenen ışık akısı değerinin gelecek
    # dört zaman adımında değişmediğini varsay.
    baseline = np.repeat(
        x[selected, -1:],
        4,
        axis=1
    )

    # Test örneklerinin süpernova kimliklerini ve
    # tahmin başlangıç zamanlarını sakla.
    test_names = names[selected]
    test_times = t0s[selected]

    # --------------------------------------------------
    # 4. Tahmin ufuklarına göre hata hesaplamaları
    # --------------------------------------------------

    rows_h = []

    # Gelecek 6, 12, 18 ve 24 saat için
    # hata değerlerini ayrı ayrı hesapla.
    for k in range(4):

        # GRU tahmin hataları.
        err_g = truth[:, k] - pred[:, k]

        # Baseline tahmin hataları.
        err_b = truth[:, k] - baseline[:, k]

        # MAE: Ortalama mutlak hata.
        # RMSE: Karekök ortalama karesel hata.
        rows_h.append({
            "horizon_h": (k + 1) * 6,
            "GRU_MAE": float(np.mean(np.abs(err_g))),
            "Baseline_MAE": float(np.mean(np.abs(err_b))),
            "GRU_RMSE": float(
                np.sqrt(np.mean(err_g ** 2))
            ),
            "Baseline_RMSE": float(
                np.sqrt(np.mean(err_b ** 2))
            ),
        })

    horizon_df = pd.DataFrame(rows_h)

    # --------------------------------------------------
    # 5. Süpernova bazında performans analizi
    # --------------------------------------------------

    object_rows = []

    # Her bağımsız süpernova için hata değerlerini
    # ayrı hesapla.
    for obj in np.unique(test_names):

        # İlgili süpernovaya ait bütün test pencereleri.
        idx = test_names == obj

        # GRU ve Baseline tahmin hataları.
        e_g = truth[idx] - pred[idx]
        e_b = truth[idx] - baseline[idx]

        object_rows.append({
            "Name": obj,
            "n_windows": int(np.sum(idx)),
            "GRU_MAE": float(np.mean(np.abs(e_g))),
            "Baseline_MAE": float(np.mean(np.abs(e_b))),
            "GRU_RMSE": float(
                np.sqrt(np.mean(e_g ** 2))
            ),
            "Baseline_RMSE": float(
                np.sqrt(np.mean(e_b ** 2))
            ),
        })

    # Sonuçları süpernova kimliğine göre sırala.
    object_df = pd.DataFrame(object_rows).sort_values(
        "Name"
    )

    # Pozitif değer GRU modelinin Baseline'dan
    # daha düşük MAE elde ettiğini gösterir.
    object_df["MAE_advantage"] = (
        object_df["Baseline_MAE"] - object_df["GRU_MAE"]
    )

    # --------------------------------------------------
    # 6. Bootstrap güven aralığı hesaplaması
    # --------------------------------------------------

    # Aynı süpernovadan üretilen zaman pencereleri
    # birbirinden bağımsız değildir.
    #
    # Bu nedenle bootstrap örneklemesi pencere
    # bazında değil, süpernova bazında yapılır.

    # Tekrarlanabilirlik için sabit rastgelelik tohumu.
    rng = np.random.default_rng(42)

    # Her süpernovanın Baseline'a göre MAE kazancı.
    differences = object_df[
        "MAE_advantage"
    ].to_numpy()

    # Süpernovalardan yerine koymalı örnekleme yap.
    # 2000 bootstrap tekrarı oluşturulur.
    boot = rng.choice(
        differences,
        size=(2000, len(differences)),
        replace=True
    ).mean(axis=1)

    # Bootstrap dağılımının %2,5 ve %97,5
    # yüzdeliklerini kullanarak yaklaşık %95
    # güven aralığını hesapla.
    lo, hi = np.quantile(
        boot,
        [0.025, 0.975]
    )

    # İstatistiksel değerlendirme sonuçları.
    stats = {
        "n_test_supernovae": int(len(object_df)),
        "n_test_windows": int(len(truth)),
        "objects_GRU_better": int(
            np.sum(differences > 0)
        ),
        "objects_GRU_worse": int(
            np.sum(differences < 0)
        ),
        "equal_object_weight_average_MAE_advantage_baseline_minus_GRU":
            float(np.mean(differences)),
        "approx_95pct_bootstrap_CI_object_level_MAE_advantage": [
            float(lo),
            float(hi)
        ],
        "note": (
            "Bootstrap CI is approximate; one seed and one split "
            "are not sufficient to claim publication-grade significance."
        ),
        "evaluation_scope": (
            "Any complete 72h history -> next 24h window, "
            "not specifically explosion onset."
        ),
    }

    # --------------------------------------------------
    # 7. Sonuç dosyalarının kaydedilmesi
    # --------------------------------------------------

    # Sonuç klasörü mevcut değilse oluştur.
    RESULTS.mkdir(
        exist_ok=True,
        parents=True
    )

    # Tahmin ufuklarına göre performans metrikleri.
    horizon_df.to_csv(
        RESULTS / "gru_horizon_metrics.csv",
        index=False
    )

    # Süpernova bazında hata değerleri.
    object_df.to_csv(
        RESULTS / "gru_object_metrics.csv",
        index=False
    )

    # Bootstrap sonuçları JSON biçiminde kaydedilir.
    (RESULTS / "gru_object_bootstrap.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8"
    )

    # --------------------------------------------------
    # 8. Tahmin ufuklarına göre hata grafiği
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(8, 4.5)
    )

    # Baseline yönteminin MAE değerleri.
    ax.plot(
        horizon_df["horizon_h"],
        horizon_df["Baseline_MAE"],
        "o-",
        label="Last-value baseline"
    )

    # GRU modelinin MAE değerleri.
    ax.plot(
        horizon_df["horizon_h"],
        horizon_df["GRU_MAE"],
        "o-",
        label="GRU"
    )

    ax.set(
        xlabel="Forecast horizon (hours)",
        ylabel="MAE (counts/s)",
        title="Test-set forecast error by horizon"
    )

    ax.set_xticks([6, 12, 18, 24])
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()

    # Grafiği PNG olarak kaydet.
    fig.savefig(
        RESULTS / "gru_horizon_comparison.png",
        dpi=170
    )

    plt.close(fig)

    # --------------------------------------------------
    # 9. Gerçek ve tahmin edilen ışık eğrileri
    # --------------------------------------------------

    # Örnekler modelin başarı değerine göre seçilmez.
    # Alfabetik olarak ilk iki test süpernovası alınır.
    #
    # Her süpernova için en erken uygun tahmin
    # penceresi kullanılır.
    for example_no, name in enumerate(
        np.sort(np.unique(test_names))[:2],
        1
    ):

        # İlgili süpernovaya ait pencereleri bul.
        eligible = np.flatnonzero(
            test_names == name
        )

        # En erken tahmin başlangıç zamanını seç.
        j = eligible[
            np.argmin(test_times[eligible])
        ]

        fig, ax = plt.subplots(
            figsize=(8, 4.5)
        )

        # Geçmiş 72 saatlik gerçek gözlemler.
        ax.plot(
            np.arange(-66, 6, 6),
            x[selected][j],
            "o-",
            label="Observed history"
        )

        # Gelecek 24 saatlik gerçek ölçümler.
        ax.plot(
            np.arange(6, 25, 6),
            truth[j],
            "o-",
            label="Observed future"
        )

        # GRU tarafından tahmin edilen değerler.
        ax.plot(
            np.arange(6, 25, 6),
            pred[j],
            "s--",
            label="GRU prediction"
        )

        # Baseline yöntemiyle tahmin edilen değerler.
        ax.plot(
            np.arange(6, 25, 6),
            baseline[j],
            ":",
            label="Last-value baseline"
        )

        # Geçmiş ve gelecek gözlemler arasındaki sınır.
        ax.axvline(
            x=0,
            color="gray",
            linestyle=":",
            linewidth=1
        )

        ax.set(
            xlabel=(
                "Hours relative to forecast "
                "(0 = last observed point)"
            ),
            ylabel="Count rate (counts/s)",
            title=f"TESS {name}: one test window"
        )

        ax.legend()
        ax.grid(alpha=0.25)

        fig.tight_layout()

        # Her örneği ayrı bir görsel olarak kaydet.
        fig.savefig(
            RESULTS / f"gru_test_example_{example_no}.png",
            dpi=170
        )

        plt.close(fig)

    # --------------------------------------------------
    # 10. Sonuçların terminalde gösterilmesi
    # --------------------------------------------------

    print("\nGRU test summary:")

    print(
        f"Supernova: {stats['n_test_supernovae']}; "
        f"tahmin penceresi: {stats['n_test_windows']}"
    )

    print(horizon_df.to_string(index=False))

    print(
        f"\nSupernova bazinda GRU'nun daha iyi oldugu: "
        f"{stats['objects_GRU_better']}/"
        f"{stats['n_test_supernovae']}"
    )

    print(
        "Supernova es-agirlikli ortalama MAE kazanci "
        f"(baseline - GRU): {np.mean(differences):.4f} counts/s"
    )

    print(
        f"Yaklasik %95 bootstrap guven araligi: "
        f"[{lo:.4f}, {hi:.4f}]"
    )

    print(
        "Kaydedildi: results/gru_horizon_metrics.csv, "
        "gru_object_metrics.csv, gru_object_bootstrap.json,"
    )

    print(
        "           gru_horizon_comparison.png, "
        "gru_test_example_1.png, gru_test_example_2.png"
    )


if __name__ == "__main__":
    main()
