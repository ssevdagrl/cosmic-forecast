
"""
Cosmic Forecast - Veri Seti Analizi

Bu dosya, NASA TESS tarafından gözlemlenen Tip Ia süpernovalarının
fotometrik verilerini analiz eder.

Gerçekleştirilen işlemler:
    1. Geçerli fotometrik ölçümlerin sayılması
    2. Bağımsız süpernova sayısının belirlenmesi
    3. Gözlem sürelerinin ve ölçüm belirsizliklerinin incelenmesi
    4. Örnek süpernova ışık eğrilerinin görselleştirilmesi
    5. Farklı tahmin süreleri için uygun veri pencerelerinin sayılması
    6. Analiz sonuçlarının CSV, JSON ve PNG olarak kaydedilmesi

Kullanım:
    python src/inspect_data.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from dataset import RESULTS, load_photometry, make_windows


def main():
    # --------------------------------------------------
    # 1. Veri setinin yüklenmesi
    # --------------------------------------------------

    # Analiz sonuçlarının kaydedileceği klasörü oluştur.
    RESULTS.mkdir(exist_ok=True, parents=True)

    # NASA TESS fotometrik verilerini yükle.
    df = load_photometry()

    # Veri setindeki benzersiz süpernova sayısını hesapla.
    count = int(df["Name"].nunique())

    print(
        f"Gecerli fotometri satiri: {len(df):,}; "
        f"bagimsiz supernova: {count}"
    )

    # --------------------------------------------------
    # 2. Süpernova bazında gözlem analizi
    # --------------------------------------------------

    # Her süpernova için ölçüm sayısını, ilk ve son gözlem
    # zamanını ve ortanca ölçüm belirsizliğini hesapla.
    summary = df.groupby("Name").agg(
        n_measurements=("BJD", "size"),
        tmin=("BJD", "min"),
        tmax=("BJD", "max"),
        median_error=("e_CRate", "median")
    )

    # İlk ve son gözlem arasındaki toplam zaman farkı.
    # BJD zaman birimi gün olduğu için sonuç gün cinsindendir.
    #
    # Bu değer kesintisiz gözlem süresini garanti etmez.
    summary["days_span"] = (
        summary["tmax"] - summary["tmin"]
    )

    # Süpernova bazındaki istatistikleri CSV olarak kaydet.
    summary.to_csv(
        RESULTS / "supernova_ozet.csv"
    )

    # Gözlem sürelerinin istatistiksel dağılımını göster.
    print("Olcum suresi (gun) dagilimi:")

    print(
        summary["days_span"]
        .describe()
        .round(2)
        .to_string()
    )

    # --------------------------------------------------
    # 3. Örnek süpernova ışık eğrileri
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    # En fazla ölçüme sahip üç süpernovayı seç.
    # Seçim tahmin başarısına göre değil,
    # mevcut gözlem sayısına göre yapılır.
    sample_names = (
        summary.sort_values(
            "n_measurements",
            ascending=False
        )
        .head(3)
        .index
    )

    for name in sample_names:

        # İlgili süpernovaya ait fotometrik ölçümler.
        g = df[df["Name"] == name]

        # Zaman eksenini ilk gözleme göre düzenle.
        # Bu sıfır noktası fiziksel patlama anı değildir.
        relative_time = (
            g["BJD"] - g["BJD"].min()
        )

        # Parlaklık değişimini zaman içerisinde çiz.
        ax.plot(
            relative_time,
            g["CRate"],
            label=str(name),
            linewidth=1.2
        )

    ax.set(
        xlabel="Ilk gozlemden itibaren gun",
        ylabel="Isik akisi (count/s)",
        title="TESS: ornek Tip Ia supernova isik egrileri"
    )

    ax.legend()
    fig.tight_layout()

    # Oluşturulan ışık eğrilerini PNG olarak kaydet.
    fig.savefig(
        RESULTS / "ilk_isik_egrileri.png",
        dpi=160
    )

    plt.close(fig)

    # --------------------------------------------------
    # 4. Genel veri seti özetinin oluşturulması
    # --------------------------------------------------

    report = {
        "n_rows": int(len(df)),
        "n_supernova": count,
        "median_observation_days": round(
            float(summary["days_span"].median()),
            2
        )
    }

    # --------------------------------------------------
    # 5. Tahmin pencerelerinin uygunluk analizi
    # --------------------------------------------------

    # Her zaman adımı 6 saati temsil eder.
    #
    # lookback=12 -> 72 saatlik geçmiş gözlem
    # lookback=20 -> 120 saatlik geçmiş gözlem
    #
    # horizon=4 -> 24 saatlik tahmin
    # horizon=8 -> 48 saatlik tahmin
    #
    # Farklı geçmiş ve gelecek sürelerinin veri setinde
    # kaç uygun pencere oluşturabildiği incelenir.
    configurations = [
        (12, 4),
        (12, 8),
        (20, 4),
        (20, 8)
    ]

    for lookback, horizon in configurations:

        # İlgili deney yapılandırmasını tanımlayan anahtar.
        key = (
            f"lookback_{lookback * 6}h_"
            f"future_{horizon * 6}h"
        )

        try:
            # Geçmiş gözlemleri ve gelecek hedeflerini oluştur.
            #
            # make_windows() yalnızca eksiksiz
            # zaman pencerelerini kabul eder.
            x, y, objects, _ = make_windows(
                df,
                lookback,
                horizon
            )

            # Üretilen tahmin pencerelerinin sayısını
            # ve bunların kaç farklı süpernovaya
            # ait olduğunu hesapla.
            report[key] = {
                "n_windows": int(len(x)),
                "n_independent_supernova": int(
                    len(np.unique(objects))
                )
            }

        except ValueError as err:

            # İlgili yapılandırma için uygun pencere
            # bulunamazsa hata mesajını rapora kaydet.
            report[key] = {
                "error": str(err)
            }

    # --------------------------------------------------
    # 6. Analiz sonuçlarının kaydedilmesi
    # --------------------------------------------------

    # Veri seti özeti ve pencere uygunluk sonuçlarını
    # JSON biçiminde kaydet.
    with open(
        RESULTS / "veri_denetimi.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            report,
            f,
            ensure_ascii=False,
            indent=2
        )

    # --------------------------------------------------
    # 7. Sonuçların terminalde gösterilmesi
    # --------------------------------------------------

    print("\nPencere uygunlugu:")

    for key, value in report.items():
        if isinstance(value, dict):
            print(key, value)

    print(
        "\nKaydedildi: results/ilk_isik_egrileri.png, "
        "supernova_ozet.csv, veri_denetimi.json"
    )


if __name__ == "__main__":
    main()
