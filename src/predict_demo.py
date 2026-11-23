"""
Cosmic Forecast - Test Süpernovası İçin Tahmin Gösterimi

Eğitimde kullanılmamış (test kümesine ayrılmış) bir TESS süpernovası
üzerinde, önceden eğitilmiş GRU veya CNN modeliyle tahmin oluşturur.
Gerçek gelecek ölçümleri ve son-değer (persistence) baseline ile
karşılaştırma yaparak sonuçları terminalde ve grafikte gösterir.

Kullanım örnekleri:
    python src/predict_demo.py --list
    python src/predict_demo.py --model cnn --horizon 24
    python src/predict_demo.py --name SN2018fhw --model gru --horizon 48

Önemli: Bu işlem, geçmişte kaydedilmiş veriler üzerinde test amaçlı
bir canlandırmadır; gerçek zamanlı bir astronomik uyarı sistemi değildir.
"""

import argparse
import re

import matplotlib

# Grafikler, pencere açılmadan dosyaya kaydedilir.
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from dataset import (
    RESULTS,
    load_photometry,
    make_windows,
    scaling_from_history,
    split_by_object,
)
from evaluate_all import MODEL_CONFIG, predict_model


def main():
    # --------------------------------------------------
    # 1. Komut satırı seçeneklerini oku
    # --------------------------------------------------

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--list",
        action="store_true",
        help="List test-only supernova names",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Name of TEST supernova",
    )
    parser.add_argument("--horizon", type=int, choices=[24, 48], default=24)
    parser.add_argument("--model", type=str, choices=["gru", "cnn"], default="gru")
    args = parser.parse_args()

    # Veri setindeki her tahmin adımı 6 saat olduğundan,
    # 24 saat = 4 adım, 48 saat = 8 adım.
    horizon_bins = args.horizon // 6

    # --------------------------------------------------
    # 2. Test kümesine ait süpernovaları belirle
    # --------------------------------------------------

    raw = load_photometry()
    x, y, names, t0 = make_windows(
        raw,
        lookback=12,
        horizon=horizon_bins,
    )

    # Eğitim ve değerlendirmeyle aynı kimlik bazlı veri ayrımı
    # kullanılır. Böylece eğitim süpernovaları "görülmemiş" diye sunulmaz.
    split = split_by_object(names, seed=42)
    eligible = sorted(set(names[split == "test"]))

    # İstenirse seçilen tahmin süresi için test kümesi adlarını listele.
    if args.list:
        print("Test-only supernovae (first 60):")
        for name in eligible[:60]:
            print(" ", name)
        return

    # Süpernova adı verilmemişse alfabetik olarak ilk test nesnesini seç.
    # Seçim, modelin tahmin başarısına bakılarak yapılmaz.
    name = args.name or eligible[0]

    if name not in eligible:
        raise ValueError(
            f'"{name}" is not in this horizon\'s held-out test set. '
            "Use --list to see names. No training examples are demoed as unseen."
        )

    # Seçilen süpernovanın test pencerelerinden, başlangıç zamanı
    # en erken olanı belirle. Böylece örnek elle seçilerek iyileştirilmez.
    indices = np.flatnonzero((split == "test") & (names == name))
    j = int(indices[np.argmin(t0[indices])])

    # --------------------------------------------------
    # 3. Geçmiş gözlemleri normalize et ve modeli yükle
    # --------------------------------------------------

    # Ortalama ve ölçek yalnızca geçmiş 72 saatlik gözlemlerden
    # hesaplanır; gelecek gerçek değerleri kullanılmaz.
    mu, sd = scaling_from_history(x[j:j + 1])
    normalized = ((x[j:j + 1] - mu) / sd).astype("float32")

    # İstenen tahmin süresi ve mimariye karşılık gelen
    # önceden eğitilmiş model dosyasını seç.
    filename, constructor = MODEL_CONFIG[horizon_bins][args.model.upper()]
    weights = RESULTS / filename

    if not weights.exists():
        raise FileNotFoundError(f"{weights} missing. Train selected model first.")

    # evaluate_all.py içindeki ortak tahmin yordamını kullan.
    # Bu fonksiyon modeli yükler ve eğitim yapmadan çıktı üretir.
    prediction_n = predict_model(weights, constructor, normalized)

    # Tahmini normalize edilmiş uzaydan orijinal counts/s birimine döndür.
    prediction = (prediction_n * sd + mu)[0]

    # Baseline: Son gözlenen ışık akısını bütün gelecek adımlarda koru.
    baseline = np.repeat(x[j, -1], horizon_bins)

    # --------------------------------------------------
    # 4. Gelecek değerleri terminalde karşılaştır
    # --------------------------------------------------

    print(f"Supernova={name}; model={args.model.upper()}; horizon={args.horizon}h")
    print("Hours  Truth_counts/s  Prediction_counts/s  Baseline_counts/s")

    for k in range(horizon_bins):
        print(
            f"{(k + 1) * 6:>5d}  "
            f"{y[j, k]:>14.4f}  "
            f"{prediction[k]:>19.4f}  "
            f"{baseline[k]:>16.4f}"
        )

    # --------------------------------------------------
    # 5. Geçmiş gözlemleri ve gelecek tahminlerini çiz
    # --------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 5))

    # Geçmiş 12 ölçümün zaman ekseni: -66, -60, ..., 0 saat.
    # Son gözlem sıfır noktası olarak kabul edilir.
    ax.plot(
        np.arange(-66, 6, 6),
        x[j],
        "o-",
        label="Observed 72h history",
    )

    # Gelecek zaman adımları: 6, 12, ... 24/48 saat.
    future_steps = np.arange(6, (horizon_bins + 1) * 6, 6)

    # Gerçek gelecek değerleri yalnızca değerlendirme amaçlı çizilir;
    # modelin tahmin girdisine dahil edilmez.
    ax.plot(
        future_steps,
        y[j],
        "o-",
        label="Actual future (for evaluation only)",
    )
    ax.plot(
        future_steps,
        prediction,
        "s--",
        label=f"{args.model.upper()} forecast",
    )
    ax.plot(
        future_steps,
        baseline,
        ":",
        label="Last-value baseline",
    )

    # Geçmiş gözlemler ile tahmin dönemini ayıran sınır.
    ax.axvline(0, linestyle=":", color="gray")
    ax.set(
        xlabel="Hours from last observed measurement",
        ylabel="Count rate (counts/s)",
        title=f"COSMIC FORECAST — {name} — {args.horizon}h",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()

    # Dosya adına uygun olmayan karakterleri temizleyerek
    # her örneğin grafiğini ayrı PNG dosyasına kaydet.
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)[:60]
    path = RESULTS / f"prediction_demo_{slug}_{args.model}_{args.horizon}h.png"
    fig.savefig(path, dpi=170)
    plt.close(fig)

    print("Saved:", path)


if __name__ == "__main__":
    main()
