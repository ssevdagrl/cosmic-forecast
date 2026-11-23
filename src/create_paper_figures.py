
"""
COSMIC FORECAST - Akademik Grafik Oluşturma

Bu dosya, mevcut deney sonuçlarını kullanarak
IEEE bildirisinde kullanılabilecek renkli
akademik grafikler oluşturur.

Oluşturulan grafikler:
    1. Model karşılaştırması - 24 saat
    2. Model karşılaştırması - 48 saat
    3. Ölçüm belirsizliği analizi - 24 saat
    4. Ölçüm belirsizliği analizi - 48 saat
    5. Bootstrap güven aralıkları - 24 saat
    6. Bootstrap güven aralıkları - 48 saat

Çıktı klasörü:
    results/paper_figures/

Her grafik iki biçimde kaydedilir:
    PNG: 450 DPI
    PDF: Vektörel grafik

Bu dosya model eğitimi yapmaz.
Mevcut deney sonuçlarını değiştirmez.

Çalıştırma:
    python src/create_paper_figures.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt


# ==========================================================
# 1. PROJE KLASÖRLERİ
# ==========================================================

# Projenin ana dizinini belirle.
ROOT = Path(__file__).resolve().parents[1]

# Deney sonuçlarının bulunduğu klasör.
RESULTS = ROOT / "results"

# Akademik grafiklerin kaydedileceği klasör.
OUTPUT = RESULTS / "paper_figures"

OUTPUT.mkdir(parents=True, exist_ok=True)


# ==========================================================
# 2. GRAFİK RENKLERİ
# ==========================================================

# Aynı model bütün grafiklerde aynı renkle gösterilir.
# Böylece makale boyunca görsel tutarlılık korunur.

COLORS = {
    "Baseline": "#94A3B8",
    "Persistence": "#94A3B8",
    "Mean": "#CBD5E1",
    "Median": "#B5BFCD",
    "LinearTrend": "#E9B44C",
    "Ridge": "#4C78A8",
    "GRU": "#2A9D8F",
    "CNN": "#E76F51",
}

# Grafiklerde görüntülenecek model isimleri.
LABELS = {
    "Baseline": "Persistence",
    "Mean": "Geçmiş Ortalaması",
    "Median": "Geçmiş Medyanı",
    "LinearTrend": "Doğrusal Trend",
    "Ridge": "Ridge Regression",
    "GRU": "GRU",
    "CNN": "1D-CNN",
    "Persistence": "Persistence",
}

# Fotometrik belirsizlik gruplarının sıralaması.
QUALITY_ORDER = [
    "Dusuk",
    "Orta",
    "Yuksek",
]

QUALITY_LABELS = [
    "Düşük",
    "Orta",
    "Yüksek",
]


# ==========================================================
# 3. AKADEMİK GRAFİK TASARIMI
# ==========================================================

# Grafiklerin yazı tipi, çizgi kalınlığı ve genel
# görünümü bütün şekillerde aynı olacak şekilde ayarlanır.

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "axes.titleweight": "bold",
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "axes.linewidth": 0.7,
    "axes.edgecolor": "#334155",
    "axes.labelcolor": "#334155",
    "xtick.color": "#475569",
    "ytick.color": "#475569",
    "text.color": "#1E293B",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def style_axes(ax, grid_axis="x"):
    """
    Grafik eksenlerine ortak akademik görünümü uygular.

    Üst ve sağ çerçeveler kaldırılır.
    Okunabilirliği artırmak için açık renkli
    yardımcı ızgara çizgileri eklenir.
    """

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.grid(
        axis=grid_axis,
        color="#E2E8F0",
        linewidth=0.7,
        linestyle="-",
        alpha=0.8
    )

    ax.set_axisbelow(True)

    ax.tick_params(
        axis="both",
        length=3,
        width=0.6
    )


def save_figure(fig, name):
    """
    Oluşturulan grafiği iki farklı biçimde kaydeder.

    PDF: Vektörel çıktı
    PNG: 450 DPI çözünürlük
    """

    fig.savefig(
        OUTPUT / f"{name}.pdf",
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.10,
    )

    fig.savefig(
        OUTPUT / f"{name}.png",
        format="png",
        dpi=450,
        bbox_inches="tight",
        pad_inches=0.10,
    )

    plt.close(fig)

    print(f"[OK] {name}.png")
    print(f"[OK] {name}.pdf")


# ==========================================================
# 4. DENEY SONUÇLARININ YÜKLENMESİ
# ==========================================================

def load_model_comparison():
    """
    Klasik tahmin yöntemleriyle sinir ağı modellerinin
    sonuçlarını tek bir veri tablosunda birleştirir.

    24 ve 48 saatlik tahminler ayrı değerlendirilir.
    """

    classical = pd.read_csv(
        RESULTS / "classical_summary_all.csv"
    )

    neural = pd.read_csv(
        RESULTS / "model_summary_all.csv"
    )

    models = [
        "Baseline",
        "Mean",
        "Median",
        "LinearTrend",
        "Ridge",
        "GRU",
        "CNN",
    ]

    rows = []

    for hours in (24, 48):

        for model in models:

            # Sinir ağı sonuçları ve klasik yöntem
            # sonuçları farklı CSV dosyalarından alınır.
            source = (
                neural
                if model in ("GRU", "CNN")
                else classical
            )

            selected = source.loc[
                (source["forecast_h"] == hours)
                & (source["model"] == model)
            ]

            # Her model ve tahmin süresi için
            # yalnızca bir sonuç bulunmalıdır.
            if len(selected) != 1:
                raise ValueError(
                    f"Sonuc eksik veya tekrarli: "
                    f"{model}, {hours} saat"
                )

            rows.append({
                "horizon_h": hours,
                "model": model,
                "MAE": float(
                    selected.iloc[0]["MAE_counts_per_s"]
                ),
                "RMSE": float(
                    selected.iloc[0]["RMSE_counts_per_s"]
                ),
            })

    return pd.DataFrame(rows)


def load_quality_results():
    """
    Fotometrik belirsizlik analizinden elde edilen
    model performanslarını yükler.
    """

    return pd.read_csv(
        RESULTS
        / "quality_analysis"
        / "quality_summary_all.csv"
    )


def load_bootstrap_results():
    """
    Pencere ağırlıklı ve süpernova bazında kümelenmiş
    bootstrap güven aralıklarını yükler.
    """

    return pd.read_csv(
        RESULTS
        / "quality_analysis"
        / "quality_weighted_bootstrap.csv"
    )


# ==========================================================
# 5. GENEL MODEL KARŞILAŞTIRMASI
# ==========================================================

def plot_model_comparison(data, hours):
    """
    Yedi farklı tahmin yöntemini MAE değerlerine
    göre karşılaştıran yatay çubuk grafik oluşturur.

    Düşük MAE daha başarılı tahmin anlamına gelir.
    Modeller en düşük hatadan en yükseğe sıralanır.
    """

    subset = (
        data.loc[data["horizon_h"] == hours]
        .sort_values("MAE")
        .reset_index(drop=True)
    )

    values = subset["MAE"].to_numpy()

    model_names = subset["model"].tolist()

    labels = [
        LABELS[name]
        for name in model_names
    ]

    colors = [
        COLORS[name]
        for name in model_names
    ]

    # IEEE makalesinin tek sütununa uygun grafik boyutu.
    fig, ax = plt.subplots(
        figsize=(3.50, 2.85)
    )

    positions = np.arange(len(values))

    bars = ax.barh(
        positions,
        values,
        height=0.66,
        color=colors,
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )

    ax.set_yticks(
        positions,
        labels
    )

    # En düşük hatalı model en üstte gösterilir.
    ax.invert_yaxis()

    # Sayısal değerlerin çubukların dışına
    # yazılabilmesi için sağ tarafta boşluk bırakılır.
    ax.set_xlim(
        0,
        float(values.max()) * 1.22
    )

    ax.set_xlabel(
        "Ortalama Mutlak Hata (counts/s)"
    )

    ax.set_title(
        f"Model Performansı - {hours} Saat",
        pad=12
    )

    style_axes(ax, grid_axis="x")

    ax.tick_params(
        axis="y",
        length=0
    )

    # Her çubuğun sonuna ilgili MAE değerini yaz.
    for bar, value in zip(bars, values):

        ax.text(
            value + values.max() * 0.020,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.3f}",
            ha="left",
            va="center",
            fontsize=7,
            color="#334155"
        )

    # En başarılı modelin ismini kalın göster.
    tick_labels = ax.get_yticklabels()

    if tick_labels:
        tick_labels[0].set_fontweight("bold")

    fig.tight_layout()

    save_figure(
        fig,
        f"fig_models_{hours}h"
    )


# ==========================================================
# 6. ÖLÇÜM BELİRSİZLİĞİ ANALİZİ
# ==========================================================

def plot_quality_analysis(data, hours):
    """
    Düşük, orta ve yüksek fotometrik belirsizlik
    gruplarında modellerin MAE değişimini gösterir.

    Belirsizlik grupları tahmin öncesindeki
    72 saatlik gözlemlerden belirlenmiştir.
    """

    subset = data.loc[
        data["horizon_h"] == hours
    ]

    # Modeller farklı çizgi ve işaretçi biçimleriyle
    # gösterilerek renkli ve siyah-beyaz baskıda
    # ayırt edilebilirlik artırılır.
    model_styles = {
        "Persistence": {
            "marker": "o",
            "linestyle": "-",
        },
        "Ridge": {
            "marker": "s",
            "linestyle": "--",
        },
        "GRU": {
            "marker": "^",
            "linestyle": "-.",
        },
        "CNN": {
            "marker": "D",
            "linestyle": ":",
        },
    }

    fig, ax = plt.subplots(
        figsize=(3.50, 2.85)
    )

    x_positions = np.arange(
        len(QUALITY_ORDER)
    )

    for model, style in model_styles.items():

        selected = (
            subset.loc[
                subset["model"] == model
            ]
            .set_index("quality_group")
        )

        # Üç kalite grubunun tamamının
        # sonuçlarda bulunduğunu doğrula.
        if not set(QUALITY_ORDER).issubset(
            selected.index
        ):
            raise ValueError(
                f"Kalite grubu eksik: "
                f"{model}, {hours} saat"
            )

        values = selected.loc[
            QUALITY_ORDER,
            "MAE_counts_per_s"
        ].to_numpy()

        ax.plot(
            x_positions,
            values,
            color=COLORS[model],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=5.3,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label=LABELS[model],
            zorder=3
        )

    ax.set_xticks(
        x_positions,
        QUALITY_LABELS
    )

    ax.set_xlabel(
        "Geçmiş Ölçüm Belirsizliği"
    )

    ax.set_ylabel(
        "Ortalama Mutlak Hata (counts/s)"
    )

    ax.set_title(
        f"Ölçüm Belirsizliği Analizi - {hours} Saat",
        pad=12
    )

    ax.set_ylim(bottom=0)

    style_axes(ax, grid_axis="y")

    ax.legend(
        frameon=False,
        ncol=2,
        loc="upper left",
        columnspacing=1.1,
        handlelength=2.1
    )

    fig.tight_layout()

    save_figure(
        fig,
        f"fig_quality_{hours}h"
    )


# ==========================================================
# 7. BOOTSTRAP GÜVEN ARALIKLARI
# ==========================================================

def plot_bootstrap(data, hours):
    """
    GRU ve CNN modellerinin Persistence yöntemine
    göre sağladığı MAE iyileşmesini gösterir.

    Her model için %95 bootstrap güven aralığı
    yatay hata çubuğu olarak çizilir.

    Bootstrap örnekleme birimi süpernovadır.
    MAE farkı pencere sayılarıyla ağırlıklandırılır.

    Pozitif fark:
        Sinir ağı modelinin MAE değeri daha düşüktür.

    Negatif fark:
        Persistence yönteminin MAE değeri daha düşüktür.
    """

    subset = data.loc[
        data["horizon_h"] == hours
    ]

    rows = []

    for group, group_label in zip(
        QUALITY_ORDER,
        QUALITY_LABELS
    ):

        for model in ("GRU", "CNN"):

            comparison = (
                f"Persistence - {model}"
            )

            selected = subset.loc[
                (
                    subset["quality_group"]
                    == group
                )
                & (
                    subset["comparison"]
                    == comparison
                )
            ]

            if len(selected) != 1:
                raise ValueError(
                    f"Bootstrap sonucu eksik: "
                    f"{hours} saat, {group}, {model}"
                )

            item = selected.iloc[0]

            rows.append({
                "label": f"{group_label} / {model}",
                "model": model,
                "value": float(
                    item["window_weighted_MAE_difference"]
                ),
                "low": float(
                    item["CI95_low"]
                ),
                "high": float(
                    item["CI95_high"]
                ),
            })

    fig, ax = plt.subplots(
        figsize=(3.50, 2.85)
    )

    positions = np.arange(len(rows))

    for position, item in zip(
        positions,
        rows
    ):

        value = item["value"]
        low = item["low"]
        high = item["high"]

        # Hesaplanan güven aralığı nokta tahminini
        # içermiyorsa veri tutarlılığı hatası oluştur.
        if not low <= value <= high:
            raise ValueError(
                "Guven araligi nokta tahminini icermiyor."
            )

        model = item["model"]

        is_cnn = model == "CNN"

        color = COLORS[model]

        lower_error = value - low
        upper_error = high - value

        # CNN elmas, GRU daire işaretçiyle gösterilir.
        ax.errorbar(
            value,
            position,
            xerr=np.array([
                [lower_error],
                [upper_error]
            ]),
            fmt="D" if is_cnn else "o",
            markersize=5.3,
            capsize=3,
            capthick=1.2,
            color=color,
            ecolor=color,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=0.6,
            elinewidth=1.4,
            zorder=3
        )

    ax.set_yticks(
        positions,
        [item["label"] for item in rows]
    )

    ax.invert_yaxis()

    # Sıfır çizgisi iki yöntemin aynı MAE değerini
    # elde ettiği referans noktasıdır.
    ax.axvline(
        0,
        color="#475569",
        linestyle="--",
        linewidth=1,
        zorder=2
    )

    ax.set_xlabel(
        "Persistence - Model MAE Farkı (counts/s)"
    )

    ax.set_title(
        f"Bootstrap Güven Aralıkları - {hours} Saat",
        pad=12
    )

    style_axes(ax, grid_axis="x")

    ax.tick_params(
        axis="y",
        length=0
    )

    # Bütün güven aralıklarının grafikte
    # eksiksiz görünmesini sağla.
    lows = [
        row["low"]
        for row in rows
    ]

    highs = [
        row["high"]
        for row in rows
    ]

    minimum = min(
        0.0,
        min(lows)
    )

    maximum = max(
        0.0,
        max(highs)
    )

    span = maximum - minimum

    padding = max(
        span * 0.09,
        0.025
    )

    ax.set_xlim(
        minimum - padding,
        maximum + padding
    )

    fig.tight_layout()

    save_figure(
        fig,
        f"fig_bootstrap_{hours}h"
    )


# ==========================================================
# 8. ANA PROGRAM
# ==========================================================

def main():
    """
    Deney sonuçlarını yükler ve altı akademik
    grafiği otomatik olarak oluşturur.
    """

    print()
    print("=" * 55)
    print("COSMIC FORECAST - AKADEMIK GRAFIKLER")
    print("=" * 55)

    print("\nDeney sonuclari yukleniyor...")

    model_data = load_model_comparison()
    quality_data = load_quality_results()
    bootstrap_data = load_bootstrap_results()

    print("[OK] Model karsilastirmalari yuklendi.")
    print("[OK] Olcum belirsizligi sonuclari yuklendi.")
    print("[OK] Bootstrap sonuclari yuklendi.")

    print("\nGrafikler olusturuluyor...\n")

    # Her tahmin süresi için üç ayrı grafik oluştur.
    for hours in (24, 48):

        plot_model_comparison(
            model_data,
            hours
        )

        plot_quality_analysis(
            quality_data,
            hours
        )

        plot_bootstrap(
            bootstrap_data,
            hours
        )

    print()
    print("=" * 55)
    print("ALTI GRAFIK BASARIYLA OLUSTURULDU")
    print("=" * 55)

    print(f"\nCikti klasoru:\n{OUTPUT}")

    print("\nBicimler: Vektorel PDF ve 450 DPI PNG")

    print(
        "\nMevcut modeller ve deney sonuclari degistirilmedi."
    )


if __name__ == "__main__":
    main()
