"""
Cosmic Forecast - Tüm Tahmin Modellerinin Değerlendirilmesi

Baseline, GRU ve 1D-CNN modellerini aynı test süpernovaları üzerinde
karşılaştırır. Her tahmin süresi için ayrı ve ortak bir değerlendirme
protokolü uygulanır:

    24 saat: gru_model.pt ve cnn_24h_model.pt
    48 saat: gru_48h_model.pt ve cnn_48h_model.pt

Test kümesi süpernova kimliklerine göre ayrılır. Aynı süpernovaya ait
zaman pencereleri örtüşebilir; bu nedenle güven aralıkları pencereler
yerine bağımsız süpernovalar yeniden örneklenerek hesaplanır.

Kullanım (proje ana klasöründe):
    python src/evaluate_all.py
"""

import json

import matplotlib

# Grafikler, ekranda pencere açılmadan PNG dosyasına kaydedilir.
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
from models_extra import CNNForecast, GRUForecastFlexible
from train_gru import GRUForecast


# Her tahmin süresi için kullanılacak eğitimli model dosyası ve mimarisi.
# Sözlük anahtarları altı saatlik zaman adımı sayılarıdır.
MODEL_CONFIG = {
    4: {
        "GRU": ("gru_model.pt", lambda: GRUForecast(hidden=32)),
        "CNN": (
            "cnn_24h_model.pt",
            lambda: CNNForecast(horizon=4, lookback=12),
        ),
    },
    8: {
        "GRU": (
            "gru_48h_model.pt",
            lambda: GRUForecastFlexible(horizon=8, hidden=32),
        ),
        "CNN": (
            "cnn_48h_model.pt",
            lambda: CNNForecast(horizon=8, lookback=12),
        ),
    },
}


def predict_model(checkpoint, constructor, normalized_input, batch_size=512):
    """Kaydedilmiş modelle normalleştirilmiş girdilerin tahminini üretir."""
    model = constructor()
    model.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    model.eval()

    pieces = []
    with torch.no_grad():
        # Test sırasında gradyan hesaplanmaz; bellek kullanımı azaltılır.
        for start in range(0, len(normalized_input), batch_size):
            xb = torch.from_numpy(
                normalized_input[start:start + batch_size]
            ).unsqueeze(-1)
            pieces.append(model(xb).numpy())

    return np.concatenate(pieces, axis=0)


def summarize(y_true, prediction):
    """Gerçek değerlerle tahminler arasındaki MAE ve RMSE'yi hesaplar."""
    err = y_true - prediction
    return (
        float(np.mean(np.abs(err))),
        float(np.sqrt(np.mean(err ** 2))),
    )


def plot_horizon(frame, tag):
    """Tahmin ufku uzadıkça üç yöntemin MAE değişimini görselleştirir."""
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for name in ["Baseline", "GRU", "CNN"]:
        subset = frame[frame["model"] == name].sort_values("horizon_h")
        ax.plot(
            subset["horizon_h"],
            subset["MAE_counts_per_s"],
            "o-",
            label=name,
        )

    ax.set(
        xlabel="Hours ahead",
        ylabel="MAE (counts/s)",
        title=f"TESS test supernovae: {tag} forecast error by horizon",
    )
    ax.set_xticks(sorted(frame["horizon_h"].unique()))
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULTS / f"model_horizon_{tag}.png", dpi=170)
    plt.close(fig)


def plot_examples(inputs, truth, predictions, names, times, tag):
    """İki test süpernovasının geçmişini ve gelecek tahminlerini çizer."""
    # Örnekler, model başarısına bakılmaksızın kimliğe göre seçilir.
    # Seçilen her süpernova için en erken uygun tahmin penceresi kullanılır.
    for plot_no, name in enumerate(sorted(set(names))[:2], start=1):
        ix = np.flatnonzero(names == name)
        j = int(ix[np.argmin(times[ix])])
        steps = truth.shape[1]

        # Son geçmiş ölçüm t=0; gelecek ölçümler 6, 12, ... saattedir.
        x_hist = np.arange(-66, 6, 6)
        x_future = np.arange(6, 6 * (steps + 1), 6)

        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(x_hist, inputs[j], "o-", label="Observed 72h history")
        ax.plot(x_future, truth[j], "o-", label="True future")

        for model in ["Baseline", "GRU", "CNN"]:
            ax.plot(
                x_future,
                predictions[model][j],
                "s--",
                label=model,
                markersize=3.5,
            )

        ax.axvline(0, ls=":", color="gray")
        ax.set(
            xlabel="Hours relative to last observation",
            ylabel="Count rate (counts/s)",
            title=f"TESS {name}: {tag} forecast",
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(RESULTS / f"model_example_{tag}_{plot_no}.png", dpi=170)
        plt.close(fig)


def cluster_bootstrap(object_frame, tag):
    """Süpernova bazında eşleştirilmiş bootstrap güven aralıkları üretir."""
    # Her süpernova ve her model için bir MAE değeri bulunur.
    # Örtüşen pencereleri bağımsız kabul etmekten kaçınmak için
    # bootstrap örneklemesi süpernova kimliği üzerinden yapılır.
    objects = sorted(object_frame["Name"].unique())
    pivot = object_frame.pivot(
        index="Name", columns="model", values="MAE_counts_per_s"
    ).loc[objects]

    rng = np.random.default_rng(42)
    comparisons = [
        ("Baseline", "GRU"),
        ("Baseline", "CNN"),
        ("CNN", "GRU"),
    ]
    result = {
        "scope": tag,
        "n_independent_test_objects": len(objects),
        "method": "Paired cluster bootstrap, resampling supernova names (n=3000)",
        "comparisons": [],
    }

    for a, b in comparisons:
        # Pozitif fark, B modelinin daha düşük MAE'ye sahip olduğunu gösterir.
        differences = (pivot[a] - pivot[b]).to_numpy()
        boots = rng.choice(
            differences,
            size=(3000, len(differences)),
            replace=True,
        ).mean(axis=1)
        lo, hi = np.quantile(boots, [0.025, 0.975])

        result["comparisons"].append({
            "A_minus_B": f"{a} - {b}",
            "mean_MAE_advantage_counts_per_s": float(np.mean(differences)),
            "n_objects_B_better": int(np.sum(differences > 0)),
            "n_objects_A_better": int(np.sum(differences < 0)),
            "CI_95pct": [float(lo), float(hi)],
        })

    (RESULTS / f"model_bootstrap_{tag}.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return result


def evaluate_horizon(df, horizon_bins):
    """Tek bir tahmin ufku için modelleri aynı test verisinde değerlendirir."""
    tag = f"{horizon_bins * 6}h"
    print(f"\n===== {tag} =====")

    # İlgili tahmin süresinin model ağırlıkları kaydedilmiş olmalıdır.
    for filename, _ in MODEL_CONFIG[horizon_bins].values():
        if not (RESULTS / filename).is_file():
            raise FileNotFoundError(
                f"Missing {RESULTS / filename}. First run python src/train_extra.py --all "
                "and for the original 24h GRU run python src/train_gru.py."
            )

    # 72 saatlik geçmişten 24 veya 48 saatlik gelecek hedefleri oluştur.
    x, y, objects, t0 = make_windows(df, lookback=12, horizon=horizon_bins)
    split = split_by_object(objects, seed=42)

    # Aynı süpernovanın hiçbir penceresi farklı kümelere dağılmamalıdır.
    groups = [
        set(objects[split == name])
        for name in ["train", "val", "test"]
    ]
    assert not any(
        groups[i] & groups[j]
        for i in range(3)
        for j in range(i + 1, 3)
    )

    mask = split == "test"
    test_x, truth, names, times = x[mask], y[mask], objects[mask], t0[mask]

    # Normalizasyon parametreleri yalnızca geçmiş girişten hesaplanır.
    # Tahminler sonra özgün counts/s ölçeğine geri dönüştürülür.
    mu, sd = scaling_from_history(test_x)
    scaled_input = ((test_x - mu) / sd).astype("float32")

    # Baseline, son geçmiş ölçümünü bütün gelecek adımlara kopyalar.
    predictions = {
        "Baseline": np.repeat(test_x[:, -1:], horizon_bins, axis=1)
    }

    for model_name, (filename, constructor) in MODEL_CONFIG[horizon_bins].items():
        pred_norm = predict_model(
            RESULTS / filename, constructor, scaled_input
        )
        pred = pred_norm * sd + mu

        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(
                f"{model_name} predictions: invalid dimensions or nonfinite values."
            )
        predictions[model_name] = pred

    # Bütün test pencereleri bir arada değerlendirilir.
    summary = []
    for model_name, pred in predictions.items():
        mae, rmse = summarize(truth, pred)
        summary.append({
            "forecast_h": horizon_bins * 6,
            "model": model_name,
            "n_test_supernova": len(groups[2]),
            "n_test_windows": len(truth),
            "MAE_counts_per_s": mae,
            "RMSE_counts_per_s": rmse,
        })
    summary_frame = pd.DataFrame(summary)

    # 24 saatlik eski deney kaydı varsa aynı test protokolünün
    # ve aynı model ağırlıklarının kullanıldığını MAE üzerinden doğrula.
    prior_path = RESULTS / "gru_vs_baseline.csv"
    if horizon_bins == 4 and prior_path.is_file():
        prior = pd.read_csv(prior_path)
        test_prior = prior[prior["split"] == "test"]

        if len(test_prior) != 1:
            raise ValueError("Expected exactly one original GRU test row.")

        old = test_prior.iloc[0]
        current = summary_frame.set_index("model")
        for model_name, key in [
            ("GRU", "GRU_MAE_ct_per_sec"),
            ("Baseline", "baseline_MAE_ct_per_sec"),
        ]:
            actual = float(current.loc[model_name, "MAE_counts_per_s"])
            expected = float(old[key])
            if not np.isclose(actual, expected, atol=0.0003, rtol=0.0002):
                raise RuntimeError(
                    f"Test protocol mismatch for {model_name}: original={expected:.6f}, "
                    f"new={actual:.6f}. Stop: check dataset, seed and saved weights."
                )

        print("  Original 24h GRU and baseline test metrics match: OK")

    summary_frame.to_csv(RESULTS / f"model_summary_{tag}.csv", index=False)
    print(summary_frame.to_string(index=False))

    # 6, 12, 18, ... saatlik her gelecek adımı için ayrı hata hesabı.
    horizons = []
    for step in range(horizon_bins):
        for model_name, pred in predictions.items():
            mae, rmse = summarize(truth[:, step], pred[:, step])
            horizons.append({
                "horizon_h": 6 * (step + 1),
                "model": model_name,
                "MAE_counts_per_s": mae,
                "RMSE_counts_per_s": rmse,
            })
    horizons_frame = pd.DataFrame(horizons)
    horizons_frame.to_csv(RESULTS / f"model_horizon_{tag}.csv", index=False)

    # Süpernova başına metrikler, aynı nesneye ait pencereler birleştirilerek
    # hesaplanır. Böylece nesne-bazlı karşılaştırma yapılabilir.
    object_rows = []
    for object_name in np.unique(names):
        i = names == object_name
        for model_name, pred in predictions.items():
            mae, rmse = summarize(truth[i], pred[i])
            object_rows.append({
                "Name": object_name,
                "model": model_name,
                "n_windows": int(i.sum()),
                "MAE_counts_per_s": mae,
                "RMSE_counts_per_s": rmse,
            })
    object_frame = pd.DataFrame(object_rows)
    object_frame.to_csv(RESULTS / f"model_object_{tag}.csv", index=False)

    # Üç çiftli model karşılaştırması için güven aralığı hesapla.
    boots = cluster_bootstrap(object_frame, tag)
    for item in boots["comparisons"]:
        print(
            f'  {item["A_minus_B"]} = {item["mean_MAE_advantage_counts_per_s"]:.4f}'
            f' counts/s; 95% cluster CI={item["CI_95pct"]}; B better on'
            f' {item["n_objects_B_better"]}/{len(groups[2])} objects'
        )

    # Deney grafikleri ve tekrar üretim bilgilerini kaydet.
    plot_horizon(horizons_frame, tag)
    plot_examples(test_x, truth, predictions, names, times, tag)
    (RESULTS / f"model_protocol_{tag}.json").write_text(
        json.dumps({
            "dataset": "Fausnaugh et al 2023, NASA TESS, VizieR J/ApJ/956/108/table2",
            "features": ["CRate"],
            "time_index": "BJD",
            "target": "CRate",
            "bin_width_h": 6,
            "lookback_bins": 12,
            "horizon_bins": horizon_bins,
            "max_windows_per_object": 60,
            "split_seed": 42,
            "split_strategy": "By supernova Name, 70/15/15 proportions",
            "n_train_objects": len(groups[0]),
            "n_val_objects": len(groups[1]),
            "n_test_objects": len(groups[2]),
            "n_test_windows": len(truth),
            "normalization": "Per-window, history-only mean and scale",
            "interpretation": (
                "Forecast from any fully observed light-curve segment, "
                "not just early explosion phase"
            ),
            "limitation": (
                "Windows overlap within objects. One split and model seed; "
                "no causal/physical discovery claim."
            ),
        }, indent=2),
        encoding="utf-8",
    )
    return summary_frame


def main():
    """Veriyi bir kez okuyarak 24 ve 48 saatlik deneyleri çalıştırır."""
    RESULTS.mkdir(parents=True, exist_ok=True)
    df = load_photometry()
    combined = []

    for horizon in [4, 8]:
        combined.append(evaluate_horizon(df, horizon))

    pd.concat(combined, ignore_index=True).to_csv(
        RESULTS / "model_summary_all.csv", index=False
    )
    print(
        "\nSaved model_summary_all.csv, horizon plots, "
        "example plots and cluster bootstrap tables."
    )
    print(
        "Note: compare 24h and 48h within their own held-out cohorts; "
        "eligible windows may differ."
    )


if __name__ == "__main__":
    main()
