
"""
Cosmic Forecast - Model Kapasitesi ve Çıkarım Maliyeti

Bu dosya, eğitilmiş CNN ve GRU modellerinin:

1. Eğitilebilir parametre sayılarını
2. Model ağırlık dosyalarının boyutlarını
3. CPU üzerindeki tahmin sürelerini
4. Kullanılan PyTorch ve Python sürümlerini

karşılaştırır.

Model eğitimi yapılmaz.
Mevcut ağırlık dosyaları değiştirilmez.

Kullanım:
    python src/analyze_capacity.py
"""

import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import torch

from dataset import RESULTS
from models_extra import CNNForecast, GRUForecastFlexible
from train_gru import GRUForecast


# Her modelin mimarisi ve kayıtlı ağırlık dosyası.
MODELS = {
    "GRU_24h": (
        lambda: GRUForecast(hidden=32),
        "gru_model.pt"
    ),
    "CNN_24h": (
        lambda: CNNForecast(horizon=4, lookback=12),
        "cnn_24h_model.pt"
    ),
    "GRU_48h": (
        lambda: GRUForecastFlexible(horizon=8, hidden=32),
        "gru_48h_model.pt"
    ),
    "CNN_48h": (
        lambda: CNNForecast(horizon=8, lookback=12),
        "cnn_48h_model.pt"
    ),
}


def count_parameters(model):
    """Modeldeki eğitilebilir parametreleri sayar."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def measure_inference(model, batch_size, repetitions=200):
    """
    CPU üzerinde tahmin süresini ölçer.

    Ölçümden önce 30 ısınma çalıştırması yapılır.
    Tekrarların ortancası kullanılır.

    Sonuç batch başına milisaniye cinsindendir.
    """

    model.eval()

    # Bütün modeller 12 zaman adımlı tek özellik kullanır.
    x = torch.randn(
        batch_size,
        12,
        1,
        dtype=torch.float32
    )

    timings = []

    with torch.inference_mode():

        # İlk çalıştırmalardaki ek yükü azalt.
        for _ in range(30):
            model(x)

        # Tekrarlı CPU çıkarım süresi ölçümü.
        for _ in range(repetitions):

            start = time.perf_counter()

            model(x)

            elapsed_ms = (
                time.perf_counter() - start
            ) * 1000

            timings.append(elapsed_ms)

    return float(np.median(timings))


def main():

    RESULTS.mkdir(parents=True, exist_ok=True)

    # Karşılaştırma için bütün modeller CPU'da çalışır.
    # Tek iş parçacığı kullanılarak ölçüm koşulu sabitlenir.
    torch.set_num_threads(1)

    torch.manual_seed(42)

    rows = []

    for name, (constructor, filename) in MODELS.items():

        path = RESULTS / filename

        if not path.is_file():
            raise FileNotFoundError(
                f"Model dosyasi bulunamadi: {path}"
            )

        # Mevcut model mimarisini oluştur.
        model = constructor()

        # Eğitilmiş ağırlıkları yükle.
        state = torch.load(
            path,
            map_location="cpu",
            weights_only=True
        )

        model.load_state_dict(state)

        model.eval()

        # Eğitilebilir parametre sayısı.
        params = count_parameters(model)

        # Kaydedilmiş ağırlık dosyasının boyutu.
        size_kb = path.stat().st_size / 1024

        # Tek pencere için tahmin süresi.
        latency_1 = measure_inference(
            model,
            batch_size=1
        )

        # 128 pencerelik batch tahmin süresi.
        latency_128 = measure_inference(
            model,
            batch_size=128
        )

        rows.append({
            "model": name,
            "trainable_parameters": params,
            "checkpoint_size_KB": round(size_kb, 2),
            "latency_batch1_ms": round(latency_1, 4),
            "latency_batch128_ms": round(latency_128, 4),
            "latency_per_window_batch128_ms": round(
                latency_128 / 128,
                6
            ),
        })

    result = pd.DataFrame(rows)

    # Model kapasitesi ve çıkarım süresi tablosu.
    result.to_csv(
        RESULTS / "model_capacity.csv",
        index=False
    )

    # Deney ortamının teknik özellikleri.
    environment = {
        "python_version": sys.version,
        "pytorch_version": torch.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "device": "CPU",
        "torch_num_threads": torch.get_num_threads(),
        "benchmark_repetitions": 200,
        "benchmark_warmup": 30,
        "note": (
            "Only inference latency is measured. "
            "Training duration is not measured."
        ),
    }

    (
        RESULTS / "capacity_environment.json"
    ).write_text(
        json.dumps(
            environment,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    print("\n===== MODEL KAPASITESI =====")
    print(result.to_string(index=False))

    print("\nPyTorch:", torch.__version__)
    print("Python:", platform.python_version())
    print("Platform:", platform.platform())

    print(
        "\nKaydedildi: results/model_capacity.csv "
        "ve results/capacity_environment.json"
    )


if __name__ == "__main__":
    main()
