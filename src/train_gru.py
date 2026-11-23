"""
Cosmic Forecast - 24 Saatlik GRU Modelinin Eğitimi

NASA TESS ışık eğrilerinin son 72 saatlik gözlemlerinden
gelecek 24 saatlik ışık akısını tahmin eden GRU modeli eğitilir.

Gerekli kütüphaneler: requirements-model.txt
Kullanım: python src/train_gru.py

Önemli: Bu dosyayı çalıştırmak, mevcut gru_model.pt ağırlıklarını
ve gru_vs_baseline.csv sonuçlarını yeniden oluşturup üzerine yazar.
"""

import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dataset import (
    load_photometry,
    make_windows,
    split_by_object,
    scaling_from_history,
    RESULTS,
)


class GRUForecast(nn.Module):
    """72 saatlik (12 adım) girdiden 24 saatlik (4 adım) tahmin üreten GRU."""

    def __init__(self, hidden=32):
        super().__init__()

        # Tek özellikli zaman serisini 32 gizli birimli GRU ile işle.
        self.rnn = nn.GRU(input_size=1, hidden_size=hidden, batch_first=True)

        # Son gizli durumdan gelecek dört adet 6 saatlik değeri tahmin et.
        self.head = nn.Linear(hidden, 4)

    def forward(self, x):
        # Giriş boyutu: (örnek sayısı, 12 zaman adımı, 1 özellik).
        # Son gizli durum bütün geçmiş dizinin öğrenilmiş temsilidir.
        _, hidden = self.rnn(x)
        return self.head(hidden[-1])


def main():
    # Eğitimdeki rastgele işlemleri mümkün olduğunca tekrarlanabilir kıl.
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    # TESS gözlemlerini yükle ve 72 saat -> 24 saat pencereleri oluştur.
    df = load_photometry()
    raw_x, raw_y, objects, _ = make_windows(df, lookback=12, horizon=4)

    # Aynı süpernovaya ait pencereler farklı veri kümelerine geçmez.
    split = split_by_object(objects)

    # Ölçekleme parametreleri yalnızca geçmiş gözlemlerden hesaplanır.
    # Gelecekteki gerçek değerler normalize etme katsayılarını belirlemez.
    mu, sd = scaling_from_history(raw_x)
    x = ((raw_x - mu) / sd).astype("float32")
    y = ((raw_y - mu) / sd).astype("float32")

    # Dizileri PyTorch tensörlerine dönüştür ve mini-batch'lere ayır.
    # Yalnızca eğitim kümesinde örnek sırası karıştırılır.
    loaders = {}
    for phase in ["train", "val", "test"]:
        ix = split == phase
        ds = TensorDataset(
            torch.from_numpy(x[ix]).unsqueeze(-1),
            torch.from_numpy(y[ix]),
        )
        loaders[phase] = DataLoader(
            ds,
            batch_size=128,
            shuffle=phase == "train",
        )

    # Model, Adam optimizasyonu ve ortalama karesel hata kaybı.
    model = GRUForecast(hidden=32)
    optim = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    # En düşük doğrulama kaybını veren ağırlıkları sakla.
    best = float("inf")
    best_weights = None
    patience = 0

    # En fazla 20 epoch eğit; 5 kez iyileşme olmazsa erken durdur.
    for epoch in range(1, 21):
        model.train()

        for xb, yb in loaders["train"]:
            optim.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()

            # Aşırı büyüyen gradyanları sınırlayarak eğitimi kararlı tut.
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        # Model ağırlıklarını değiştirmeden doğrulama kaybını hesapla.
        model.eval()
        losses = []
        with torch.no_grad():
            for xb, yb in loaders["val"]:
                losses.append((loss_fn(model(xb), yb).item(), len(xb)))

        # Son küçük batch'in ağırlığını doğru hesaba katmak için
        # doğrulama kaybı örnek sayısına göre ağırlıklandırılır.
        val_loss = sum(v * n for v, n in losses) / sum(n for _, n in losses)
        print(f"Epoch {epoch:02d}: validation normalized MSE={val_loss:.5f}")

        if val_loss < best - 1e-5:
            best = val_loss
            patience = 0
            best_weights = {
                k: v.detach().clone()
                for k, v in model.state_dict().items()
            }
        else:
            patience += 1
            if patience >= 5:
                break

    # Son epoch yerine en iyi doğrulama sonucunu veren ağırlıkları kullan.
    model.load_state_dict(best_weights)
    model.eval()

    # Doğrulama ve test kümelerini eğitimde kullanılan ölçeklemeyi
    # tersine çevirerek gerçek ışık akısı biriminde (counts/s) değerlendir.
    metrics = []
    with torch.no_grad():
        for phase in ["val", "test"]:
            ix = split == phase
            normalized_pred = model(
                torch.from_numpy(x[ix]).unsqueeze(-1)
            ).numpy()
            pred = normalized_pred * sd[ix] + mu[ix]
            err = raw_y[ix] - pred

            # Persistence baseline: son gözlenen değeri dört adıma kopyala.
            baseline = np.repeat(raw_x[ix, -1:], 4, axis=1)
            ebaseline = raw_y[ix] - baseline

            # MAE: ortalama mutlak hata; RMSE: kareli hataların
            # ortalamasının karekökü. Her ikisi de counts/s biriminde.
            metrics.append({
                "split": phase,
                "n_supernova": int(len(set(objects[ix]))),
                "GRU_MAE_ct_per_sec": float(np.mean(np.abs(err))),
                "GRU_RMSE_ct_per_sec": float(np.sqrt(np.mean(err**2))),
                "baseline_MAE_ct_per_sec": float(np.mean(np.abs(ebaseline))),
                "baseline_RMSE_ct_per_sec": float(
                    np.sqrt(np.mean(ebaseline**2))
                ),
            })

    # Eğitilmiş modeli ve karşılaştırma sonuçlarını kaydet.
    RESULTS.mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), RESULTS / "gru_model.pt")
    pd.DataFrame(metrics).to_csv(RESULTS / "gru_vs_baseline.csv", index=False)

    print(pd.DataFrame(metrics).to_string(index=False))
    print("Kaydedildi: results/gru_model.pt, results/gru_vs_baseline.csv")


if __name__ == "__main__":
    main()
