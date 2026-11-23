
"""
Cosmic Forecast - Yapay Sinir Ağı Mimarileri

Bu dosya, süpernova ışık eğrilerinin kısa vadeli tahmini için
kullanılan CNN ve GRU modellerini tanımlar.

Modeller:
    1. CNNForecast:
       Işık eğrilerindeki yerel zamansal örüntüleri
       öğrenmek için 1 boyutlu evrişimli sinir ağı kullanır.

    2. GRUForecastFlexible:
       Geçmiş gözlemler arasındaki zamansal ilişkileri
       öğrenmek için GRU tabanlı tekrarlayan sinir ağı kullanır.

Her iki model de 6 saatlik zaman adımlarından oluşan
ışık akısı dizilerini kullanarak gelecek değerleri tahmin eder.

Not:
    Daha önce eğitilmiş 24 saatlik GRU modelinin mimarisi
    train_gru.py dosyasında korunmaktadır.
"""

from torch import nn


# --------------------------------------------------
# 1. Bir Boyutlu Evrişimli Sinir Ağı (1D-CNN)
# --------------------------------------------------

class CNNForecast(nn.Module):
    """
    Süpernova ışık eğrilerinin tahmini için 1D-CNN modeli.

    Mimari:
        Giriş -> Conv1D -> ReLU -> Conv1D -> ReLU
              -> Flatten -> Linear -> ReLU -> Linear

    Parametreler:
        horizon  : Tahmin edilecek gelecek zaman adımı sayısı
        lookback : Geçmiş gözlem adımı sayısı

    Giriş boyutu:
        (batch_size, lookback, 1)

    Çıkış boyutu:
        (batch_size, horizon)
    """

    def __init__(self, horizon=4, lookback=12):
        super().__init__()

        # Evrişim katmanları, geçmiş ışık akısı değerlerindeki
        # kısa süreli değişimleri ve yerel örüntüleri öğrenir.
        self.conv = nn.Sequential(

            # İlk evrişim katmanı:
            # Tek özellikli girişten 16 özellik haritası üretir.
            # kernel_size=3 ile üç komşu zaman adımı incelenir.
            # padding=1, zaman boyutunun korunmasını sağlar.
            nn.Conv1d(
                1, 16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),

            # İkinci evrişim katmanı:
            # Öğrenilen 16 özellik haritasından
            # 32 yeni özellik haritası oluşturur.
            nn.Conv1d(
                16, 32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
        )

        # Evrişim katmanlarının çıkışını gelecek
        # ışık akısı tahminlerine dönüştüren katmanlar.
        self.head = nn.Sequential(

            # Özellik haritalarını tek boyutlu vektöre dönüştür.
            nn.Flatten(),

            # 32 özellik haritası ve lookback zaman adımından
            # oluşan vektörü 64 nöronlu gizli katmana aktar.
            nn.Linear(32 * lookback, 64),
            nn.ReLU(),

            # Gelecek zaman adımları için tahmin üret.
            nn.Linear(64, horizon),
        )

    def forward(self, x):
        """
        CNN modelinin ileri yayılım işlemi.

        x:
            (batch_size, lookback, 1)

        Conv1d katmanı:
            (batch_size, channels, lookback)

        Çıkış:
            (batch_size, horizon)
        """

        # Conv1d, kanal boyutunu zaman boyutundan önce bekler.
        # Bu nedenle girişin son iki boyutu yer değiştirilir.
        x = x.transpose(1, 2)

        # Işık eğrisinden zamansal özellikleri çıkar.
        features = self.conv(x)

        # Öğrenilen özelliklerden gelecek değerleri tahmin et.
        return self.head(features)


# --------------------------------------------------
# 2. GRU Tabanlı Tekrarlayan Sinir Ağı
# --------------------------------------------------

class GRUForecastFlexible(nn.Module):
    """
    Geçmiş ışık akısı ölçümlerinden geleceği tahmin eden
    GRU tabanlı yapay sinir ağı.

    GRU (Gated Recurrent Unit), zaman serilerindeki
    geçmiş bilgilerin hangilerinin korunacağını ve
    hangilerinin güncelleneceğini öğrenir.

    Parametreler:
        horizon : Gelecek zaman adımı sayısı
        hidden  : GRU gizli durum vektörünün boyutu

    Giriş boyutu:
        (batch_size, sequence_length, 1)

    Çıkış boyutu:
        (batch_size, horizon)
    """

    def __init__(self, horizon=8, hidden=32):
        super().__init__()

        # GRU katmanı, ışık eğrisindeki zamansal
        # bağımlılıkları öğrenmek için kullanılır.
        self.rnn = nn.GRU(

            # Her zaman adımında tek bir özellik bulunur:
            # Normalize edilmiş ışık akısı değeri.
            input_size=1,

            # GRU'nun gizli durum boyutu.
            hidden_size=hidden,

            # Giriş verilerinin ilk boyutu örnek sayısıdır.
            batch_first=True
        )

        # GRU'nun öğrendiği son gizli durumdan
        # gelecek zaman adımlarına ait tahminleri üretir.
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        """
        GRU modelinin ileri yayılım işlemi.

        x:
            (batch_size, sequence_length, 1)

        Çıkış:
            (batch_size, horizon)
        """

        # GRU, bütün zaman adımlarının çıktılarını
        # ve son gizli durumunu döndürür.
        #
        # Tahmin için yalnızca son gizli durum kullanılır.
        _, hidden = self.rnn(x)

        # hidden[-1], son GRU katmanının gizli durumudur.
        # Geçmiş gözlemlerden öğrenilen bilgileri temsil eder.
        last_hidden = hidden[-1]

        # Son gizli durumu gelecek ışık akısı tahminlerine dönüştür.
        return self.head(last_hidden)
