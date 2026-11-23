
"""
Cosmic Forecast - Veri Yükleme ve Ön İşleme

Bu dosya, NASA TESS tarafından gözlemlenen Tip Ia süpernovalarının
fotometrik ölçümlerini yapay sinir ağı modellerine uygun hâle getirir.

Veri kaynağı:
    VizieR - J/ApJ/956/108/table2
    Fausnaugh et al. (2023)

Temel işlemler:
    1. Fotometrik ölçümlerin yüklenmesi ve temizlenmesi
    2. Işık eğrilerinin 6 saatlik zaman aralıklarına dönüştürülmesi
    3. Geçmiş gözlem ve gelecek tahmin pencerelerinin oluşturulması
    4. Verilerin süpernova kimliğine göre ayrılması
    5. Yalnızca geçmiş gözlemlerle normalizasyon yapılması

Ham astronomik veriler bu kaynak kodla birlikte dağıtılmaz.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------
# Proje dizinleri
# --------------------------------------------------

# Projenin ana klasörünü belirle.
ROOT = Path(__file__).resolve().parents[1]

# NASA TESS fotometrik veri dosyalarının konumları.
DATA_FILE = ROOT / "data" / "raw" / "table2.dat"
TSV_FILE = ROOT / "data" / "raw" / "table2.tsv"

# Model sonuçlarının ve grafiklerin kaydedileceği klasör.
RESULTS = ROOT / "results"


# --------------------------------------------------
# 1. Fotometrik verilerin yüklenmesi
# --------------------------------------------------

def load_photometry(path=DATA_FILE):
    """
    NASA TESS fotometrik ölçümlerini yükler ve temizler.

    Desteklenen dosya biçimleri:
        - TSV: VizieR ASU sorgu arayüzünden indirilen veri
        - DAT: Sabit sütun genişliğine sahip orijinal ASCII veri

    Döndürür:
        pandas.DataFrame:
            Name    : Süpernova kimliği
            BJD     : Gözlem zamanı
            CRate   : Işık akısı (counts/s)
            e_CRate : Ölçüm belirsizliği
    """

    path = Path(path)

    # Varsayılan veri yolu kullanıldığında TSV dosyası mevcutsa
    # onu tercih et. Böylece yeni indirme yöntemiyle uyum sağlanır.
    if path == DATA_FILE and TSV_FILE.exists():
        path = TSV_FILE

    # Veri dosyasının mevcut olup olmadığını kontrol et.
    if not path.exists():
        raise FileNotFoundError(
            f"Veri dosyasi yok: {path}. "
            "Once python src/download_data.py calistir."
        )

    # TSV biçimindeki verilerin okunması.
    if path.suffix.lower() == ".tsv":
        df = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="error"
        )

        # Sütun isimlerinin başındaki ve sonundaki
        # gereksiz boşlukları temizle.
        df.columns = df.columns.str.strip()

        required = ["Name", "BJD", "CRate", "e_CRate"]

        # Veri setinde gerekli sütunların bulunduğunu doğrula.
        if not set(required).issubset(df.columns):
            raise ValueError(
                f"Beklenen TSV sutunlari yok. "
                f"Gelen: {list(df.columns)}"
            )

        # Modelleme için gerekli sütunları seç.
        df = df[required].copy()

    else:
        # Orijinal DAT dosyasında veriler sabit sütun
        # genişlikleriyle saklanır.
        #
        # Sütun konumları CDS ReadMe dosyasına dayanır:
        # Name    : 1-10
        # BJD     : 12-21
        # CRate   : 23-30
        # e_CRate : 32-38

        df = pd.read_fwf(
            path,
            colspecs=[
                (0, 10),
                (11, 21),
                (22, 30),
                (31, 38)
            ],
            names=["Name", "BJD", "CRate", "e_CRate"],
            dtype={"Name": "string"}
        )

    # Sayısal sütunları uygun veri tipine dönüştür.
    # Dönüştürülemeyen değerler NaN olarak işaretlenir.
    for col in ["BJD", "CRate", "e_CRate"]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    # Süpernova kimliklerindeki gereksiz boşlukları kaldır.
    df["Name"] = df["Name"].astype("string").str.strip()

    # Kimliği, zamanı, ışık akısı veya ölçüm belirsizliği
    # eksik olan kayıtları veri setinden çıkar.
    df = df.dropna(
        subset=["Name", "BJD", "CRate", "e_CRate"]
    )

    # Ölçüm belirsizliği sıfırdan büyük olmalıdır.
    df = df[df["e_CRate"] > 0].copy()

    # Her süpernovanın ölçümlerini kronolojik sıraya koy.
    df = df.sort_values(
        ["Name", "BJD"]
    ).reset_index(drop=True)

    # Temizlik sonrasında veri kalıp kalmadığını kontrol et.
    if df.empty:
        raise ValueError(
            "Veri bos: tablo bicimini ve "
            "indirme islemini kontrol edin."
        )

    return df


# --------------------------------------------------
# 2. Zaman serisi pencerelerinin oluşturulması
# --------------------------------------------------

def make_windows(
    df,
    lookback=12,
    horizon=4,
    max_per_object=60
):
    """
    Süpernova ışık eğrilerini tahmin pencerelerine dönüştürür.

    Her zaman adımı 6 saati temsil eder.

    Varsayılan parametreler:
        lookback = 12 -> 72 saatlik geçmiş
        horizon = 4  -> 24 saatlik gelecek

    Eksik zaman aralıkları doldurulmaz.
    Yalnızca tüm ölçümleri mevcut pencereler kullanılır.

    Döndürür:
        xs      : Geçmiş gözlem pencereleri
        ys      : Gelecek hedef değerleri
        objects : Süpernova kimlikleri
        t0s     : Tahminin başladığı gözlem zamanı
    """

    xs = []
    ys = []
    objects = []
    t0s = []

    # Her süpernovanın ışık eğrisini ayrı işle.
    for name, g in df.groupby("Name", sort=False):

        # Gözlemleri zamana göre sırala.
        g = g.sort_values("BJD")

        # BJD zamanlarını 6 saatlik aralıklara dönüştür.
        # Bir gün 4 adet 6 saatlik zaman diliminden oluşur.
        #
        # Küçük sabit, kayan nokta hassasiyetinden
        # kaynaklanabilecek sınır hatalarını azaltmak içindir.
        bins = np.floor(
            g["BJD"].to_numpy() * 4.0 + 1e-7
        ).astype(np.int64)

        # Zaman dilimleriyle ışık akısı değerlerini eşleştir.
        small = pd.DataFrame({
            "bin": bins,
            "flux": g["CRate"].to_numpy()
        })

        # Aynı 6 saatlik dilimdeki ölçümlerin medyanını al.
        # Medyan, uç değerlere ortalamadan daha dayanıklıdır.
        series = small.groupby("bin")["flux"].median()

        # İlk ve son gözlem arasındaki bütün zaman
        # dilimlerini oluştur.
        idx = np.arange(
            series.index.min(),
            series.index.max() + 1
        )

        # Ölçüm bulunmayan zaman dilimleri NaN kalır.
        binned = series.reindex(idx).to_numpy(
            dtype=np.float32
        )

        valid = []

        # Geçmiş ve gelecek ölçümleri eksiksiz olan
        # zaman pencerelerini belirle.
        for i in range(
            lookback,
            len(binned) - horizon + 1
        ):
            chunk = binned[
                i - lookback:i + horizon
            ]

            # Eksik veya geçersiz değer içeren
            # pencereler kullanılmaz.
            if np.isfinite(chunk).all():
                valid.append(i)

        # Bir süpernovadan çok fazla pencere üretilmesini
        # sınırlandır. Böylece uzun gözlemlere sahip
        # süpernovaların veri setindeki ağırlığı azaltılır.
        if max_per_object and len(valid) > max_per_object:

            # Pencereleri gözlem süresine yaklaşık
            # eşit aralıklarla dağıtarak seç.
            chosen = np.linspace(
                0,
                len(valid) - 1,
                max_per_object
            ).round().astype(int)

            valid = [valid[j] for j in chosen]

        # Uygun pencereleri model giriş ve hedeflerine ayır.
        for i in valid:

            # Model girdisi: Geçmiş gözlemler.
            xs.append(
                binned[i - lookback:i]
            )

            # Tahmin hedefi: Gelecek gözlemler.
            ys.append(
                binned[i:i + horizon]
            )

            # Pencerenin ait olduğu süpernovayı kaydet.
            objects.append(str(name))

            # Tahmin başlangıç zamanını BJD cinsinden kaydet.
            t0s.append(float(idx[i] / 4.0))

    # Hiç uygun pencere bulunamazsa işlemi durdur.
    if not xs:
        raise ValueError(
            "Hic tam pencere cikmadi. "
            "Veri kapsam/kalitesini inceleyin veya "
            "tahmin parametrelerini degistirin."
        )

    # Listeleri NumPy dizilerine dönüştür.
    return (
        np.stack(xs),
        np.stack(ys),
        np.asarray(objects),
        np.asarray(t0s)
    )


# --------------------------------------------------
# 3. Eğitim, doğrulama ve test veri ayrımı
# --------------------------------------------------

def split_by_object(objects, seed=42):
    """
    Verileri süpernova kimliklerine göre üç kümeye ayırır.

    Yaklaşık dağılım:
        %70 Eğitim
        %15 Doğrulama
        %15 Test

    Aynı süpernovanın farklı zaman pencereleri
    farklı kümelere dağıtılmaz.

    seed=42 kullanılarak tekrarlanabilir bir
    rastgele bölme işlemi gerçekleştirilir.
    """

    # Benzersiz süpernova kimliklerini elde et.
    ids = np.unique(objects)

    if len(ids) < 10:
        raise ValueError(
            f"Bagimsiz supernova sayisi cok az ({len(ids)}). "
            "En az 10 ile baslayin."
        )

    # Sabit tohum ile tekrarlanabilir rastgele sıralama.
    rng = np.random.default_rng(seed)
    ids = rng.permutation(ids)

    # Eğitim ve doğrulama kümesi büyüklüklerini hesapla.
    train_n = int(len(ids) * 0.7)
    val_n = int(len(ids) * 0.15)

    # Süpernova kimliklerini ilgili kümelere dağıt.
    train_ids = set(ids[:train_n])

    val_ids = set(
        ids[train_n:train_n + val_n]
    )

    test_ids = set(
        ids[train_n + val_n:]
    )

    # Kümeler arasında ortak süpernova kimliği
    # bulunmadığını doğrula.
    assert not (
        train_ids & val_ids
        or train_ids & test_ids
        or val_ids & test_ids
    )

    # Her zaman penceresine ait küme etiketini oluştur.
    arr = np.array([
        "train" if a in train_ids
        else "val" if a in val_ids
        else "test"
        for a in objects
    ])

    return arr


# --------------------------------------------------
# 4. Geçmiş gözlemlere dayalı normalizasyon
# --------------------------------------------------

def scaling_from_history(x):
    """
    Her tahmin penceresi için normalizasyon
    parametrelerini yalnızca geçmiş gözlemlerden hesaplar.

    Gelecekteki gerçek ölçümler kullanılmadığı için
    hedef verilerden normalizasyon kaynaklı bilgi sızması
    engellenir.

    Döndürür:
        mu    : Geçmiş pencerenin ortalaması
        scale : Normalizasyon ölçeği
    """

    # Her pencerenin geçmiş gözlemlerinin ortalaması.
    mu = x.mean(
        axis=1,
        keepdims=True
    )

    # Her pencerenin geçmiş gözlemlerinin standart sapması.
    sd = x.std(
        axis=1,
        keepdims=True
    )

    # Standart sapmanın çok küçük olduğu durumlarda
    # aşırı ölçeklemeyi önlemek için alt sınır kullanılır.
    #
    # Ölçek; standart sapma, ortalamanın %10'u
    # ve 1.0 değerlerinden en büyüğü olarak belirlenir.
    scale = np.maximum(
        sd,
        np.maximum(
            0.1 * np.abs(mu),
            1.0
        )
    )

    return (
        mu.astype("float32"),
        scale.astype("float32")
    )
