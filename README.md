# Cosmic Forecast

**NASA TESS verileriyle Tip Ia süpernova ışık eğrilerinin kısa vadeli tahmini**

Kocaeli Üniversitesi · YZM437 Yapay Sinir Ağları Dönem Projesi

## Proje hakkında

Cosmic Forecast, NASA'nın TESS (*Transiting Exoplanet Survey Satellite*) görevi tarafından gözlemlenen Tip Ia süpernovalarının ışık eğrilerinden gelecek 24 ve 48 saatlik ışık akısı değerlerini tahmin etmeyi amaçlayan bir zaman serisi tahmin projesidir.

Bu çalışmada GRU (*Gated Recurrent Unit*) ve 1D-CNN (*One-Dimensional Convolutional Neural Network*) modelleri geliştirilmiştir. Sinir ağlarının başarısını değerlendirmek amacıyla beş klasik tahmin yöntemiyle karşılaştırma yapılmıştır:

- Persistence (Son Değer)
- Historical Mean (Geçmiş Ortalaması)
- Historical Median (Geçmiş Medyanı)
- Linear Trend (Doğrusal Eğilim)
- Ridge Regression
- GRU
- 1D-CNN

Projenin temel araştırma sorusu, **72 saatlik geçmiş fotometrik gözlemler kullanılarak gelecek 24 ve 48 saatlik ışık akısının ne ölçüde tahmin edilebildiği ve sinir ağlarının klasik yöntemlere karşı ne kadar avantaj sağladığıdır.**

Çalışma yalnızca genel tahmin hatalarını karşılaştırmakla sınırlı değildir. Model kapasitesi, hesaplama maliyeti, rastgele eğitim başlangıçları, farklı süpernova veri bölmeleri, geçmiş ışık eğrisi eğilimleri ve fotometrik ölçüm belirsizlikleri de değerlendirilmiştir.

> **Kapsam:** Modeller yeterli ve kesintisiz gözlem içeren ışık eğrisi bölümlerinden tahmin üretmektedir. Mevcut deneyler yalnızca süpernova patlamasının ilk günlerini kapsamamaktadır. Proje gerçek zamanlı bir erken uyarı veya süpernova tespit sistemi değil, arşiv gözlemleri üzerinde gerçekleştirilen bir tahmin çalışmasıdır.

## Veri seti

Veri kaynağı, Fausnaugh ve arkadaşlarının (2023) yayımladığı TESS Tip Ia süpernova fotometri kataloğudur.

- **Kaynak:** Fausnaugh et al. (2023), *Four Years of Type Ia Supernovae Observed by TESS: Early-time Light-curve Shapes and Constraints on Companion Interaction Models*, The Astrophysical Journal, 956(2), 108.
- **DOI:** https://doi.org/10.3847/1538-4357/aceaef
- **Katalog:** [CDS/VizieR – J/ApJ/956/108](https://vizier.cds.unistra.fr/viz-bin/VizieR?-source=J/ApJ/956/108)
- **Kullanılan tablo:** `table2` (`table2.tsv`; `table2.dat` biçimi de desteklenmektedir).

Yerel ortamda incelenen veri kümesi:

| Özellik | Değer |
|---|---:|
| Fotometrik ölçüm | 680.384 |
| Farklı süpernova | 307 |
| Ölçüm alanı | 4 |
| Ölçüm içeren 6 saatlik zaman dilimi | 30.921 |
| Medyan `e_CRate` | 1,0479 counts/s |
| Negatif `CRate` ölçümü | 173.587 |

Kullanılan veri alanları:

| Alan | Açıklama |
|---|---|
| `Name` | Süpernova kimliği |
| `BJD` | Gözlem zamanı |
| `CRate` | Işık akısı (counts/s) |
| `e_CRate` | Fotometrik ölçüm belirsizliği (counts/s) |

`CRate`, model girdisi ve tahmin hedefi olarak kullanılmaktadır. `e_CRate` ise geçersiz kayıtların elenmesinde ve ölçüm kalitesi analizlerinde kullanılmaktadır. Mevcut sinir ağı modellerine doğrudan belirsizlik özelliği olarak verilmemektedir.

Negatif `CRate` ölçümleri otomatik olarak silinmemiştir. Arka plan çıkarılmış fotometrik ölçümlerde negatif değerler oluşabilmektedir.

Katalogdaki `Frac` alanı kullanılmamaktadır. Tepe değerine dayalı normalizasyon, tahmin anında bilinmeyen gelecek bilgilerini içerebileceğinden veri sızıntısı riski oluşturabilir.

## Yöntem

### Veri ön işleme

Bütün modeller için ortak bir veri hazırlama protokolü kullanılmıştır.

1. Fotometrik ölçümler süpernova kimliği ve gözlem zamanına göre sıralanır.
2. Eksik, geçersiz ve pozitif olmayan `e_CRate` değerlerine sahip kayıtlar çıkarılır.
3. `BJD` zamanları 6 saatlik aralıklara dönüştürülür.
4. Aynı zaman dilimindeki `CRate` ölçümlerinin medyanı alınır.
5. Eksik zaman dilimleri doldurulmaz.
6. Eksiksiz geçmiş ve gelecek gözlemler içeren tahmin pencereleri oluşturulur.
7. Tek bir süpernovanın aşırı temsil edilmesini azaltmak için her nesneden en fazla 60 pencere seçilir.

Tahmin yapılandırması:

| Parametre | 24 saat | 48 saat |
|---|---:|---:|
| Zaman dilimi | 6 saat | 6 saat |
| Geçmiş gözlem | 12 adım | 12 adım |
| Geçmiş süre | 72 saat | 72 saat |
| Tahmin adımı | 4 | 8 |
| Tahmin süresi | 24 saat | 48 saat |

### Veri ayrımı

Veriler pencere bazında değil, süpernova kimliğine göre ayrılır.

- Eğitim: yaklaşık %70
- Doğrulama: yaklaşık %15
- Test: yaklaşık %15

Ana deneylerde `split_seed=42` kullanılmaktadır.

Aynı süpernovadan elde edilen bütün pencereler yalnızca bir veri kümesinde bulunabilir. Böylece aynı nesnenin farklı gözlemlerinin hem eğitim hem test kümesine girmesi engellenir.

Ana test bölmesinde:

| Tahmin ufku | Test süpernovası | Test penceresi |
|---|---:|---:|
| 24 saat | 47 | 2.393 |
| 48 saat | 47 | 2.113 |

İki tahmin ufkunda uygun pencere sayıları farklıdır. Bu nedenle hata sonuçları tamamen aynı tahmin örnekleri üzerinden hesaplanmış gibi yorumlanmamalıdır.

### Normalizasyon

Her pencere kendi geçmiş gözlemlerine göre normalize edilir.

Geçmiş pencerenin ortalaması:

\[
\mu = \operatorname{mean}(x)
\]

Ölçek:

\[
s = \max\left(\operatorname{std}(x),\ 0.1|\mu|,\ 1\right)
\]

Normalize edilmiş giriş ve hedef:

\[
\tilde{x} = \frac{x-\mu}{s}
\]

\[
\tilde{y} = \frac{y-\mu}{s}
\]

Normalizasyon parametreleri yalnızca geçmiş 72 saatten hesaplanır. Gelecekteki gerçek gözlemler bu parametrelerin belirlenmesinde kullanılmaz.

Tahminler değerlendirme öncesinde tekrar `counts/s` birimine dönüştürülür.

## Karşılaştırılan tahmin yöntemleri

### Klasik yöntemler

**1. Persistence**

Son gözlenen `CRate` değerinin gelecek bütün zaman adımlarında değişmeden devam edeceğini varsayar.

**2. Historical Mean**

Geçmiş 12 gözlemin aritmetik ortalamasını gelecekteki bütün adımlar için kullanır.

**3. Historical Median**

Geçmiş 12 gözlemin medyanını gelecekteki bütün adımlar için kullanır.

**4. Linear Trend**

Geçmiş 12 gözleme en küçük kareler yöntemiyle doğrusal eğilim uydurur ve bu eğilimi gelecek 4 veya 8 zaman adımına uzatır.

**5. Ridge Regression**

Geçmiş 12 gözlemden gelecekteki 4 veya 8 değeri doğrudan tahmin eden, L2 düzenlileştirmeli çok çıktılı doğrusal regresyon modelidir.

Ridge Regression yalnızca eğitim kümesinde öğrenilir. Düzenlileştirme parametresi (`alpha`), doğrulama kümesindeki normalize edilmiş ortalama karesel hataya göre seçilir.

Ana deneylerde seçilen değerler:

| Tahmin | Seçilen `alpha` |
|---|---:|
| 24 saat | 0,01 |
| 48 saat | 1.000 |

Test kümesi hiperparametre seçiminde kullanılmaz.

### Yapay sinir ağı modelleri

**GRU**

- Tekrarlayan sinir ağı mimarisi
- 32 gizli birim
- Geçmiş 12 zaman adımını işler
- Doğrusal çıkış katmanı üzerinden 4 veya 8 tahmin değeri üretir

**1D-CNN**

- Bir boyutlu evrişim mimarisi
- 16 ve 32 kanallı iki evrişim katmanı
- Çekirdek boyutu: 3
- ReLU aktivasyonu
- Tam bağlantılı katmanlar üzerinden 4 veya 8 tahmin değeri üretir

### Eğitim yapılandırması

Tekrarlı deneylerde kullanılan ortak eğitim ayarları:

| Parametre | Değer |
|---|---|
| Optimizasyon | Adam |
| Öğrenme oranı | 0,001 |
| Batch boyutu | 128 |
| Maksimum epoch | 20 |
| Early stopping patience | 5 |
| Eğitim kaybı | MSE |
| Gradyan normu sınırı | 1,0 |
| Model seçimi | En düşük doğrulama kaybı |

Test verileri model seçimi ve eğitim sırasında kullanılmamaktadır.

## Deney sonuçları

### 1. Genel model karşılaştırması

Ana test bölmesi `split_seed=42` kullanılarak yedi tahmin yöntemi karşılaştırılmıştır.

MAE ve RMSE değerleri özgün `CRate` birimi olan `counts/s` cinsindedir. Düşük hata daha iyi tahmin performansını gösterir.

**24 saatlik tahmin**

| Yöntem | MAE | RMSE |
|---|---:|---:|
| Persistence | 1,6470 | 2,7418 |
| Historical Mean | 2,5892 | 5,4313 |
| Historical Median | 2,5832 | 5,4501 |
| Linear Trend | 1,5186 | 2,5740 |
| Ridge Regression | 1,4839 | 2,4606 |
| GRU | 1,4768 | 2,4476 |
| **1D-CNN** | **1,4004** | **2,2956** |

CNN, Persistence yöntemine göre yaklaşık **%14,97**, Ridge Regression'a göre ise yaklaşık **%5,63** daha düşük MAE elde etmiştir.

**48 saatlik tahmin**

| Yöntem | MAE | RMSE |
|---|---:|---:|
| Persistence | 2,0502 | 3,7560 |
| Historical Mean | 3,0642 | 6,6018 |
| Historical Median | 3,0523 | 6,6059 |
| Linear Trend | 1,8492 | 3,4197 |
| Ridge Regression | 1,8216 | 3,2449 |
| **GRU** | **1,7303** | **3,0727** |
| 1D-CNN | 1,7711 | 3,1628 |

GRU, Persistence yöntemine göre yaklaşık **%15,60**, Ridge Regression'a göre ise yaklaşık **%5,01** daha düşük MAE elde etmiştir.

Bu sonuçlar ana deney koşullarına aittir. Modellerin genel üstünlüğü hakkında yorum yapılırken tekrarlı eğitim ve farklı veri bölmesi analizleri ayrıca dikkate alınmalıdır.

### 2. Model kapasitesi ve hesaplama maliyeti

GRU ve CNN modellerinin eğitilebilir parametre sayıları, ağırlık dosyası büyüklükleri ve CPU çıkarım süreleri ölçülmüştür.

| Model | Parametre | Ağırlık dosyası | Tek pencere tahmini |
|---|---:|---:|---:|
| GRU 24h | 3.492 | 16,61 KB | 0,3459 ms |
| CNN 24h | 26.532 | 107,03 KB | 0,0876 ms |
| GRU 48h | 3.624 | 17,03 KB | 0,3268 ms |
| CNN 48h | 26.792 | 108,03 KB | 0,0917 ms |

Ölçümler Windows 11, Python `3.13.9` ve PyTorch `2.15.0+cpu` ortamında, CPU üzerinde tek iş parçacığı kullanılarak yapılmıştır.

CNN modelleri daha fazla parametre içermesine rağmen ölçülen CPU çıkarım süreleri GRU modellerinden daha düşüktür.

Ancak mimariler arasında belirgin bir kapasite farkı bulunduğundan performans farklılıkları yalnızca ağ mimarisine bağlanamaz.

### 3. Farklı eğitim başlangıçlarıyla tekrarlı deneyler

Model eğitimindeki rastgeleliğin etkisini incelemek için veri bölmesi sabit tutulmuş ve üç farklı eğitim başlangıcı kullanılmıştır.

- Veri bölmesi: `split_seed=42`
- Eğitim başlangıçları: `41`, `42`, `43`
- Model sayısı: 4
- Toplam eğitim: 12

Sonuçlar:

| Model | Ortalama MAE | Standart sapma |
|---|---:|---:|
| GRU 24h | 1,4565 | 0,0200 |
| **CNN 24h** | **1,4100** | 0,0236 |
| GRU 48h | 1,7487 | 0,0173 |
| CNN 48h | 1,7481 | 0,0228 |

CNN, 24 saatlik tahminde üç eğitim başlangıcının tamamında GRU'dan daha düşük MAE üretmiştir.

48 saatlik tahminde ise modellerin sıralaması başlangıca göre değişmiştir. İki modelin ortalama MAE değerleri birbirine çok yakındır.

### 4. Farklı süpernova bölmeleriyle tekrarlı deneyler

Model başarısının eğitim ve test süpernovalarının seçimine duyarlılığını incelemek için üç farklı veri bölmesi kullanılmıştır.

- Veri bölmesi seed değerleri: `21`, `22`, `23`
- Eğitim başlangıcı: `42`
- Model sayısı: 4
- Toplam eğitim: 12

**24 saatlik tahmin**

| Bölme seed | GRU MAE | CNN MAE |
|---|---:|---:|
| 21 | **1,3026** | 1,3034 |
| 22 | 1,0936 | **1,0855** |
| 23 | **1,4212** | 1,4266 |

**48 saatlik tahmin**

| Bölme seed | GRU MAE | CNN MAE |
|---|---:|---:|
| 21 | **1,5023** | 1,5480 |
| 22 | 1,2641 | **1,2540** |
| 23 | 1,7310 | **1,7287** |

Üç bölmenin ortalama sonuçları:

| Model | Ortalama MAE | Standart sapma |
|---|---:|---:|
| GRU 24h | 1,2725 | 0,1659 |
| CNN 24h | 1,2718 | 0,1727 |
| GRU 48h | 1,4992 | 0,2335 |
| CNN 48h | 1,5103 | 0,2396 |

Bu deneyler, model performansının süpernova bölmesine duyarlı olduğunu göstermektedir.

Toplamda **24 tekrarlı sinir ağı eğitimi** gerçekleştirilmiştir.

Sabit bölme deneyleri eğitim başlangıcının etkisini, farklı bölme deneyleri ise veri seçiminin etkisini incelemektedir. Bunlar birbirinden ayrı deney protokolleridir.

### 5. Eşleştirilmiş model karşılaştırması

GRU ve CNN arasındaki hata farkları süpernova kimliği düzeyinde eşleştirilerek değerlendirilmiştir.

Aynı süpernovaya ait tahmin pencereleri ilişkili olduğundan bağımsız örnekler olarak kabul edilmemiştir.

Her deney koşulunda süpernova bazlı MAE farkı için 3.000 tekrarlı bootstrap uygulanmıştır.

Analiz sonucunda, 12 GRU–CNN karşılaştırmasının 3 tanesinde hesaplanan %95 bootstrap güven aralığı sıfırı dışlamıştır.

Bu sonuçlar, model sıralamasının veri bölmesine ve eğitim başlangıcına bağlı değişebildiğini desteklemektedir.

Güven aralıkları keşifsel amaçlıdır. Çoklu karşılaştırma düzeltmesi uygulanmamıştır ve farklı deney koşullarının tamamında tek bir mimarinin üstün olduğunu göstermemektedir.

### 6. Işık eğrisi eğilim analizi

Modellerin farklı geçmiş gözlem davranışlarındaki performansını incelemek için geçmiş 72 saatlik pencereler üç gruba ayrılmıştır:

- **Artan:** Pozitif doğrusal eğilim
- **Azalan:** Negatif doğrusal eğilim
- **Yatay:** Belirlenen eşik altında kalan doğrusal eğilim

Sınıflandırmada yalnızca geçmiş gözlemler kullanılmaktadır. Normalize edilmiş geçmiş pencerenin 11 aralığa yayılan doğrusal değişim büyüklüğü için `0,5` eşiği kullanılmıştır.

Her tahmin ufkunda elde edilen en düşük pencere ağırlıklı MAE değerleri:

| Geçmiş eğilim | 24 saatlik en iyi yöntem | 48 saatlik en iyi yöntem |
|---|---|---|
| Artan | CNN — 1,8533 | GRU — 2,2266 |
| Azalan | CNN — 1,6933 | Persistence — 2,4714 |
| Yatay | CNN — 0,7598 | GRU — 0,8427 |

Özellikle 48 saatlik azalan eğilim grubunda Persistence yöntemi diğer karşılaştırılan yöntemlerden daha düşük MAE elde etmiştir.

Bu durum, sinir ağlarının bütün gözlemsel koşullarda üstün performans göstermediğine işaret etmektedir.

**Önemli:** Bu sınıflar süpernovanın fiziksel yükseliş, tepe veya sönüm evrelerini temsil etmez. Yalnızca geçmiş gözlem penceresinin sayısal eğilimini tanımlar.

### 7. Fotometrik ölçüm belirsizliği analizi

Model performansının fotometrik ölçüm belirsizliğiyle ilişkisini incelemek için `e_CRate` alanı kullanılmıştır.

Her 6 saatlik zaman diliminde ölçüm belirsizliklerinin medyanı alınmış, ardından her tahmin penceresinin geçmiş 12 zaman dilimindeki medyan belirsizliği hesaplanmıştır.

Bu değer, binlenmiş ışık akısının istatistiksel olarak yayılmış hata tahmini değil, bir **ölçüm kalitesi göstergesidir**.

Kalite gruplarının eşikleri yalnızca eğitim pencerelerinin dağılımındaki üçte birlik dilimlerden belirlenmiştir.

| Grup | 24 saat eşikleri | 48 saat eşikleri |
|---|---|---|
| Düşük | ≤ 1,0027 | ≤ 1,0140 |
| Orta | 1,0027–3,9842 | 1,0140–4,0102 |
| Yüksek | > 3,9842 | > 4,0102 |

Eşiklerin birimi `counts/s` değeridir.

**24 saatlik tahmin sonuçları**

| Belirsizlik | Persistence MAE | Ridge MAE | GRU MAE | CNN MAE |
|---|---:|---:|---:|---:|
| Düşük | 0,2549 | 0,2358 | **0,2154** | 0,2176 |
| Orta | 1,1245 | 1,0966 | 1,1217 | **1,0076** |
| Yüksek | 2,7119 | 2,3883 | 2,3684 | **2,2735** |

**48 saatlik tahmin sonuçları**

| Belirsizlik | Persistence MAE | Ridge MAE | GRU MAE | CNN MAE |
|---|---:|---:|---:|---:|
| Düşük | 0,2815 | 0,2864 | 0,2376 | **0,2367** |
| Orta | 1,3762 | 1,4117 | **1,3622** | 1,4119 |
| Yüksek | 3,4785 | 2,9475 | **2,8051** | 2,8635 |

Yüksek belirsizlik grubunda:

- 24 saatlik CNN, Persistence'a göre **%16,17** daha düşük MAE elde etmiştir.
- 48 saatlik GRU, Persistence'a göre **%19,36** daha düşük MAE elde etmiştir.

Bununla birlikte hata düzeylerinin belirsizlik grupları arasında değişmesi yalnızca ölçüm gürültüsüne bağlanamaz. Süpernovaların parlaklıkları ve gözlemsel özellikleri de farklı olabilir.

### 8. Ölçüm kalitesi bootstrap analizi

Kalite gruplarında model farklarının istatistiksel kararlılığını değerlendirmek için iki tamamlayıcı yaklaşım kullanılmıştır:

1. **Nesneye eşit ağırlıklı bootstrap:** Her süpernovanın grup içindeki ortalama MAE değeri eşit ağırlıkla değerlendirilir.
2. **Pencere ağırlıklı küme bootstrap:** Süpernovalar yeniden örneklenirken aynı süpernovanın bütün pencereleri birlikte korunur. Böylece ana MAE tablolarıyla uyumlu ağırlıklı farklar elde edilir.

Her iki analizde 5.000 bootstrap tekrarı uygulanmıştır.

Pencere ağırlıklı analizden seçilen sonuçlar:

| Tahmin / Belirsizlik | Karşılaştırma | MAE farkı | %95 güven aralığı |
|---|---|---:|---|
| 24h / Düşük | Persistence − GRU | 0,0395 | [0,0283; 0,0499] |
| 24h / Orta | Persistence − CNN | 0,1169 | [0,0505; 0,1780] |
| 24h / Yüksek | Persistence − CNN | 0,4384 | [0,3396; 0,5486] |
| 48h / Düşük | Persistence − CNN | 0,0448 | [0,0342; 0,0613] |
| 48h / Yüksek | Persistence − GRU | 0,6734 | [0,4228; 0,9707] |

Pozitif fark, ikinci modelin daha düşük hata ürettiğini gösterir.

Bu sonuçlar keşifsel değerlendirme niteliğindedir. Güven aralıklarına çoklu karşılaştırma düzeltmesi uygulanmamıştır.

Aynı süpernova farklı kalite gruplarına ait pencereler içerebildiği için kalite grupları tamamen bağımsız nesne toplulukları değildir.

## Deney ortamı

Çalışmalar Windows 11 ortamında Python ve PyTorch kullanılarak gerçekleştirilmiştir.

| Bileşen | Kullanılan ortam |
|---|---|
| İşletim sistemi | Windows 11 |
| Python | 3.13.9 |
| PyTorch | 2.15.0+cpu |
| Model çalıştırma | CPU |
| Geliştirme ortamı | VS Code |
| Temel kütüphaneler | NumPy, pandas, PyTorch, Matplotlib |

Model kapasitesi ölçümleri tek CPU iş parçacığında yapılmıştır. Tekrarlı eğitimlerde en fazla dört PyTorch CPU iş parçacığı kullanılmıştır.

Eğitim süreleri, modellerin farklı epoch'larda erken durması nedeniyle eşit iş yükü altında yapılan bir hız testi olarak yorumlanmamalıdır.

## Kurulum

Proje Python ile geliştirilmiştir. Aşağıdaki komutlar proje ana klasöründe çalıştırılmalıdır.

Windows PowerShell:

```powershell
python --version

python -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements-model.txt
```

`requirements-model.txt`, model eğitimi için gerekli bağımlılıkları yüklemek amacıyla kullanılır.

Matplotlib kurulu değilse:

```powershell
.\.venv\Scripts\python.exe -m pip install matplotlib
```

PyTorch kurulumu için gerekirse [resmî kurulum sayfasındaki](https://pytorch.org/get-started/locally/) işletim sistemine uygun yönergeler kullanılabilir.

Aşağıdaki örneklerde `python` komutu kullanılmaktadır. Komutları çalıştırmadan önce proje için hazırlanan sanal ortam etkinleştirilmelidir. Alternatif olarak `python` yerine `.\.venv\Scripts\python.exe` kullanılabilir.

### 1. Veri indirme ve ilk kontroller

```powershell
python src/download_data.py
python src/inspect_data.py
python src/baseline.py
```

Ham veri dosyaları depoya eklenmez.

İndirme betiği `data/raw/table2.tsv` dosyasını VizieR sunucularından alır. Geçerli bir dosya mevcutsa yeniden indirmeden kullanabilir.

Bağlantı sorunları için `INTERNET_HATASI_COZUMU.md` dosyasına bakılabilir.

### 2. Orijinal sinir ağı deneyleri

Model eğitimi:

```powershell
python src/train_gru.py
python src/train_extra.py --all
```

**Uyarı:** Bu komutlar ana `results/` klasöründeki mevcut model ağırlıklarının üzerine yazabilir. Eğitimden önce mevcut modellerin yedeği alınmalıdır.

Mevcut model ağırlıklarıyla yeniden değerlendirme:

```powershell
python src/evaluate_gru.py
python src/evaluate_all.py
```

Temel proje kontrolleri:

```powershell
python src/smoke_test.py
python src/check_project.py
```

`smoke_test.py`, gerçek TESS verisi olmadan sentetik örneklerle temel yazılım kontrolleri gerçekleştirir.

### 3. Klasik tahmin yöntemleri

```powershell
python src/evaluate_classical.py
```

Bu işlem Historical Mean, Historical Median, Linear Trend ve Ridge Regression modellerini Persistence yöntemiyle birlikte değerlendirir.

Önemli çıktılar:

- `results/classical_summary_all.csv`
- `results/classical_summary_24h.csv`
- `results/classical_summary_48h.csv`
- `results/classical_object_24h.csv`
- `results/classical_object_48h.csv`
- `results/ridge_24h.npz`
- `results/ridge_48h.npz`

### 4. Model kapasitesi analizi

```powershell
python src/analyze_capacity.py
```

Bu işlem eğitilmiş modellerin parametre sayılarını, ağırlık dosyası büyüklüklerini ve CPU çıkarım sürelerini ölçer.

Çıktılar:

- `results/model_capacity.csv`
- `results/capacity_environment.json`

### 5. Tekrarlı eğitim deneyleri

Sabit veri bölmesinde farklı eğitim başlangıçları:

```powershell
python src/repeat_experiments.py --mode init --seeds 41 42 43
```

Farklı süpernova veri bölmeleri:

```powershell
python src/repeat_experiments.py --mode split --seeds 21 22 23
```

Tekrarlı deneylerin istatistiksel karşılaştırması:

```powershell
python src/analyze_repeat_results.py
```

Yeni modeller ayrı klasörlere kaydedilir:

- `results/repeat_runs/init/`
- `results/repeat_runs/split/`

Bu işlemler ana model ağırlıklarının üzerine yazmaz.

Önemli çıktılar:

- `metrics.csv`
- `object_metrics.csv`
- `protocol.json`
- `paired_model_comparison.csv`
- `experiment_summary.csv`

### 6. Işık eğrisi eğilim analizi

```powershell
python src/analyze_trend_regimes.py
```

Bu analiz geçmiş pencereleri artan, azalan ve yatay eğilim sınıflarına ayırarak Persistence, Ridge, GRU ve CNN modellerini karşılaştırır.

Çıktılar:

- `results/trend_regimes/summary_all.csv`
- `results/trend_regimes/summary_24h.csv`
- `results/trend_regimes/summary_48h.csv`
- Süpernova bazlı eğilim hata tabloları

### 7. Fotometrik belirsizlik analizi

Ölçüm alanlarının incelenmesi:

```powershell
python src/inspect_quality_fields.py
```

Belirsizlik gruplarına göre model performansı:

```powershell
python src/analyze_quality.py
```

Nesneye eşit ağırlıklı bootstrap:

```powershell
python src/analyze_quality_bootstrap.py
```

Pencere ağırlıklı küme bootstrap:

```powershell
python src/analyze_quality_weighted_bootstrap.py
```

Çıktılar:

- `results/quality_analysis/quality_summary_all.csv`
- `results/quality_analysis/quality_objects_all.csv`
- `results/quality_analysis/quality_protocol.json`
- `results/quality_analysis/quality_bootstrap.csv`
- `results/quality_analysis/quality_weighted_bootstrap.csv`

### 8. Akademik grafiklerin oluşturulması

Grafik üretim betiği proje klasörüne eklendikten sonra:

```powershell
python src/create_paper_figures.py
```

Betiğin oluşturacağı grafikler:

| Grafik | İçerik |
|---|---|
| `fig_models_24h` | 24 saatlik genel model karşılaştırması |
| `fig_models_48h` | 48 saatlik genel model karşılaştırması |
| `fig_quality_24h` | 24 saatlik ölçüm belirsizliği analizi |
| `fig_quality_48h` | 48 saatlik ölçüm belirsizliği analizi |
| `fig_bootstrap_24h` | 24 saatlik bootstrap güven aralıkları |
| `fig_bootstrap_48h` | 48 saatlik bootstrap güven aralıkları |

Grafikler `results/paper_figures/` klasörüne kaydedilecek şekilde tasarlanmıştır.

Her grafik için:

- Vektörel PDF
- 450 DPI PNG
- Modeller arasında tutarlı renk paleti
- Türkçe eksen açıklamaları
- Akademik makaleye uygun boyutlandırma

kullanılmaktadır.

### 9. Tek bir süpernovada örnek tahmin

```powershell
python src/predict_demo.py --list

python src/predict_demo.py --model cnn --horizon 24

python src/predict_demo.py --name SN2018fhw --model gru --horizon 48
```

Örnek tahminlerde gerçek gelecek ölçümler, model tahmini ve Persistence yöntemi karşılaştırılır.

Bu işlem arşiv verileri üzerinde geriye dönük değerlendirmedir; canlı gözlem tahmini değildir.

## Proje yapısı

```text
Cosmic-Forecast/
│
├── data/
│   └── raw/                         # İndirilen TESS fotometrik verileri
│
├── results/                         # Model ağırlıkları ve deney çıktıları
│   ├── repeat_runs/                 # Tekrarlı eğitim deneyleri
│   │   ├── init/                    # Farklı eğitim başlangıçları
│   │   └── split/                   # Farklı süpernova bölmeleri
│   │
│   ├── trend_regimes/               # Işık eğrisi eğilim analizleri
│   ├── quality_analysis/            # Ölçüm belirsizliği analizleri
│   └── paper_figures/               # Akademik grafik çıktıları
│
├── src/
│   ├── download_data.py             # Veri indirme
│   ├── dataset.py                   # Veri hazırlama ve normalizasyon
│   ├── inspect_data.py              # Veri kümesi incelemesi
│   ├── baseline.py                  # Persistence karşılaştırması
│   │
│   ├── train_gru.py                 # 24 saatlik GRU eğitimi
│   ├── train_extra.py               # CNN ve 48 saatlik modeller
│   ├── models_extra.py              # Ek model mimarileri
│   ├── evaluate_gru.py              # GRU değerlendirmesi
│   ├── evaluate_all.py              # Ortak model değerlendirmesi
│   ├── predict_demo.py              # Örnek tahmin
│   │
│   ├── evaluate_classical.py        # Klasik tahmin yöntemleri
│   ├── analyze_capacity.py          # Model kapasitesi ve CPU analizi
│   ├── repeat_experiments.py        # Tekrarlı model eğitimleri
│   ├── analyze_repeat_results.py    # Tekrarlı deney istatistikleri
│   ├── analyze_trend_regimes.py      # Geçmiş eğilim analizi
│   │
│   ├── inspect_quality_fields.py    # Belirsizlik veri kontrolü
│   ├── analyze_quality.py           # Belirsizlik grupları
│   ├── analyze_quality_bootstrap.py # Nesne ağırlıklı bootstrap
│   ├── analyze_quality_weighted_bootstrap.py
│   │                                # Pencere ağırlıklı bootstrap
│   ├── create_paper_figures.py      # Akademik grafik üretimi
│   │
│   ├── smoke_test.py                # Sentetik veriyle temel testler
│   └── check_project.py             # Proje tutarlılık kontrolleri
│
├── requirements.txt
├── requirements-model.txt
├── INTERNET_HATASI_COZUMU.md
└── README.md
```

`results/` altındaki ağırlık dosyaları, CSV tabloları ve görseller deneyler çalıştırıldıkça oluşturulur.

## Sınırlılıklar ve sonraki çalışmalar

Cosmic Forecast, gerçek astronomik gözlemler üzerinde gerçekleştirilen bir kısa vadeli zaman serisi tahmin çalışmasıdır. Bulgular değerlendirilirken aşağıdaki sınırlılıklar dikkate alınmalıdır.

**1. Fiziksel evre hizalaması**

Tahmin pencereleri patlama başlangıcına veya maksimum parlaklık zamanına göre hizalanmamıştır. Bu nedenle çalışma yalnızca erken evre süpernovaların fiziksel davranışını modellediğini iddia etmez.

**2. Gözlemsel eğilim ve fiziksel evre ayrımı**

Artan, azalan ve yatay gruplar geçmiş verilerden hesaplanan eğilimlerdir. Bunlar doğrudan süpernova yükseliş, tepe ve sönüm evreleri olarak değerlendirilmemelidir.

**3. Ölçüm belirsizliğinin kullanımı**

`e_CRate`, kalite duyarlılığı analizinde kullanılmaktadır. Ancak mevcut modellerin doğrudan girdi özelliği değildir.

Belirsizlik gruplarının medyan temelli ölçüsü, binlenmiş fotometrinin tam hata yayılımını temsil etmez.

**4. Model kapasitesi farklılıkları**

CNN modellerinin parametre sayıları GRU modellerinden belirgin biçimde yüksektir. Bu nedenle gözlenen başarı farklılıkları yalnızca mimari türüne bağlanamaz. Benzer kapasitedeki modellerle ek karşılaştırma gelecekte yapılabilir.

**5. Tekrarlı deneylerin kapsamı**

Üç eğitim başlangıcı ve üç alternatif veri bölmesi kullanılarak toplam 24 model eğitimi yapılmıştır. Bu, tek deneylik değerlendirmeye göre daha fazla bilgi sağlamaktadır; ancak bütün eğitim rastgeleliğini ve veri dağılımlarını kapsamaz.

**6. Örtüşen zaman pencereleri**

Aynı süpernovadan elde edilen tahmin pencereleri örtüşebilmektedir. Bu bağımlılık nedeniyle nesne bazlı eşleştirme ve küme bootstrap yöntemleri kullanılmıştır.

**7. İstatistiksel yorum**

Güven aralıkları keşifsel amaçlıdır. Çoklu karşılaştırma düzeltmesi yapılmamıştır. Sınırlı sayıda farklı veri bölmesi ve başlangıç üzerinden evrensel mimari üstünlüğü iddia edilmemektedir.

**8. Astrofiziksel yorum**

Daha düşük fotometrik tahmin hatası, tek başına daha iyi fiziksel parametre kestirimi veya astronomik gözlem kararı anlamına gelmez. Bu tür faydaların doğrulanması için ek fiziksel analizler gereklidir.

**9. Gerçek zamanlı kullanım**

Mevcut sistem arşiv gözlemleriyle değerlendirilmiştir. Canlı veri akışı, otomatik gözlem planlama veya gerçek zamanlı astronomik uyarı sistemi uygulanmamıştır.

## Kaynakça

1. Fausnaugh, M. M., et al. (2023). *Four Years of Type Ia Supernovae Observed by TESS: Early-time Light-curve Shapes and Constraints on Companion Interaction Models*. The Astrophysical Journal, 956(2), 108. https://doi.org/10.3847/1538-4357/aceaef

2. PyTorch Documentation. *torch.nn.GRU*. https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html

3. PyTorch Documentation. *Reproducibility*. https://docs.pytorch.org/docs/stable/notes/randomness.html

---

Bu README, Cosmic Forecast projesinin veri hazırlama, model mimarileri, deney protokolleri, istatistiksel karşılaştırmaları ve yeniden üretilebilir çalıştırma adımlarını belgelemek amacıyla hazırlanmıştır.

Ayrıntılı bilimsel değerlendirme ve araştırma bulguları, IEEE biçimindeki proje bildirisi kapsamında ayrıca sunulacaktır.