"""Cosmic Forecast - VizieR TESS fotometri verilerinin güvenli indirilmesi.

Kullanım: python src/download_data.py
Veri, kaynak katalogdan alınır; TLS sertifika doğrulaması devre dışı bırakılmaz.
"""

from pathlib import Path

import pandas as pd
import requests

from dataset import DATA_FILE, TSV_FILE


# Bir sunucu yanıt vermediğinde alternatif VizieR aynaları denenir.
SERVERS = [
    "https://vizier.cfa.harvard.edu/viz-bin/asu-tsv",
    "https://vizier.idia.ac.za/viz-bin/asu-tsv",
    "https://vizier.cds.unistra.fr/viz-bin/asu-tsv",
]

PARAMS = {
    "-source": "J/ApJ/956/108/table2",
    "-out": "Name,BJD,CRate,e_CRate",
    "-out.max": "unlimited",
}

EXPECTED_ROWS = 680384
EXPECTED_SUPERNOVAE = 307
MINIMUM_LINES = 600000


def _check_catalog_content(df, source):
    """Kaynak tablonun sütunlarını, geçerli ölçüm ve nesne sayısını doğrula."""
    df.columns = df.columns.str.strip()
    required = ["Name", "BJD", "CRate", "e_CRate"]
    if not set(required).issubset(df.columns):
        raise ValueError(f"{source}: gerekli sütunlar eksik. Bulunan: {list(df.columns)}")

    df = df[required].copy()
    df["Name"] = df["Name"].astype("string").str.strip()
    for column in ("BJD", "CRate", "e_CRate"):
        df[column] = pd.to_numeric(df[column], errors="coerce")

    # dataset.load_photometry() ile aynı geçerlilik filtresi uygulanır.
    df = df.dropna(subset=required)
    df = df[(df["Name"] != "") & (df["e_CRate"] > 0)]

    n_rows = len(df)
    n_supernovae = int(df["Name"].nunique())
    if (n_rows, n_supernovae) != (EXPECTED_ROWS, EXPECTED_SUPERNOVAE):
        raise ValueError(
            f"{source}: beklenen {EXPECTED_ROWS:,} geçerli ölçüm / "
            f"{EXPECTED_SUPERNOVAE} süpernova; "
            f"bulunan {n_rows:,} ölçüm / {n_supernovae} süpernova."
        )


def verify_download(path: Path):
    """TSV dosyasını biçim, satır sayısı ve gerçek ölçüm sayısıyla doğrula."""
    path = Path(path)
    with path.open("rb") as source:
        head = source.read(500000)

    lower = head.lower().lstrip()
    if b"<html" in lower[:1000] or b"<!doctype" in lower[:1000]:
        raise ValueError("Tablo yerine HTML sayfası döndü.")
    if b"#error" in lower or b"#****" in lower:
        raise ValueError("VizieR sorgu hatası: " + head.decode("utf-8", "replace")[:500])
    if b"\t" not in head:
        raise ValueError("Sunucu TSV yerine başka biçimde veri döndürdü.")

    # Hızlı ön kontrol, açıkça eksik indirmeleri reddeder.
    with path.open("rb") as source:
        nlines = sum(
            chunk.count(b"\n")
            for chunk in iter(lambda: source.read(1024 * 1024), b"")
        )
    if nlines < MINIMUM_LINES:
        raise ValueError(f"Dosya eksik: {nlines:,} satır.")

    # Kritik ilave kontrol: 600.000'den fazla satır tek başına
    # 680.384 ölçümün eksiksiz indirildiğini kanıtlamaz.
    try:
        frame = pd.read_csv(
            path,
            sep="\t",
            comment="#",
            low_memory=False,
            encoding="utf-8",
            on_bad_lines="error",
        )
    except (pd.errors.ParserError, UnicodeError, ValueError) as exc:
        raise ValueError(f"TSV okunamadı: {exc}") from exc

    _check_catalog_content(frame, str(path))
    return nlines


def verify_legacy_dat(path: Path):
    """DAT geri dönüş seçeneğinde yalnızca dosya boyutuna güvenme."""
    path = Path(path)
    frame = pd.read_fwf(
        path,
        colspecs=[(0, 10), (11, 21), (22, 30), (31, 38)],
        names=["Name", "BJD", "CRate", "e_CRate"],
        dtype={"Name": "string"},
    )
    _check_catalog_content(frame, str(path))


def _quarantine_invalid_tsv(path: Path):
    """Geçersiz TSV'yi sakla; böylece dataset.py geçerli DAT'ı tercih edebilir."""
    target = path.with_name(path.name + ".invalid")
    suffix = 1
    while target.exists():
        target = path.with_name(path.name + f".invalid.{suffix}")
        suffix += 1
    path.replace(target)
    print(f"Geçersiz TSV yedeklendi: {target}")


def main():
    TSV_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Daha önce indirilen dosya tam ise tekrar indirilmez.
    if TSV_FILE.exists():
        try:
            lines = verify_download(TSV_FILE)
            print(f"Mevcut veri kullanılıyor: {TSV_FILE} ({lines:,} satır)")
            return
        except (ValueError, OSError) as exc:
            print(f"TSV dosyası geçersiz: {exc}. Yeniden denenecek.")

    # Önceki sürümlerin sabit-genişlikli DAT arşivi de desteklenir.
    if DATA_FILE.exists():
        try:
            verify_legacy_dat(DATA_FILE)
        except (ValueError, OSError, pd.errors.ParserError) as exc:
            print(f"DAT dosyası geçersiz: {exc}. Yeniden indirme denenecek.")
        else:
            # dataset.load_photometry() normalde TSV'yi öncelikli açar.
            # Mevcut TSV bozuksa DAT'ın seçilebilmesi için onu yedekle.
            if TSV_FILE.exists():
                _quarantine_invalid_tsv(TSV_FILE)
            print(f"Geçerli sabit genişlikli dosya kullanılıyor: {DATA_FILE}")
            return

    tmp = TSV_FILE.with_suffix(".tsv.part")
    for host in SERVERS:
        print(f"\nVizieR sorgulanıyor: {host}", flush=True)
        try:
            with requests.get(
                host,
                params=PARAMS,
                stream=True,
                timeout=(25, 180),
                headers={"User-Agent": "CosmicForecastCourseProject/1.1"},
            ) as response:
                print("URL:", response.url)
                response.raise_for_status()
                with tmp.open("wb") as dest:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            dest.write(chunk)

            lines = verify_download(tmp)
            tmp.replace(TSV_FILE)
            print(f"BAŞARILI: {TSV_FILE} ({lines:,} satır)")
            print("Sıradaki komut: python src/inspect_data.py")
            return
        except (requests.RequestException, OSError, ValueError, pd.errors.ParserError) as exc:
            print("Bu sunucuda başarısız:", str(exc)[:350])
            tmp.unlink(missing_ok=True)

    print("\nBütün VizieR sunucularındaki indirmeler başarısız oldu.")
    print("Tabloyu elle TSV olarak data/raw/table2.tsv konumuna kaydedebilirsin.")
    print("Katalog: https://vizier.cfa.harvard.edu/viz-bin/VizieR?-source=J/ApJ/956/108/table2")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
