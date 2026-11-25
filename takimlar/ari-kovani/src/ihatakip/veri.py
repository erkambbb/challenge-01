"""Test seti okuma: yarışmanın "Nesne tespiti" biçimi (YOLO etiketleri).

test-seti/
├── siniflar.txt          her satırda bir sınıf (0'dan başlar)
├── goruntuler/*.jpg|png
└── etiketler/*.txt       sinif_no x_merkez y_merkez genislik yukseklik (0-1 arası); boş dosya = hedef yok
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

GORUNTU_UZANTILARI = (".jpg", ".jpeg", ".png", ".bmp")


def siniflari_oku(test_seti: str | Path) -> list[str]:
    yol = Path(test_seti) / "siniflar.txt"
    if not yol.is_file():
        return ["iha"]
    return [s.strip() for s in yol.read_text(encoding="utf-8").splitlines() if s.strip()]


def ornekleri_listele(test_seti: str | Path) -> list[tuple[Path, Path]]:
    test_seti = Path(test_seti)
    goruntuler = sorted(p for p in (test_seti / "goruntuler").iterdir() if p.suffix.lower() in GORUNTU_UZANTILARI)
    return [(p, test_seti / "etiketler" / (p.stem + ".txt")) for p in goruntuler]


def yolo_etiketi_oku(etiket_yolu: Path, genislik: int, yukseklik: int) -> np.ndarray:
    """YOLO etiketini piksel cinsinden (M, 5) [x1, y1, x2, y2, sinif] dizisine çevirir."""
    if not etiket_yolu.is_file():
        return np.zeros((0, 5), np.float64)
    satirlar = []
    for satir in etiket_yolu.read_text(encoding="utf-8").splitlines():
        p = satir.split()
        if len(p) < 5:
            continue
        c, xc, yc, w, h = int(float(p[0])), float(p[1]), float(p[2]), float(p[3]), float(p[4])
        satirlar.append([(xc - w / 2) * genislik, (yc - h / 2) * yukseklik,
                         (xc + w / 2) * genislik, (yc + h / 2) * yukseklik, c])
    if not satirlar:
        return np.zeros((0, 5), np.float64)
    return np.asarray(satirlar, np.float64)
