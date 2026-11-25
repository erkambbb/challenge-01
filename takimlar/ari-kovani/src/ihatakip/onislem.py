"""Görüntüyü modelin giriş boyutuna getirme.

Eğitimdeki (mmdet YOLOX test hattı) ve TI model zoo yapılandırmasındaki önişlemle aynıdır:
en-boy oranı korunarak küçültülür, sağ ve alt kenar 114 gri değeriyle doldurulur (köşe hizalı letterbox).
Model BGR sırasında, 0-255 aralığında, normalleştirilmemiş float32 NCHW girdi bekler.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - kartta cv2 beklenir, yoksa açık hata ver
    cv2 = None


def letterbox(bgr: np.ndarray, genislik: int, yukseklik: int, dolgu: int = 114):
    """BGR uint8 HxWx3 görüntüyü (yukseklik, genislik) tuvaline köşe hizalı yerleştirir.

    Döndürür: (tuval uint8 HxWx3, olcek r). Model koordinatı / r = özgün görüntü koordinatı.
    """
    if cv2 is None:
        raise RuntimeError("OpenCV (cv2) bulunamadı; requirements.txt'deki paketleri kurun")
    h, w = bgr.shape[:2]
    r = min(yukseklik / h, genislik / w)
    yeni_w, yeni_h = int(round(w * r)), int(round(h * r))
    tuval = np.full((yukseklik, genislik, 3), dolgu, dtype=np.uint8)
    if (yeni_w, yeni_h) != (w, h):
        kucuk = cv2.resize(bgr, (yeni_w, yeni_h), interpolation=cv2.INTER_LINEAR)
    else:
        kucuk = bgr
    tuval[:yeni_h, :yeni_w] = kucuk
    return tuval, r


def tensore_cevir(tuval: np.ndarray, ortalama=(0.0, 0.0, 0.0), olcek=(1.0, 1.0, 1.0)) -> np.ndarray:
    """HxWx3 uint8 -> 1x3xHxW float32. Ortalama/ölçek (x - ortalama) * ölçek biçiminde uygulanır."""
    x = tuval.astype(np.float32)
    if any(o != 0.0 for o in ortalama) or any(s != 1.0 for s in olcek):
        x = (x - np.asarray(ortalama, np.float32)) * np.asarray(olcek, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])
