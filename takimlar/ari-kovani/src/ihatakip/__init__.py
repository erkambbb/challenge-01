"""ihatakip: T3 Gemstone O1 üzerinde havadan havaya İHA tespiti ve takibi.

Modüller:
    onislem     görüntüyü modelin giriş boyutuna getirir (letterbox)
    algilayici  ONNX Runtime oturumu; TIDL (C7x/MMA) ya da yalnız CPU
    takipci     ByteTrack benzeri iki aşamalı, Kalman filtreli çoklu hedef takipçisi
    metrikler   AP@0.5, AP@[.5:.95], hassasiyet/duyarlılık, boyuta göre duyarlılık
    veri        test seti (YOLO biçimi) okuma
    cizim       kare üzerine kutu, iz ve durum bilgisi çizme
    kamera      kamera / video / GStreamer kaynakları
"""

__version__ = "0.1.0"
