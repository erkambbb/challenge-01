"""Tespit metrikleri (yalnız numpy).

AP hesabı pycocotools ile aynı kuralları izler: görüntü başına en fazla 100 tespit, skora göre açgözlü
eşleştirme, 101 noktalı ara değerli hassasiyet. training/metrik_dogrula.py bu modülün sonucunu
pycocotools ile karşılaştırır.

Kutular piksel cinsinden [x1, y1, x2, y2] biçimindedir.
"""

from __future__ import annotations

import numpy as np

IOU_ESIKLERI = np.linspace(0.5, 0.95, 10)
GERI_CAGIRMA_NOKTALARI = np.linspace(0.0, 1.0, 101)
# Hedef boyutu (kutu alanının karekökü, özgün görüntü pikseli) için aralıklar.
BOYUT_ARALIKLARI = (("0-8px", 0, 8), ("8-16px", 8, 16), ("16-32px", 16, 32), ("32-64px", 32, 64),
                    ("64px+", 64, 1e9))


def iou_matrisi(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float64)
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    kesisim = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    alan_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    alan_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    birlesim = alan_a[:, None] + alan_b[None, :] - kesisim
    return np.where(birlesim > 0, kesisim / np.maximum(birlesim, 1e-12), 0.0)


def _eslestir(tespit: np.ndarray, gercek: np.ndarray, esik: float) -> np.ndarray:
    """Skora göre sıralı tespitleri açgözlü eşleştirir; her tespit için TP (True) / FP (False)."""
    tp = np.zeros(len(tespit), bool)
    if len(tespit) == 0 or len(gercek) == 0:
        return tp
    iou = iou_matrisi(tespit[:, :4], gercek)
    kullanildi = np.zeros(len(gercek), bool)
    for i in range(len(tespit)):
        en_iyi, en_iyi_j = min(esik, 1 - 1e-10), -1
        for j in range(len(gercek)):
            if kullanildi[j] or iou[i, j] < en_iyi:
                continue
            en_iyi, en_iyi_j = iou[i, j], j
        if en_iyi_j >= 0:
            kullanildi[en_iyi_j] = True
            tp[i] = True
    return tp


def _ap(skorlar: np.ndarray, tp: np.ndarray, gercek_sayisi: int) -> float:
    if gercek_sayisi == 0:
        return float("nan")
    if len(skorlar) == 0:
        return 0.0
    sira = np.argsort(-skorlar, kind="mergesort")
    tp = tp[sira]
    tp_birikimli = np.cumsum(tp).astype(np.float64)
    fp_birikimli = np.cumsum(~tp).astype(np.float64)
    duyarlilik = tp_birikimli / gercek_sayisi
    hassasiyet = tp_birikimli / np.maximum(tp_birikimli + fp_birikimli, np.spacing(1))
    # sağdan sola monoton artan zarf (pycocotools ile aynı)
    for i in range(len(hassasiyet) - 1, 0, -1):
        if hassasiyet[i] > hassasiyet[i - 1]:
            hassasiyet[i - 1] = hassasiyet[i]
    q = np.zeros(len(GERI_CAGIRMA_NOKTALARI))
    indeksler = np.searchsorted(duyarlilik, GERI_CAGIRMA_NOKTALARI, side="left")
    for k, idx in enumerate(indeksler):
        if idx < len(hassasiyet):
            q[k] = hassasiyet[idx]
    return float(q.mean())


class TespitDegerlendirici:
    """Görüntü görüntü tespit ve gerçek kutu ekleyip sonunda özet metrikleri hesaplar.

    Tek sınıflı (İHA) kullanım için tasarlandı; çok sınıflıda AP sınıflar üzerinden ortalanır.
    """

    def __init__(self, sinif_sayisi: int = 1, en_fazla_tespit: int = 100, calisma_esigi: float = 0.3):
        self.sinif_sayisi = sinif_sayisi
        self.en_fazla_tespit = en_fazla_tespit
        self.calisma_esigi = calisma_esigi
        self.goruntuler = []  # (tespit (N,6), gercek (M,5): x1 y1 x2 y2 sinif)

    def ekle(self, tespit: np.ndarray, gercek: np.ndarray) -> None:
        tespit = np.asarray(tespit, np.float64).reshape(-1, 6)
        gercek = np.asarray(gercek, np.float64).reshape(-1, 5)
        sira = np.argsort(-tespit[:, 4], kind="mergesort")
        self.goruntuler.append((tespit[sira], gercek))

    # ------------------------------------------------------------------
    def _sinif_ap(self, sinif: int, esik: float) -> float:
        skorlar, tps, toplam = [], [], 0
        for tespit, gercek in self.goruntuler:
            t = tespit[tespit[:, 5] == sinif][: self.en_fazla_tespit]
            g = gercek[gercek[:, 4] == sinif][:, :4]
            toplam += len(g)
            skorlar.append(t[:, 4])
            tps.append(_eslestir(t, g, esik))
        if not skorlar:
            return float("nan")
        return _ap(np.concatenate(skorlar), np.concatenate(tps), toplam)

    def ap(self, esik: float = 0.5) -> float:
        degerler = [self._sinif_ap(c, esik) for c in range(self.sinif_sayisi)]
        degerler = [d for d in degerler if not np.isnan(d)]
        return float(np.mean(degerler)) if degerler else 0.0

    def calisma_noktasi(self, esik_skor: float | None = None, iou: float = 0.5) -> dict:
        """Belirli bir skor eşiğinde TP/FP/FN, hassasiyet, duyarlılık, F1 ve boyuta göre duyarlılık."""
        esik_skor = self.calisma_esigi if esik_skor is None else esik_skor
        tp = fp = fn = 0
        bos_goruntu = bos_fp = 0
        boyut_toplam = {ad: 0 for ad, _, _ in BOYUT_ARALIKLARI}
        boyut_bulunan = {ad: 0 for ad, _, _ in BOYUT_ARALIKLARI}
        for tespit, gercek in self.goruntuler:
            t = tespit[tespit[:, 4] >= esik_skor][: self.en_fazla_tespit]
            for c in range(self.sinif_sayisi):
                tc = t[t[:, 5] == c]
                gc = gercek[gercek[:, 4] == c][:, :4]
                eslesme = _eslestir(tc, gc, iou)
                tp += int(eslesme.sum())
                fp += int((~eslesme).sum())
                fn += len(gc) - int(eslesme.sum())
                # hangi gerçek kutuların bulunduğu (boyut kırılımı için)
                if len(gc):
                    bulundu = np.zeros(len(gc), bool)
                    if len(tc):
                        iou_m = iou_matrisi(tc[:, :4], gc)
                        kullanildi = np.zeros(len(gc), bool)
                        for i in range(len(tc)):
                            adaylar = np.where(~kullanildi & (iou_m[i] >= iou))[0]
                            if len(adaylar):
                                j = adaylar[np.argmax(iou_m[i, adaylar])]
                                kullanildi[j] = True
                        bulundu = kullanildi
                    boyut = np.sqrt(np.clip((gc[:, 2] - gc[:, 0]) * (gc[:, 3] - gc[:, 1]), 0, None))
                    for ad, alt, ust in BOYUT_ARALIKLARI:
                        m = (boyut >= alt) & (boyut < ust)
                        boyut_toplam[ad] += int(m.sum())
                        boyut_bulunan[ad] += int((m & bulundu).sum())
            if len(gercek) == 0:
                bos_goruntu += 1
                bos_fp += len(t)
        hassasiyet = tp / (tp + fp) if tp + fp else 0.0
        duyarlilik = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * hassasiyet * duyarlilik / (hassasiyet + duyarlilik) if hassasiyet + duyarlilik else 0.0
        return {
            "skor_esigi": esik_skor,
            "tp": tp, "fp": fp, "fn": fn,
            "hassasiyet": hassasiyet, "duyarlilik": duyarlilik, "f1": f1,
            "bos_goruntu_basina_yanlis_alarm": (bos_fp / bos_goruntu) if bos_goruntu else None,
            "boyuta_gore_duyarlilik": {ad: (boyut_bulunan[ad] / boyut_toplam[ad]) if boyut_toplam[ad] else None
                                       for ad, _, _ in BOYUT_ARALIKLARI},
            "boyuta_gore_hedef_sayisi": boyut_toplam,
        }

    def en_iyi_f1(self, adaylar=None) -> dict:
        adaylar = np.round(np.arange(0.05, 0.96, 0.05), 2) if adaylar is None else adaylar
        en_iyi = None
        for e in adaylar:
            n = self.calisma_noktasi(float(e))
            if en_iyi is None or n["f1"] > en_iyi["f1"]:
                en_iyi = n
        return en_iyi

    def ozet(self) -> dict:
        ap50 = self.ap(0.5)
        ap_aralik = float(np.mean([self.ap(float(e)) for e in IOU_ESIKLERI]))
        return {
            "AP@0.5": ap50,
            "AP@0.75": self.ap(0.75),
            "AP@[.5:.95]": ap_aralik,
            "goruntu_sayisi": len(self.goruntuler),
            "hedef_sayisi": int(sum(len(g) for _, g in self.goruntuler)),
            "calisma_noktasi": self.calisma_noktasi(),
            "en_iyi_f1": self.en_iyi_f1(),
        }
