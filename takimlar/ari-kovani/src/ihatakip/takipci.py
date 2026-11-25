"""Çoklu hedef takipçisi (yalnız numpy).

ByteTrack'in iki aşamalı eşleştirme fikrine dayanır (Zhang ve ark., 2022): önce yüksek skorlu tespitler
izlerle eşleşir, sonra kalan izler düşük skorlu tespitlerle eşleştirilir. Uzaktaki küçük İHA'lar çoğu
karede düşük skorla görünür; ikinci aşama bu kareleri kurtarıp izin kopmasını önler.

Her iz sabit hızlı bir Kalman filtresi taşır: durum [cx, cy, w, h, vx, vy, vw, vh].
Eşleştirme maliyeti IoU'dur. Küçük hedeflerde IoU çok oynak olduğu için, normalleştirilmiş merkez uzaklığı da
ikinci bir benzerlik olarak eklenir. Atama açgözlü yapılır (az sayıda hedef için Macar algoritmasıyla
hemen hemen aynı sonuç, scipy bağımlılığı yok).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrikler import iou_matrisi


class KalmanKutu:
    """Sabit hızlı model; ölçüm [cx, cy, w, h]."""

    def __init__(self, kutu_cxcywh: np.ndarray):
        self.x = np.zeros(8)
        self.x[:4] = kutu_cxcywh
        olcek = max(kutu_cxcywh[2], kutu_cxcywh[3], 4.0)
        self.P = np.diag([olcek, olcek, olcek, olcek, 10 * olcek, 10 * olcek, olcek, olcek]) ** 2 * 0.01
        self.F = np.eye(8)
        self.F[:4, 4:] = np.eye(4)
        self.H = np.eye(4, 8)

    def tahmin(self) -> np.ndarray:
        olcek = max(self.x[2], self.x[3], 4.0)
        q_pos, q_hiz = 0.05 * olcek, 0.0125 * olcek
        Q = np.diag([q_pos, q_pos, q_pos, q_pos, q_hiz, q_hiz, q_hiz, q_hiz]) ** 2
        self.x = self.F @ self.x
        self.x[2:4] = np.maximum(self.x[2:4], 1.0)
        self.P = self.F @ self.P @ self.F.T + Q
        return self.x[:4].copy()

    def guncelle(self, olcum: np.ndarray) -> None:
        olcek = max(olcum[2], olcum[3], 4.0)
        r = 0.05 * olcek
        R = np.diag([r, r, r, r]) ** 2
        y = olcum - self.H @ self.x
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(8) - K @ self.H) @ self.P


def xyxy_cxcywh(k: np.ndarray) -> np.ndarray:
    return np.array([(k[0] + k[2]) / 2, (k[1] + k[3]) / 2, k[2] - k[0], k[3] - k[1]], np.float64)


def cxcywh_xyxy(k: np.ndarray) -> np.ndarray:
    return np.array([k[0] - k[2] / 2, k[1] - k[3] / 2, k[0] + k[2] / 2, k[1] + k[3] / 2], np.float64)


@dataclass
class Iz:
    kimlik: int
    kalman: KalmanKutu
    skor: float
    sinif: int
    yas: int = 1                 # kaç karedir yaşıyor
    eslesme_sayisi: int = 1      # kaç karede tespitle eşleşti
    kayip: int = 0               # art arda kaç karedir eşleşmedi
    onayli: bool = False
    gecmis: list = field(default_factory=list)  # merkez noktaları (çizim için)

    @property
    def kutu(self) -> np.ndarray:
        return cxcywh_xyxy(self.kalman.x[:4])

    @property
    def hiz(self) -> np.ndarray:
        return self.kalman.x[4:6].copy()


def _benzerlik(izler: list, tespitler: np.ndarray) -> np.ndarray:
    if not izler or len(tespitler) == 0:
        return np.zeros((len(izler), len(tespitler)))
    iz_kutulari = np.stack([iz.kutu for iz in izler])
    iou = iou_matrisi(iz_kutulari, tespitler[:, :4])
    # normalleştirilmiş merkez uzaklığına dayalı benzerlik: küçük hedeflerde IoU sıfır olsa da eşleşebilsin
    iz_m = np.stack([(iz_kutulari[:, 0] + iz_kutulari[:, 2]) / 2, (iz_kutulari[:, 1] + iz_kutulari[:, 3]) / 2], 1)
    te_m = np.stack([(tespitler[:, 0] + tespitler[:, 2]) / 2, (tespitler[:, 1] + tespitler[:, 3]) / 2], 1)
    olcek = np.maximum(np.maximum(iz_kutulari[:, 2] - iz_kutulari[:, 0], iz_kutulari[:, 3] - iz_kutulari[:, 1]), 8.0)
    uzaklik = np.linalg.norm(iz_m[:, None] - te_m[None], axis=2) / olcek[:, None]
    merkez = np.clip(1.0 - uzaklik / 2.0, 0.0, 1.0)  # 2 kutu boyu uzakta sıfır
    return np.maximum(iou, 0.8 * merkez)


def _acgozlu_ata(benzerlik: np.ndarray, esik: float):
    eslesmeler = []
    if benzerlik.size == 0:
        return eslesmeler, list(range(benzerlik.shape[0])), list(range(benzerlik.shape[1]))
    b = benzerlik.copy()
    while True:
        i, j = np.unravel_index(np.argmax(b), b.shape)
        if b[i, j] < esik:
            break
        eslesmeler.append((i, j))
        b[i, :] = -1
        b[:, j] = -1
    eslesen_i = {i for i, _ in eslesmeler}
    eslesen_j = {j for _, j in eslesmeler}
    return (eslesmeler, [i for i in range(benzerlik.shape[0]) if i not in eslesen_i],
            [j for j in range(benzerlik.shape[1]) if j not in eslesen_j])


class Takipci:
    def __init__(self, yuksek_esik: float = 0.4, dusuk_esik: float = 0.1, yeni_iz_esigi: float = 0.5,
                 eslesme_esigi: float = 0.2, onay_karesi: int = 3, en_fazla_kayip: int = 30,
                 onaysiz_tolerans: int = 1, gecmis_uzunlugu: int = 40):
        self.onaysiz_tolerans = onaysiz_tolerans
        self.yuksek_esik = yuksek_esik
        self.dusuk_esik = dusuk_esik
        self.yeni_iz_esigi = yeni_iz_esigi
        self.eslesme_esigi = eslesme_esigi
        self.onay_karesi = onay_karesi
        self.en_fazla_kayip = en_fazla_kayip
        self.gecmis_uzunlugu = gecmis_uzunlugu
        self.izler: list[Iz] = []
        self._sonraki_kimlik = 1

    def _yeni_iz(self, t: np.ndarray) -> None:
        iz = Iz(self._sonraki_kimlik, KalmanKutu(xyxy_cxcywh(t[:4])), float(t[4]), int(t[5]))
        self._sonraki_kimlik += 1
        self.izler.append(iz)

    def guncelle(self, tespitler: np.ndarray) -> list[Iz]:
        """Bir karenin tespitleriyle (N, 6) izleri günceller; onaylı ve bu karede görünür izleri döndürür."""
        tespitler = np.asarray(tespitler, np.float64).reshape(-1, 6)
        for iz in self.izler:
            iz.kalman.tahmin()
            iz.yas += 1
        yuksek = tespitler[tespitler[:, 4] >= self.yuksek_esik]
        dusuk = tespitler[(tespitler[:, 4] >= self.dusuk_esik) & (tespitler[:, 4] < self.yuksek_esik)]

        # 1. aşama: tüm izler <-> yüksek skorlu tespitler
        eslesme, kalan_iz, kalan_yuksek = _acgozlu_ata(_benzerlik(self.izler, yuksek), self.eslesme_esigi)
        for i, j in eslesme:
            self._eslestir(self.izler[i], yuksek[j])
        # 2. aşama: kalan izler <-> düşük skorlu tespitler (yalnız daha önce onaylanmış izler için)
        aday_izler = [self.izler[i] for i in kalan_iz if self.izler[i].onayli]
        eslesme2, kalan2, _ = _acgozlu_ata(_benzerlik(aday_izler, dusuk), self.eslesme_esigi + 0.1)
        eslesen2 = set()
        for i, j in eslesme2:
            self._eslestir(aday_izler[i], dusuk[j])
            eslesen2.add(id(aday_izler[i]))
        for i in kalan_iz:
            iz = self.izler[i]
            if id(iz) not in eslesen2:
                iz.kayip += 1
        # yeni izler: eşleşmeyen yüksek skorlu tespitlerden
        for j in kalan_yuksek:
            if yuksek[j, 4] >= self.yeni_iz_esigi:
                self._yeni_iz(yuksek[j])
        # ölü izleri sil: onaysız iz onaysiz_tolerans kareden, onaylı iz en_fazla_kayip kareden uzun kaybolursa
        self.izler = [iz for iz in self.izler
                      if (iz.onayli and iz.kayip <= self.en_fazla_kayip)
                      or (not iz.onayli and iz.kayip <= self.onaysiz_tolerans)]
        return [iz for iz in self.izler if iz.onayli and iz.kayip == 0]

    def _eslestir(self, iz: Iz, t: np.ndarray) -> None:
        iz.kalman.guncelle(xyxy_cxcywh(t[:4]))
        iz.skor = float(t[4])
        iz.eslesme_sayisi += 1
        iz.kayip = 0
        if iz.eslesme_sayisi >= self.onay_karesi:
            iz.onayli = True
        k = iz.kutu
        iz.gecmis.append(((k[0] + k[2]) / 2, (k[1] + k[3]) / 2))
        if len(iz.gecmis) > self.gecmis_uzunlugu:
            iz.gecmis.pop(0)

    def tahmini_izler(self) -> list[Iz]:
        """Bu karede tespit edilemeyen ama Kalman ile tahmin edilen onaylı izler (kısa kayıplar)."""
        return [iz for iz in self.izler if iz.onayli and 0 < iz.kayip <= self.en_fazla_kayip]
