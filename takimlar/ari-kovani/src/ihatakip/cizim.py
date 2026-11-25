"""Kare üzerine tespit, iz ve durum bilgisi çizme."""

from __future__ import annotations

import cv2
import numpy as np

YESIL = (60, 200, 60)
SARI = (0, 200, 255)
KIRMIZI = (40, 40, 230)
GRI = (160, 160, 160)
BEYAZ = (255, 255, 255)


def _renk(kimlik: int) -> tuple:
    rng = np.random.default_rng(kimlik * 7919)
    r, g, b = rng.integers(80, 256, 3)
    return int(b), int(g), int(r)


def tespitleri_ciz(kare: np.ndarray, tespitler: np.ndarray, esik: float = 0.3) -> None:
    for t in tespitler:
        if t[4] < esik:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in t[:4])
        cv2.rectangle(kare, (x1, y1), (x2, y2), SARI, 1)


def izleri_ciz(kare: np.ndarray, izler, tahmini=()) -> None:
    for iz in izler:
        renk = _renk(iz.kimlik)
        x1, y1, x2, y2 = (int(round(v)) for v in iz.kutu)
        # küçük hedefler görünür olsun diye en az 24 px'lik çerçeve
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        yw, yh = max(x2 - x1, 24) // 2, max(y2 - y1, 24) // 2
        cv2.rectangle(kare, (cx - yw, cy - yh), (cx + yw, cy + yh), renk, 2)
        etiket = f"IHA #{iz.kimlik} {iz.skor:.2f}"
        konum = (cx - yw, max(cy - yh - 6, 12))
        cv2.putText(kare, etiket, konum, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(kare, etiket, konum, cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1, cv2.LINE_AA)
        if len(iz.gecmis) > 1:
            pts = np.asarray(iz.gecmis, np.int32).reshape(-1, 1, 2)
            cv2.polylines(kare, [pts], False, renk, 1, cv2.LINE_AA)
    for iz in tahmini:
        x1, y1, x2, y2 = (int(round(v)) for v in iz.kutu)
        cv2.rectangle(kare, (x1, y1), (x2, y2), GRI, 1)


def durum_ciz(kare: np.ndarray, satirlar: list[str]) -> None:
    y = 22
    for s in satirlar:
        cv2.putText(kare, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(kare, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, BEYAZ, 1, cv2.LINE_AA)
        y += 22
