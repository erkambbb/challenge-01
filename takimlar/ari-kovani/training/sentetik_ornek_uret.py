#!/usr/bin/env python3
"""Prosedürel sentetik İHA görüntüsü üretici (yalnız numpy + OpenCV).

Amaç: asıl veri seti (X-Plane + gerçek görüntüler) hazır olmadan eğitim -> ONNX -> TIDL -> değerlendirme
zincirini uçtan uca sınamak (duman testi). Üretilen görüntülerin tamamı bu betikle çizilir; üçüncü taraf
içerik yoktur. Asıl modeli eğitmek için KULLANILMAZ.

Çıktı (YOLO biçimi, yarışmanın test seti biçimiyle aynı):
    <cikti>/goruntuler/<onek>_000001.jpg
    <cikti>/etiketler/<onek>_000001.txt     sinif xc yc w h (0-1)
    <cikti>/siniflar.txt                    iha

Örnek:
    python3 sentetik_ornek_uret.py --cikti /tmp/sentetik/train --adet 2000 --tohum 1
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import cv2
import numpy as np


def gok(rng, h, w):
    ust = np.array(rng.choice([[200, 140, 70], [210, 170, 120], [170, 160, 150], [230, 200, 160]]), np.float32)
    alt = np.clip(ust + rng.uniform(20, 70), 0, 255)
    t = np.linspace(0, 1, h, dtype=np.float32)[:, None, None]
    img = ust * (1 - t) + alt * t
    img = np.repeat(img, w, axis=1)
    # bulut: farklı ölçeklerde bulanık gürültü
    bulut = np.zeros((h, w), np.float32)
    for olcek in (8, 32, 96):
        n = rng.random((max(h // olcek, 2), max(w // olcek, 2))).astype(np.float32)
        bulut += cv2.resize(n, (w, h), interpolation=cv2.INTER_CUBIC) * (olcek / 96)
    bulut = (bulut - bulut.min()) / (np.ptp(bulut) + 1e-6)
    yogunluk = rng.uniform(0.0, 0.9)
    maske = np.clip((bulut - (1 - yogunluk)) * 3, 0, 1)[..., None]
    beyaz = rng.uniform(200, 250)
    img = img * (1 - maske) + beyaz * maske
    return img


def yer_ekle(rng, img):
    h, w = img.shape[:2]
    ufuk = int(h * rng.uniform(0.35, 0.8))
    zemin = np.array(rng.choice([[60, 90, 70], [70, 80, 90], [90, 100, 110], [50, 70, 60]]), np.float32)
    doku = cv2.resize(rng.random((max((h - ufuk) // 6, 2), max(w // 6, 2))).astype(np.float32), (w, h - ufuk))
    img[ufuk:] = zemin + (doku[..., None] - 0.5) * 60
    # bina blokları (şehir dokusu)
    for _ in range(rng.integers(0, 60)):
        bx = rng.integers(0, w)
        bw = rng.integers(4, 40)
        bh = rng.integers(4, int(h * 0.25))
        renk = rng.uniform(70, 190) + rng.uniform(-15, 15, 3)
        cv2.rectangle(img, (int(bx), ufuk - int(bh) // 3), (int(bx + bw), ufuk + int(bh)), renk.tolist(), -1)
    img[ufuk - 2:ufuk + 2] = cv2.GaussianBlur(img[ufuk - 4:ufuk + 4], (1, 5), 0)[2:6]
    return img


def iha_cokgen(rng, tur):
    """Birim ölçekte (yaklaşık -1..1) İHA silueti çokgenleri."""
    if tur == "sabit_kanat":
        govde = np.array([[-1.0, -0.06], [0.9, -0.05], [1.0, 0], [0.9, 0.05], [-1.0, 0.06]])
        kanat_ac = rng.uniform(0.7, 1.1)
        kanat = np.array([[0.1, -kanat_ac], [0.25, -kanat_ac], [0.3, 0], [0.25, kanat_ac], [0.1, kanat_ac], [0.0, 0]])
        kuyruk = np.array([[-0.95, -0.3], [-0.8, -0.3], [-0.75, 0], [-0.8, 0.3], [-0.95, 0.3]])
        return [govde, kanat, kuyruk]
    if tur == "dortlu":
        parcalar = [np.array([[-0.2, -0.2], [0.2, -0.2], [0.2, 0.2], [-0.2, 0.2]])]
        for aci in (45, 135, 225, 315):
            a = math.radians(aci)
            uc = np.array([math.cos(a), math.sin(a)]) * 0.85
            dik = np.array([-math.sin(a), math.cos(a)]) * 0.05
            parcalar.append(np.array([dik, uc + dik, uc - dik, -dik]))
            daire = uc + 0.22 * np.stack([np.cos(np.linspace(0, 2 * math.pi, 12)),
                                          np.sin(np.linspace(0, 2 * math.pi, 12))], 1)
            parcalar.append(daire)
        return parcalar
    # delta kanat
    return [np.array([[1.0, 0], [-0.8, -0.9], [-0.5, 0], [-0.8, 0.9]])]


def iha_ciz(rng, img, boyut_px):
    h, w = img.shape[:2]
    tur = rng.choice(["sabit_kanat", "dortlu", "delta"], p=[0.55, 0.3, 0.15])
    parcalar = iha_cokgen(rng, tur)
    aci = rng.uniform(0, 2 * math.pi)
    egim = rng.uniform(0.35, 1.0)  # bakış açısından dolayı basıklık
    R = np.array([[math.cos(aci), -math.sin(aci)], [math.sin(aci), math.cos(aci)]])
    S = np.diag([1.0, egim])
    ss = 4  # alt örnekleme ile kenar yumuşatma
    yarim = boyut_px / 2
    cx = rng.uniform(-0.1 * boyut_px, w + 0.1 * boyut_px)  # bazıları kenarda kesik
    cy = rng.uniform(-0.1 * boyut_px, h + 0.1 * boyut_px)
    maske = np.zeros((h * ss, w * ss), np.uint8) if boyut_px > 40 else None
    x0, y0 = int(max(cx - boyut_px, 0)), int(max(cy - boyut_px, 0))
    x1, y1 = int(min(cx + boyut_px, w)), int(min(cy + boyut_px, h))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    yerel = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    for p in parcalar:
        q = (p @ S.T @ R.T) * yarim
        q[:, 0] += cx - x0
        q[:, 1] += cy - y0
        cv2.fillPoly(yerel, [np.round(q * ss).astype(np.int32)], 255, lineType=cv2.LINE_AA)
    alfa = cv2.resize(yerel, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA).astype(np.float32) / 255
    if alfa.max() < 0.2:
        return None
    # mesafeye bağlı bulanıklık / kontrast
    if boyut_px < 12:
        alfa = cv2.GaussianBlur(alfa, (3, 3), 0.6)
    renk = rng.uniform(20, 110) if rng.random() < 0.8 else rng.uniform(170, 240)
    kontrast = rng.uniform(0.55, 1.0)
    bolge = img[y0:y1, x0:x1]
    img[y0:y1, x0:x1] = bolge * (1 - alfa[..., None] * kontrast) + renk * alfa[..., None] * kontrast
    ys, xs = np.where(alfa > 0.25)
    if len(xs) == 0:
        return None
    return (x0 + xs.min(), y0 + ys.min(), x0 + xs.max() + 1, y0 + ys.max() + 1)


def bir_goruntu(rng, h, w):
    img = gok(rng, h, w)
    if rng.random() < 0.45:
        img = yer_ekle(rng, img)
    kutular = []
    sayi = rng.choice([0, 1, 2, 3, 4], p=[0.1, 0.6, 0.15, 0.1, 0.05])
    for _ in range(sayi):
        # boyut dağılımı: çoğu küçük (uzak), az sayıda büyük
        boyut = float(np.exp(rng.uniform(math.log(5), math.log(160))))
        k = iha_ciz(rng, img, boyut)
        if k is not None and (k[2] - k[0]) >= 2 and (k[3] - k[1]) >= 2:
            kutular.append(k)
    # sensör gürültüsü, hafif bulanıklık, pus
    if rng.random() < 0.3:
        pus = rng.uniform(0.1, 0.5)
        img = img * (1 - pus) + 210 * pus
    img = img + rng.normal(0, rng.uniform(1, 6), img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (3, 3), 0)
    return img, kutular


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--adet", type=int, default=1000)
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--genislik", type=int, default=960)
    ap.add_argument("--yukseklik", type=int, default=540)
    ap.add_argument("--onek", default="sentetik")
    args = ap.parse_args()

    kok = Path(args.cikti)
    (kok / "goruntuler").mkdir(parents=True, exist_ok=True)
    (kok / "etiketler").mkdir(parents=True, exist_ok=True)
    (kok / "siniflar.txt").write_text("iha\n", encoding="utf-8")
    rng = np.random.default_rng(args.tohum)
    for i in range(args.adet):
        img, kutular = bir_goruntu(rng, args.yukseklik, args.genislik)
        ad = f"{args.onek}_{args.tohum:02d}_{i:06d}"
        cv2.imwrite(str(kok / "goruntuler" / f"{ad}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        satirlar = []
        for x0, y0, x1, y1 in kutular:
            satirlar.append(f"0 {(x0 + x1) / 2 / args.genislik:.6f} {(y0 + y1) / 2 / args.yukseklik:.6f} "
                            f"{(x1 - x0) / args.genislik:.6f} {(y1 - y0) / args.yukseklik:.6f}")
        (kok / "etiketler" / f"{ad}.txt").write_text("\n".join(satirlar) + ("\n" if satirlar else ""),
                                                     encoding="utf-8")
    print(f"{args.adet} görüntü yazıldı: {kok}")


if __name__ == "__main__":
    main()
