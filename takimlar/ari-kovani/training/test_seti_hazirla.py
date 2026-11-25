#!/usr/bin/env python3
"""Gerçek çekimlerden yarışma test setini hazırlama araçları.

Alt komutlar:
  kare-cikar   videodan her N. kareyi çıkarır (etiketlemeye hazırlık). Dosya adı oturum adını taşır.
  on-etiketle  mevcut modelle taslak YOLO etiketleri yazar (Label Studio / CVAT'ta düzeltmek için).
  paketle      etiketlenmiş kareleri yarışmanın "Nesne tespiti" biçimine dönüştürür:
               siniflar.txt, goruntuler/, etiketler/; en fazla 500 örnek, oturumlar dengeli, adlar kurala uygun.

Sızıntı kuralı: test setine giren oturumlar (çekim videoları) eğitim/doğrulamada KULLANILMAZ. paketle,
seçilen oturumların listesini <cikti>/oturumlar.txt olarak yazar; veri_hazirla.py'ye verilen eğitim
kaynaklarından bu oturumlar çıkarılmalıdır.

Örnekler:
  python3 test_seti_hazirla.py kare-cikar --video kayit_01.mp4 --oturum ucus01 --her 10 --cikti /veri/gercek/kareler
  python3 test_seti_hazirla.py on-etiketle --goruntuler /veri/gercek/kareler --model ../model/iha_yolox --esik 0.3
  python3 test_seti_hazirla.py paketle --goruntuler /veri/gercek/kareler --etiketler /veri/gercek/etiketler \\
      --cikti ../test-seti --en-fazla 500 --bos-oran 0.15
"""

from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

BURASI = Path(__file__).resolve().parent
UZANTILAR = (".jpg", ".jpeg", ".png")


def kare_cikar(args):
    import cv2

    cikti = Path(args.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    oturum = re.sub(r"[^a-z0-9-]", "-", args.oturum.lower())
    cap = cv2.VideoCapture(args.video)
    i = yazilan = 0
    while True:
        ok, kare = cap.read()
        if not ok:
            break
        if i % args.her == 0:
            cv2.imwrite(str(cikti / f"{oturum}_{i:06d}.jpg"), kare, [cv2.IMWRITE_JPEG_QUALITY, 95])
            yazilan += 1
        i += 1
    print(f"{args.video}: {i} kare okundu, {yazilan} kare yazıldı -> {cikti}")


def on_etiketle(args):
    import cv2

    sys.path.insert(0, str(BURASI.parent / "src"))
    from ihatakip.algilayici import Algilayici

    a = Algilayici(args.model, mod=args.mod, esik=args.esik)
    hedef = Path(args.etiketler or Path(args.goruntuler).parent / "etiketler_taslak")
    hedef.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(Path(args.goruntuler).iterdir()):
        if p.suffix.lower() not in UZANTILAR:
            continue
        kare = cv2.imread(str(p))
        h, w = kare.shape[:2]
        satirlar = []
        for x1, y1, x2, y2, s, c in a.tespit_et(kare):
            satirlar.append(f"{int(c)} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
        (hedef / f"{p.stem}.txt").write_text("\n".join(satirlar) + ("\n" if satirlar else ""), encoding="utf-8")
        n += 1
    print(f"{n} görüntü için taslak etiket yazıldı -> {hedef} (elle düzeltin!)")


def paketle(args):
    goruntuler = sorted(p for p in Path(args.goruntuler).iterdir() if p.suffix.lower() in UZANTILAR)
    etiket_dizini = Path(args.etiketler)
    oturumlar = defaultdict(list)
    for p in goruntuler:
        e = etiket_dizini / f"{p.stem}.txt"
        if not e.is_file():
            continue  # etiketlenmemiş kare test setine girmez
        oturumlar[p.stem.rsplit("_", 1)[0]].append((p, e))
    rng = random.Random(args.tohum)
    dolu, bos = [], []
    for ot, liste in sorted(oturumlar.items()):
        for p, e in liste:
            (dolu if e.read_text(encoding="utf-8").strip() else bos).append((ot, p, e))
    n_bos = min(len(bos), int(args.en_fazla * args.bos_oran))
    n_dolu = min(len(dolu), args.en_fazla - n_bos)

    def dengeli(liste, n):
        # oturumlar arasında sırayla seç: tek bir uzun çekim test setine baskın olmasın
        gruplar = defaultdict(list)
        for x in liste:
            gruplar[x[0]].append(x)
        for g in gruplar.values():
            rng.shuffle(g)
        secilen, anahtarlar = [], sorted(gruplar)
        while len(secilen) < n and any(gruplar.values()):
            for k in anahtarlar:
                if gruplar[k] and len(secilen) < n:
                    secilen.append(gruplar[k].pop())
        return secilen

    secim = dengeli(dolu, n_dolu) + dengeli(bos, n_bos)
    cikti = Path(args.cikti)
    for alt in ("goruntuler", "etiketler"):
        if (cikti / alt).exists():
            shutil.rmtree(cikti / alt)
        (cikti / alt).mkdir(parents=True)
    (cikti / "siniflar.txt").write_text("\n".join(args.sinif) + "\n", encoding="utf-8")
    for ot, p, e in sorted(secim, key=lambda x: x[1].name):
        ad = re.sub(r"[^a-z0-9_-]", "-", p.stem.lower())
        shutil.copy(p, cikti / "goruntuler" / f"{ad}{p.suffix.lower()}")
        shutil.copy(e, cikti / "etiketler" / f"{ad}.txt")
    secilen_oturumlar = sorted({ot for ot, _, _ in secim})
    (cikti / "oturumlar.txt").write_text("\n".join(secilen_oturumlar) + "\n", encoding="utf-8")
    print(f"{len(secim)} örnek ({n_dolu} hedefli, {n_bos} boş), {len(secilen_oturumlar)} oturum -> {cikti}")
    print(f"Bu oturumları eğitim/doğrulama verisinden çıkarın: {cikti / 'oturumlar.txt'}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    alt = ap.add_subparsers(dest="komut", required=True)
    k = alt.add_parser("kare-cikar")
    k.add_argument("--video", required=True)
    k.add_argument("--oturum", required=True, help="çekim adı (küçük harf, rakam, tire)")
    k.add_argument("--her", type=int, default=10)
    k.add_argument("--cikti", required=True)
    o = alt.add_parser("on-etiketle")
    o.add_argument("--goruntuler", required=True)
    o.add_argument("--model", required=True)
    o.add_argument("--etiketler", default=None)
    o.add_argument("--mod", default="cpu", choices=["cpu", "tidl", "otomatik"])
    o.add_argument("--esik", type=float, default=0.3)
    p = alt.add_parser("paketle")
    p.add_argument("--goruntuler", required=True)
    p.add_argument("--etiketler", required=True)
    p.add_argument("--cikti", required=True)
    p.add_argument("--en-fazla", type=int, default=500)
    p.add_argument("--bos-oran", type=float, default=0.15, help="hedefsiz kare oranı (yanlış alarm ölçümü için)")
    p.add_argument("--sinif", action="append", default=None)
    p.add_argument("--tohum", type=int, default=0)
    args = ap.parse_args()
    if args.komut == "paketle" and not args.sinif:
        args.sinif = ["iha"]
    {"kare-cikar": kare_cikar, "on-etiketle": on_etiketle, "paketle": paketle}[args.komut](args)


if __name__ == "__main__":
    main()
