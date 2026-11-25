#!/usr/bin/env python3
"""YOLO biçimli veri setlerini mmdetection'ın beklediği COCO JSON'a çevirir ve istatistik çıkarır.

Görüntüler kopyalanmaz; JSON'da mutlak yollarla gösterilir.

Desteklenen klasör düzenleri (her --kaynak için otomatik bulunur):
    A) <kok>/images/<bolum>/*.jpg     + <kok>/labels/<bolum>/*.txt        (Ultralytics düzeni)
    B) <kok>/<bolum>/images/*.jpg     + <kok>/<bolum>/labels/*.txt        (Roboflow düzeni)
    C) <kok>/goruntuler/*.jpg         + <kok>/etiketler/*.txt             (yarışma test seti düzeni)

Örnek:
    python3 veri_hazirla.py --cikti $IHA_CALISMA/veri/coco \\
        --kaynak train=/veri/xplane_v4:train --kaynak train=/veri/gercek:train \\
        --kaynak val=/veri/xplane_v4:valid --kaynak val=/veri/gercek:valid

Her --kaynak: <hedef_bolum>=<kok>[:<kaynak_bolum>]. Çıktı:
    <cikti>/train.json, <cikti>/val.json, <cikti>/istatistik.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp")
BOYUT_KOVALARI = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]


def dizinleri_bul(kok: Path, bolum: str | None) -> tuple[Path, Path]:
    adaylar = []
    if bolum:
        adaylar += [(kok / "images" / bolum, kok / "labels" / bolum), (kok / bolum / "images", kok / bolum / "labels"),
                    (kok / bolum / "goruntuler", kok / bolum / "etiketler")]
    adaylar += [(kok / "goruntuler", kok / "etiketler"), (kok / "images", kok / "labels")]
    for g, e in adaylar:
        if g.is_dir():
            return g, e
    raise SystemExit(f"Görüntü klasörü bulunamadı: {kok} (bölüm={bolum})")


def kaynak_oku(goruntu_dizini: Path, etiket_dizini: Path, siniflar: dict, en_kucuk_px: float,
               haric: frozenset = frozenset()):
    kayitlar = []
    for p in sorted(goruntu_dizini.rglob("*")):
        if p.suffix.lower() not in UZANTILAR:
            continue
        if haric and p.stem.rsplit("_", 1)[0] in haric:
            continue  # test setine ayrılmış oturum: eğitime/doğrulamaya girmez
        etiket = etiket_dizini / p.relative_to(goruntu_dizini).with_suffix(".txt")
        with Image.open(p) as im:
            w, h = im.size
        kutular = []
        if etiket.is_file():
            for satir in etiket.read_text(encoding="utf-8").splitlines():
                parca = satir.split()
                if len(parca) < 5:
                    continue
                c = int(float(parca[0]))
                xc, yc, bw, bh = (float(v) for v in parca[1:5])
                x0, y0 = max((xc - bw / 2) * w, 0.0), max((yc - bh / 2) * h, 0.0)
                x1, y1 = min((xc + bw / 2) * w, w), min((yc + bh / 2) * h, h)
                if x1 - x0 < en_kucuk_px or y1 - y0 < en_kucuk_px:
                    continue
                kutular.append((c, x0, y0, x1 - x0, y1 - y0))
                siniflar.setdefault(c, f"sinif{c}")
        kayitlar.append((p.resolve(), w, h, kutular))
    return kayitlar


def coco_yaz(kayitlar, sinif_adlari: dict, yol: Path):
    images, annotations = [], []
    ann_id = 1
    for img_id, (p, w, h, kutular) in enumerate(kayitlar, start=1):
        images.append({"id": img_id, "file_name": str(p), "width": w, "height": h})
        for c, x, y, bw, bh in kutular:
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": c + 1,
                                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                                "area": round(bw * bh, 2), "iscrowd": 0})
            ann_id += 1
    categories = [{"id": c + 1, "name": ad} for c, ad in sorted(sinif_adlari.items())]
    yol.write_text(json.dumps({"images": images, "annotations": annotations, "categories": categories}),
                   encoding="utf-8")


def istatistik(kayitlar) -> dict:
    kova = Counter()
    hedef_sayilari = Counter()
    for _, w, h, kutular in kayitlar:
        hedef_sayilari[min(len(kutular), 5)] += 1
        for _, _, _, bw, bh in kutular:
            s = (bw * bh) ** 0.5
            for a, b in BOYUT_KOVALARI:
                if a <= s < b:
                    kova[f"{a}-{int(b) if b < 1e8 else 'ust'}px"] += 1
                    break
    return {
        "goruntu": len(kayitlar),
        "hedef": int(sum(len(k[3]) for k in kayitlar)),
        "bos_goruntu": int(sum(1 for k in kayitlar if not k[3])),
        "goruntu_basina_hedef": {("5+" if k == 5 else str(k)): v for k, v in sorted(hedef_sayilari.items())},
        "hedef_boyutu_sqrt_alan": dict(sorted(kova.items(), key=lambda x: float(x[0].split("-")[0]))),
        "cozunurlukler": dict(Counter(f"{w}x{h}" for _, w, h, _ in kayitlar).most_common(5)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--kaynak", action="append", required=True,
                    help="<hedef_bolum>=<kok>[:<kaynak_bolum>], ör. train=/veri/xplane:train")
    ap.add_argument("--sinif", action="append", default=None, help="sınıf adları sırayla (varsayılan: iha)")
    ap.add_argument("--en-kucuk-px", type=float, default=1.0, help="bundan küçük kutuları at")
    ap.add_argument("--haric-oturumlar", default=None,
                    help="test setine ayrılmış oturumlar (test-seti/oturumlar.txt); bu oturumların kareleri atlanır")
    args = ap.parse_args()

    haric = frozenset()
    if args.haric_oturumlar:
        haric = frozenset(s.strip() for s in Path(args.haric_oturumlar).read_text(encoding="utf-8").splitlines()
                          if s.strip())
    sinif_adlari = {i: ad for i, ad in enumerate(args.sinif or ["iha"])}
    bolumler: dict[str, list] = {}
    for k in args.kaynak:
        hedef, yol = k.split("=", 1)
        kok, _, bolum = yol.partition(":")
        g, e = dizinleri_bul(Path(kok), bolum or None)
        kayitlar = kaynak_oku(g, e, sinif_adlari, args.en_kucuk_px, haric)
        print(f"{hedef:6s} <- {g} : {len(kayitlar)} görüntü, {sum(len(x[3]) for x in kayitlar)} hedef")
        bolumler.setdefault(hedef, []).extend(kayitlar)

    cikti = Path(args.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    rapor = {}
    for bolum, kayitlar in bolumler.items():
        coco_yaz(kayitlar, sinif_adlari, cikti / f"{bolum}.json")
        rapor[bolum] = istatistik(kayitlar)
    rapor["siniflar"] = sinif_adlari
    (cikti / "istatistik.json").write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rapor, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
