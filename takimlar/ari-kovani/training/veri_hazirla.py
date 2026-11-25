#!/usr/bin/env python3
"""YOLO ya da COCO etiketli veri setlerini tek sınıflı (iha) mmdetection COCO JSON'una çevirir, istatistik çıkarır.

Görüntüler kopyalanmaz, JSON'da mutlak yollarla gösterilir. Uzun kenarı --en-buyuk-kenar'dan büyük görüntüler
(ör. MMFW-UAV'nin 3840x2160 kareleri) küçültülüp <cikti>/kucultulmus/ altına yazılır; eğitimde veri yükleyici
4K JPEG çözmekle vakit kaybetmez.

Kaynak tanımı (her --kaynak için):
    <hedef_bolum>=<kok>[:<kaynak_bolum>][,secenek=deger ...]

    YOLO klasör düzenleri otomatik bulunur:
        A) <kok>/images/<bolum>/*.jpg + <kok>/labels/<bolum>/*.txt     (Ultralytics, X-Plane çıktısı)
        B) <kok>/<bolum>/images/*.jpg + <kok>/<bolum>/labels/*.txt     (Roboflow dışa aktarımı)
        C) <kok>/goruntuler/*.jpg     + <kok>/etiketler/*.txt          (yarışma test seti)
    Sınıf adları <kok>/*.yaml içindeki 'names' alanından okunur (varsa).

    Seçenekler:
        coco=<ann.json>   kaynak COCO JSON'udur; <kok> görüntülerin klasörüdür (ör. MMFW-UAV)
        esle=<kural;...>  sınıf eşleme. Kural: <sınıf adı ya da no> > <iha|negatif|atla>; '*' diğer hepsi.
                          iha: kutu 'iha' olur. negatif: kutu silinir, görüntü hedefsiz örnek olarak kalır
                          (ör. kuş). atla: bu sınıfı içeren görüntü hiç kullanılmaz (ör. insanlı uçak).
                          Varsayılan: '*>iha'
        her=<N>           ada göre sıralı her N. görüntüyü al (videodan çıkarılmış ardışık kareler için)

Örnekler:
    python3 veri_hazirla.py --cikti $IHA_CALISMA/veri/coco \\
        --kaynak train=/veri/xplane_v4:train --kaynak val=/veri/xplane_v4:valid \\
        --kaynak "train=/veri/rf_merged:train,esle=bird>negatif;airplane>atla;helicopter>atla;*>iha" \\
        --kaynak "train=/veri/mmfw/optical_zoom,coco=/veri/mmfw/zoom_train.json,her=5"

Çıktı: <cikti>/train.json, <cikti>/val.json, <cikti>/istatistik.json
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from PIL import Image

UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp")
BOYUT_KOVALARI = [(0, 4), (4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 1e9)]
EYLEMLER = ("iha", "negatif", "atla")


# --------------------------------------------------------------------------- kaynak tanımı

def kaynak_ayristir(metin: str) -> dict:
    hedef, geri = metin.split("=", 1)
    parcalar = geri.split(",")
    yol = parcalar[0]
    kok, _, bolum = yol.partition(":") if not re.match(r"^[A-Za-z]:\\", yol) else (yol, "", "")
    k = {"hedef": hedef, "kok": Path(kok), "bolum": bolum or None, "coco": None, "esle": "*>iha", "her": 1}
    for p in parcalar[1:]:
        anahtar, _, deger = p.partition("=")
        if anahtar == "coco":
            k["coco"] = Path(deger)
        elif anahtar == "esle":
            k["esle"] = deger
        elif anahtar == "her":
            k["her"] = max(1, int(deger))
        else:
            raise SystemExit(f"Bilinmeyen kaynak seçeneği: {anahtar}")
    return k


def esleme_kur(kurallar: str) -> dict:
    esleme = {}
    for kural in filter(None, (x.strip() for x in kurallar.split(";"))):
        sinif, _, eylem = kural.partition(">")
        if eylem not in EYLEMLER:
            raise SystemExit(f"Geçersiz eşleme eylemi '{eylem}' (iha|negatif|atla): {kural}")
        esleme[sinif.strip()] = eylem
    esleme.setdefault("*", "iha")
    return esleme


def eylem_bul(esleme: dict, no: int, ad: str | None) -> str:
    for anahtar in (ad, str(no)):
        if anahtar is not None and anahtar in esleme:
            return esleme[anahtar]
    return esleme["*"]


def yaml_sinif_adlari(kok: Path) -> dict:
    """Ultralytics/Roboflow data.yaml'daki 'names' alanı -> {no: ad}."""
    for y in sorted(kok.glob("*.yaml")) + sorted(kok.glob("*.yml")):
        try:
            import yaml
            veri = yaml.safe_load(y.read_text(encoding="utf-8")) or {}
        except Exception:
            continue
        adlar = veri.get("names")
        if isinstance(adlar, list):
            return {i: str(a) for i, a in enumerate(adlar)}
        if isinstance(adlar, dict):
            return {int(i): str(a) for i, a in adlar.items()}
    return {}


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


# --------------------------------------------------------------------------- okuma

def yolo_oku(k: dict, en_kucuk_px: float, haric: frozenset, sayac: Counter):
    g, e = dizinleri_bul(k["kok"], k["bolum"])
    adlar = yaml_sinif_adlari(k["kok"])
    esleme = esleme_kur(k["esle"])
    goruntuler = [p for p in sorted(g.rglob("*")) if p.suffix.lower() in UZANTILAR]
    for i, p in enumerate(goruntuler):
        if i % k["her"]:
            continue
        if haric and p.stem.rsplit("_", 1)[0] in haric:
            sayac["haric_oturum"] += 1
            continue  # test setine ayrılmış oturum: eğitime/doğrulamaya girmez
        etiket = e / p.relative_to(g).with_suffix(".txt")
        satirlar = etiket.read_text(encoding="utf-8").splitlines() if etiket.is_file() else []
        kutular_norm, atla = [], False
        for satir in satirlar:
            parca = satir.split()
            if len(parca) < 5:
                continue
            no = int(float(parca[0]))
            eylem = eylem_bul(esleme, no, adlar.get(no))
            sayac[f"sinif:{adlar.get(no, no)}->{eylem}"] += 1
            if eylem == "atla":
                atla = True
                break
            if eylem == "iha":
                kutular_norm.append(tuple(float(v) for v in parca[1:5]))
        if atla:
            sayac["atlanan_goruntu"] += 1
            continue
        with Image.open(p) as im:
            w, h = im.size
        kutular = []
        for xc, yc, bw, bh in kutular_norm:
            x0, y0 = max((xc - bw / 2) * w, 0.0), max((yc - bh / 2) * h, 0.0)
            x1, y1 = min((xc + bw / 2) * w, w), min((yc + bh / 2) * h, h)
            if x1 - x0 >= en_kucuk_px and y1 - y0 >= en_kucuk_px:
                kutular.append((x0, y0, x1 - x0, y1 - y0))
        yield p.resolve(), w, h, kutular


def coco_oku(k: dict, en_kucuk_px: float, haric: frozenset, sayac: Counter):
    veri = json.loads(k["coco"].read_text(encoding="utf-8"))
    adlar = {c["id"]: c["name"] for c in veri.get("categories", [])}
    esleme = esleme_kur(k["esle"])
    anns = {}
    for a in veri.get("annotations", []):
        anns.setdefault(a["image_id"], []).append(a)
    goruntuler = sorted(veri["images"], key=lambda im: im["file_name"])
    for i, im in enumerate(goruntuler):
        if i % k["her"]:
            continue
        p = (k["kok"] / im["file_name"]).resolve()
        if haric and p.stem.rsplit("_", 1)[0] in haric:
            sayac["haric_oturum"] += 1
            continue
        if not p.is_file():
            sayac["bulunamayan_goruntu"] += 1
            continue
        kutular, atla = [], False
        for a in anns.get(im["id"], []):
            eylem = eylem_bul(esleme, a["category_id"], adlar.get(a["category_id"]))
            sayac[f"sinif:{adlar.get(a['category_id'], a['category_id'])}->{eylem}"] += 1
            if eylem == "atla":
                atla = True
                break
            x, y, bw, bh = a["bbox"]
            if eylem == "iha" and bw >= en_kucuk_px and bh >= en_kucuk_px:
                kutular.append((float(x), float(y), float(bw), float(bh)))
        if atla:
            sayac["atlanan_goruntu"] += 1
            continue
        w, h = im.get("width"), im.get("height")
        if not w or not h:
            with Image.open(p) as f:
                w, h = f.size
        yield p, int(w), int(h), kutular


def kucult(kayit, hedef_dizin: Path, en_buyuk: int, kok: Path):
    p, w, h, kutular = kayit
    if max(w, h) <= en_buyuk:
        return kayit
    r = en_buyuk / max(w, h)
    yeni_w, yeni_h = int(round(w * r)), int(round(h * r))
    try:
        goreli = p.relative_to(kok.resolve())
    except ValueError:
        goreli = Path(p.name)
    hedef = (hedef_dizin / goreli).with_suffix(".jpg")
    if not hedef.is_file():
        hedef.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(p) as im:
            im.convert("RGB").resize((yeni_w, yeni_h), Image.BILINEAR).save(hedef, quality=92)
    return hedef.resolve(), yeni_w, yeni_h, [(x * r, y * r, bw * r, bh * r) for x, y, bw, bh in kutular]


# --------------------------------------------------------------------------- yazma ve istatistik

def coco_yaz(kayitlar, yol: Path):
    images, annotations = [], []
    ann_id = 1
    for img_id, (p, w, h, kutular) in enumerate(kayitlar, start=1):
        images.append({"id": img_id, "file_name": str(p), "width": w, "height": h})
        for x, y, bw, bh in kutular:
            annotations.append({"id": ann_id, "image_id": img_id, "category_id": 1,
                                "bbox": [round(x, 2), round(y, 2), round(bw, 2), round(bh, 2)],
                                "area": round(bw * bh, 2), "iscrowd": 0})
            ann_id += 1
    yol.write_text(json.dumps({"images": images, "annotations": annotations,
                               "categories": [{"id": 1, "name": "iha"}]}), encoding="utf-8")


def istatistik(kayitlar) -> dict:
    kova = Counter()
    hedef_sayilari = Counter()
    for _, w, h, kutular in kayitlar:
        hedef_sayilari[min(len(kutular), 5)] += 1
        for _, _, bw, bh in kutular:
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
    ap.add_argument("--kaynak", action="append", required=True, help="bkz. yukarıdaki kaynak tanımı")
    ap.add_argument("--en-kucuk-px", type=float, default=1.0, help="bundan küçük kutuları at")
    ap.add_argument("--en-buyuk-kenar", type=int, default=1920,
                    help="uzun kenarı bundan büyük görüntüler küçültülüp <cikti>/kucultulmus/ altına yazılır (0 = kapalı)")
    ap.add_argument("--haric-oturumlar", default=None,
                    help="test setine ayrılmış oturumlar (test-seti/oturumlar.txt); bu oturumların kareleri atlanır")
    args = ap.parse_args()

    haric = frozenset()
    if args.haric_oturumlar:
        haric = frozenset(s.strip() for s in Path(args.haric_oturumlar).read_text(encoding="utf-8").splitlines()
                          if s.strip())
    cikti = Path(args.cikti)
    cikti.mkdir(parents=True, exist_ok=True)
    bolumler: dict[str, list] = {}
    kaynak_raporu = []
    for metin in args.kaynak:
        k = kaynak_ayristir(metin)
        sayac = Counter()
        okuyucu = coco_oku if k["coco"] else yolo_oku
        kayitlar = []
        for kayit in okuyucu(k, args.en_kucuk_px, haric, sayac):
            if args.en_buyuk_kenar:
                kayit = kucult(kayit, cikti / "kucultulmus" / k["kok"].name, args.en_buyuk_kenar, k["kok"])
            kayitlar.append(kayit)
        print(f"{k['hedef']:6s} <- {metin}: {len(kayitlar)} görüntü, {sum(len(x[3]) for x in kayitlar)} hedef "
              f"{dict(sayac)}", flush=True)
        kaynak_raporu.append({"kaynak": metin, "goruntu": len(kayitlar), "sayac": dict(sayac)})
        bolumler.setdefault(k["hedef"], []).extend(kayitlar)

    rapor = {"kaynaklar": kaynak_raporu}
    for bolum, kayitlar in bolumler.items():
        coco_yaz(kayitlar, cikti / f"{bolum}.json")
        rapor[bolum] = istatistik(kayitlar)
    (cikti / "istatistik.json").write_text(json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({b: rapor[b] for b in bolumler}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
