#!/usr/bin/env python3
"""Test seti üzerinde İHA algılayıcısını değerlendirir ve sonuc.json yazar.

Kullanım (degerlendir.sh bunu çağırır):
    python3 src/degerlendir.py --test-seti test-seti --model model/iha_yolox --sonuc cikti/sonuc.json

Birincil metrik mAP@0.5 (tek sınıf olduğundan AP@0.5). Ek metrikler: AP@[.5:.95], AP@0.75,
çalışma eşiğinde hassasiyet/duyarlılık/F1, hedef boyutuna göre duyarlılık, boş karede yanlış alarm,
uçtan uca gecikme ve C7x işlem süresi. İsteğe bağlı olarak aynı ONNX modeli yalnız CPU'da float32
olarak da çalıştırılıp INT8 (TIDL) ile karşılaştırılır.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402

from ihatakip.algilayici import Algilayici  # noqa: E402
from ihatakip.metrikler import TespitDegerlendirici  # noqa: E402
from ihatakip.veri import ornekleri_listele, siniflari_oku, yolo_etiketi_oku  # noqa: E402


def calistir(algilayici: Algilayici, ornekler, isinma: int, sinif_sayisi: int, calisma_esigi: float,
             tahmin_kaydi: list | None = None, en_fazla: int | None = None):
    degerlendirici = TespitDegerlendirici(sinif_sayisi=sinif_sayisi, calisma_esigi=calisma_esigi)
    gecikmeler, c7x = [], []
    if en_fazla:
        ornekler = ornekler[:en_fazla]
    # ısınma: ilk çalıştırmalar (bellek ayırma, önbellek) gecikme istatistiğine katılmaz
    if ornekler:
        ilk = cv2.imread(str(ornekler[0][0]))
        for _ in range(isinma):
            algilayici.tespit_et(ilk)
    for goruntu_yolu, etiket_yolu in ornekler:
        kare = cv2.imread(str(goruntu_yolu))
        if kare is None:
            raise RuntimeError(f"Görüntü okunamadı: {goruntu_yolu}")
        tespit = algilayici.tespit_et(kare)
        gecikmeler.append(algilayici.son_sureler["toplam_ms"])
        ti = algilayici.ti_olcumleri()
        if "c7x_islem_ms" in ti:
            c7x.append(ti["c7x_islem_ms"])
        gercek = yolo_etiketi_oku(etiket_yolu, kare.shape[1], kare.shape[0])
        degerlendirici.ekle(tespit, gercek)
        if tahmin_kaydi is not None:
            tahmin_kaydi.append({"goruntu": goruntu_yolu.name,
                                 "tespitler": [[round(float(v), 2) for v in t[:5]] + [int(t[5])] for t in tespit]})
    return degerlendirici, np.asarray(gecikmeler), np.asarray(c7x)


def duzlestir(onek: str, sozluk: dict, hedef: dict) -> None:
    for ad, deger in sozluk.items():
        if isinstance(deger, dict):
            duzlestir(f"{onek}{ad}/", deger, hedef)
        elif isinstance(deger, (int, float)) and not isinstance(deger, bool) and deger is not None \
                and np.isfinite(deger):
            hedef[f"{onek}{ad}"] = round(float(deger), 5)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test-seti", required=True)
    ap.add_argument("--model", required=True, help="model.json içeren klasör")
    ap.add_argument("--sonuc", required=True, help="yazılacak sonuc.json yolu")
    ap.add_argument("--mod", default="otomatik", choices=["otomatik", "tidl", "cpu"])
    ap.add_argument("--calisma-esigi", type=float, default=None,
                    help="hassasiyet/duyarlılık için skor eşiği (varsayılan: model.json'daki canli_esik)")
    ap.add_argument("--isinma", type=int, default=5)
    ap.add_argument("--cpu-karsilastirma", type=int, default=0,
                    help="ilk N örnekte float32 CPU çalıştırıp INT8 TIDL ile karşılaştır (0 = kapalı)")
    args = ap.parse_args()

    test_seti = Path(args.test_seti)
    sonuc_yolu = Path(args.sonuc)
    sonuc_yolu.parent.mkdir(parents=True, exist_ok=True)
    siniflar = siniflari_oku(test_seti)
    ornekler = ornekleri_listele(test_seti)
    meta = json.loads((Path(args.model) / "model.json").read_text(encoding="utf-8"))
    calisma_esigi = args.calisma_esigi if args.calisma_esigi is not None else float(meta.get("canli_esik", 0.3))

    t0 = time.time()
    algilayici = Algilayici(args.model, mod=args.mod, esik=float(meta.get("degerlendirme_esigi", 0.01)))
    print(f"[degerlendir] model={algilayici.bilgi.ad} mod={algilayici.mod} "
          f"giris={algilayici.bilgi.genislik}x{algilayici.bilgi.yukseklik} ornek={len(ornekler)}", flush=True)
    tahminler: list = []
    degerlendirici, gecikme, c7x = calistir(algilayici, ornekler, args.isinma, len(siniflar), calisma_esigi,
                                            tahminler)
    ozet = degerlendirici.ozet()
    print(f"[degerlendir] AP@0.5={ozet['AP@0.5']:.4f} AP@[.5:.95]={ozet['AP@[.5:.95]']:.4f} "
          f"sure={time.time() - t0:.1f}s", flush=True)

    ek: dict = {}
    duzlestir("", {k: v for k, v in ozet.items() if k not in ("AP@0.5",)}, ek)
    if len(c7x):
        ek["c7x_islem_ort_ms"] = round(float(c7x.mean()), 3)
    ek["calisma_modu_tidl"] = 1 if algilayici.mod == "tidl" else 0

    if args.cpu_karsilastirma and algilayici.mod == "tidl":
        del algilayici
        cpu = Algilayici(args.model, mod="cpu", esik=float(meta.get("degerlendirme_esigi", 0.01)))
        n = args.cpu_karsilastirma
        d_cpu, g_cpu, _ = calistir(cpu, ornekler, 1, len(siniflar), calisma_esigi, None, n)
        # aynı alt kümede TIDL sonucu: kaydedilmiş tahminlerden yeniden hesapla
        d_tidl_alt = TespitDegerlendirici(sinif_sayisi=len(siniflar), calisma_esigi=calisma_esigi)
        for (tespit, gercek) in degerlendirici.goruntuler[:n]:
            d_tidl_alt.ekle(tespit, gercek)
        ek["karsilastirma/ornek_sayisi"] = min(n, len(ornekler))
        ek["karsilastirma/cpu_float32_AP@0.5"] = round(d_cpu.ap(0.5), 5)
        ek["karsilastirma/tidl_int8_AP@0.5"] = round(d_tidl_alt.ap(0.5), 5)
        ek["karsilastirma/cpu_float32_gecikme_ort_ms"] = round(float(g_cpu.mean()), 3)
        ek["karsilastirma/tidl_int8_gecikme_ort_ms"] = round(float(gecikme[:n].mean()), 3)
        print(f"[degerlendir] CPU float32 AP@0.5={ek['karsilastirma/cpu_float32_AP@0.5']:.4f} "
              f"gecikme={ek['karsilastirma/cpu_float32_gecikme_ort_ms']:.1f} ms", flush=True)

    sonuc = {
        "metrik_adi": "mAP@0.5",
        "metrik_degeri": round(float(ozet["AP@0.5"]), 5),
        "fps": round(float(1000.0 / gecikme.mean()), 3) if len(gecikme) else None,
        "gecikme_ort_ms": round(float(gecikme.mean()), 3) if len(gecikme) else None,
        "gecikme_p95_ms": round(float(np.percentile(gecikme, 95)), 3) if len(gecikme) else None,
        "ornek_sayisi": len(ornekler),
        "ek_metrikler": ek,
    }
    sonuc = {k: v for k, v in sonuc.items() if v is not None}
    sonuc_yolu.write_text(json.dumps(sonuc, ensure_ascii=False, indent=2), encoding="utf-8")
    (sonuc_yolu.parent / "tahminler.json").write_text(json.dumps(tahminler, ensure_ascii=False), encoding="utf-8")
    (sonuc_yolu.parent / "ayrintili_ozet.json").write_text(json.dumps(ozet, ensure_ascii=False, indent=2,
                                                                       default=float), encoding="utf-8")
    print(json.dumps(sonuc, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
