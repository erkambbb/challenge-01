#!/usr/bin/env python3
"""Derlenmiş model klasörlerinin kart üzerindeki gecikmesini ölçer (model seçimi için).

Her model klasörü (model.json + model.onnx + artifacts/) için sabit bir görüntüyle N kez çalıştırır;
uçtan uca (önişlem + çıkarım + son işlem) ve C7x işlem süresini raporlar, Markdown tablo yazar.

Kartta (root, 'source /opt/t3-edgeai-env' sonrası):
    python3 src/hiz_olc.py --modeller /root/varyantlar/* --tekrar 100 --rapor hiz_raporu.md
    python3 src/hiz_olc.py --modeller model/iha_yolox --cpu      # aynı modelin CPU float32 süresi
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ihatakip.algilayici import Algilayici  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--modeller", nargs="+", required=True)
    ap.add_argument("--tekrar", type=int, default=100)
    ap.add_argument("--isinma", type=int, default=10)
    ap.add_argument("--cpu", action="store_true", help="TIDL yerine CPU float32")
    ap.add_argument("--goruntu", default=None, help="ölçümde kullanılacak görüntü (varsayılan: rastgele 1280x720)")
    ap.add_argument("--cekirdek", type=int, default=1, help="C7x çekirdek numarası (1 ya da 2)")
    ap.add_argument("--rapor", default=None)
    args = ap.parse_args()

    if args.goruntu:
        import cv2
        kare = cv2.imread(args.goruntu)
    else:
        kare = np.random.default_rng(0).integers(0, 255, (720, 1280, 3), dtype=np.uint8)

    satirlar = ["| Model | Giriş | GMAC | Mod | Uçtan uca ort (ms) | p95 (ms) | C7x (ms) | FPS |",
                "|---|---|---|---|---|---|---|---|"]
    for yol in args.modeller:
        yol = Path(yol)
        if not (yol / "model.json").is_file():
            continue
        meta = json.loads((yol / "model.json").read_text(encoding="utf-8"))
        try:
            a = Algilayici(yol, mod="cpu" if args.cpu else "tidl", cekirdek=args.cekirdek)
        except Exception as hata:  # bir varyant çalışmazsa diğerlerine devam
            print(f"{yol.name}: HATA {hata}", flush=True)
            satirlar.append(f"| {yol.name} | - | - | hata | {hata} | | | |")
            continue
        for _ in range(args.isinma):
            a.tespit_et(kare)
        sureler, c7x = [], []
        for _ in range(args.tekrar):
            a.tespit_et(kare)
            sureler.append(a.son_sureler["toplam_ms"])
            t = a.ti_olcumleri()
            if "c7x_islem_ms" in t:
                c7x.append(t["c7x_islem_ms"])
        s = np.asarray(sureler)
        mod = "CPU fp32" if args.cpu else f"TIDL im={meta.get('derleme', {}).get('inference_mode', 0)}"
        gmac = meta.get("hizlandirici", {}).get("toplam_gmac")
        satir = (f"| {meta.get('ad', yol.name)} | {a.bilgi.genislik}x{a.bilgi.yukseklik} | "
                 f"{gmac if gmac is not None else '-'} | {mod} | {s.mean():.2f} | {np.percentile(s, 95):.2f} | "
                 f"{np.mean(c7x):.2f} | {1000 / s.mean():.1f} |" if c7x else
                 f"| {meta.get('ad', yol.name)} | {a.bilgi.genislik}x{a.bilgi.yukseklik} | "
                 f"{gmac if gmac is not None else '-'} | {mod} | {s.mean():.2f} | {np.percentile(s, 95):.2f} | - | "
                 f"{1000 / s.mean():.1f} |")
        print(satir, flush=True)
        satirlar.append(satir)
        del a
        time.sleep(0.5)
    metin = "\n".join(satirlar) + "\n"
    print("\n" + metin)
    if args.rapor:
        Path(args.rapor).write_text(metin, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
