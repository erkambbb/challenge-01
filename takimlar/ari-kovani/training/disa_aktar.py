#!/usr/bin/env python3
"""Eğitilmiş YOLOX-lite ağırlığını ONNX'e ve TIDL meta-arch prototxt'ine dışa aktarır.

TI'nin edgeai-mmdetection/tools/torch2onnx.py aracını kullanır. Çıktı klasörüne:
    model.onnx        sabit girişli (1x3xHxW) ONNX; çıkışlar dets [N,5] ve labels [N]
    model.prototxt    TIDL algılama çıkış katmanı tanımı (YOLOX, NMS dahil)

Örnek:
    python3 disa_aktar.py --ayar $IHA_CALISMA/egitim/nano960/iha_ayar.py \\
        --agirlik $IHA_CALISMA/egitim/nano960/best_coco_bbox_mAP_50_epoch_58.pth \\
        --genislik 960 --yukseklik 544 --cikti $IHA_CALISMA/disa/nano960
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BURASI = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ayar", required=True, help="mmdet yapılandırması (egit.py'nin yazdığı iha_ayar.py)")
    ap.add_argument("--agirlik", required=True)
    ap.add_argument("--genislik", type=int, required=True)
    ap.add_argument("--yukseklik", type=int, required=True)
    ap.add_argument("--cikti", required=True)
    ap.add_argument("--ornek-goruntu", default=None)
    args = ap.parse_args()

    ti_mmdet = Path(os.environ["TI_MMDET"]).resolve()
    cikti = Path(args.cikti).resolve()
    cikti.mkdir(parents=True, exist_ok=True)
    ornek = args.ornek_goruntu or str(ti_mmdet / "demo" / "demo.jpg")
    with tempfile.TemporaryDirectory() as gecici:
        komut = [sys.executable, "tools/torch2onnx.py", str(BURASI / "configs" / "deploy_onnx_static.py"),
                 str(Path(args.ayar).resolve()), str(Path(args.agirlik).resolve()), ornek,
                 "--img-size", str(args.yukseklik), str(args.genislik), "--work-dir", gecici, "--simplify"]
        print(" ".join(komut), flush=True)
        kod = subprocess.call(komut, cwd=ti_mmdet)
        if kod != 0:
            return kod
        onnxlar = sorted(Path(gecici).glob("*.onnx"))
        protolar = sorted(Path(gecici).glob("*.prototxt"))
        if not onnxlar or not protolar:
            print("HATA: ONNX ya da prototxt üretilmedi", file=sys.stderr)
            return 1
        shutil.copy(onnxlar[0], cikti / "model.onnx")
        shutil.copy(protolar[0], cikti / "model.prototxt")
    print(f"yazıldı: {cikti}/model.onnx, {cikti}/model.prototxt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
