#!/usr/bin/env python3
"""ONNX modelini AM67A (T3 Gemstone O1) C7x/MMA hızlandırıcısı için TIDL ile derler.

Derleyici sürümü kartın TIDL çalışma zamanıyla eşleşmelidir: edgeai-tidl-tools 11_00_06_00
(t3-gem-o1-edgeai 11.0 / SDK 11_00_00). 11_00_08_00 ile derlenen dosyalar kartta sürüm uyuşmazlığı verir.

Yaptıkları:
  1. ONNX'i ve TIDL meta-arch prototxt'ini çıktı klasörüne kopyalar; prototxt'te güven eşiği, NMS ve
     en fazla kutu sayısını ayarlar (algılama çıkışı + NMS de C7x'te çalışır).
  2. Kalibrasyon görüntülerini kartla AYNI önişlemle (src/ihatakip/onislem.py) hazırlar.
  3. TIDLCompilationProvider ile INT8 (isteğe göre karışık 16 bit) derler -> artifacts/.
  4. Hangi düğümlerin C7x'te, hangilerinin CPU'da çalıştığını ve katman türü / MAC dağılımını
     derleme_raporu/ altına yazar. model.json'ı üretir.

Örnek:
    source $IHA_CALISMA/ortam.env
    python3 tidl_derle.py --onnx $IHA_CALISMA/disa/nano960/model.onnx \\
        --prototxt $IHA_CALISMA/disa/nano960/model.prototxt \\
        --kalibrasyon $IHA_CALISMA/veri/kalibrasyon --adet 50 \\
        --cikti ../model/iha_yolox --ad iha_yolox_nano_lite_960x544
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI.parent / "src"))

UZANTILAR = (".jpg", ".jpeg", ".png", ".bmp")
TIDL_SURUMU = "11_00_06_00"


def prototxt_ayarla(metin: str, guven: float, nms: float, en_fazla: int) -> str:
    metin = re.sub(r"confidence_threshold:\s*[0-9.eE+-]+", f"confidence_threshold: {guven}", metin)
    metin = re.sub(r"nms_threshold:\s*[0-9.eE+-]+", f"nms_threshold: {nms}", metin)
    metin = re.sub(r"(?<!keep_)top_k:\s*\d+", f"top_k: {max(en_fazla, 200)}", metin)
    metin = re.sub(r"keep_top_k:\s*\d+", f"keep_top_k: {en_fazla}", metin)
    return metin


def karisik_hassasiyet_katmanlari(onnx_yolu: Path, prototxt: str) -> list[str]:
    """İlk konvolüsyon çıkışı + algılama başlıklarının (reg/obj/cls) konvolüsyon çıkışları.

    TI model zoo'daki YOLOX-lite yapılandırmasıyla aynı seçim: küçük kutu regresyonunun
    duyarlılığını korumak için bu tensörler 16 bit tutulur.
    """
    import onnx

    m = onnx.load(str(onnx_yolu))
    uretici = {o: n for n in m.graph.node for o in n.output}
    adlar = []
    # ONNX düğümleri topolojik sıralıdır: ilk Conv, girişe (gerekirse uint8->float Cast üzerinden) bağlı olandır
    for n in m.graph.node:
        if n.op_type == "Conv":
            adlar.append(n.output[0])
            break
    for ad in re.findall(r'input:\s*"([^"]+)"', prototxt):
        dugum = uretici.get(ad)
        if dugum is None:
            continue
        if dugum.op_type == "Concat":
            adlar += [g for g in dugum.input if g in uretici and uretici[g].op_type == "Conv"]
        elif dugum.op_type == "Conv":
            adlar.append(ad)
    return adlar


def uint8_girise_cevir(onnx_yolu: Path) -> None:
    """Model girişini uint8 yapar; float32 dönüşümü (Cast) grafın içine, dolayısıyla C7x'e taşınır.

    Böylece A53 her karede 4 kat büyük float32 tensör üretmek yerine uint8 kareyi doğrudan verir.
    """
    import onnx
    from onnx import TensorProto, helper

    m = onnx.load(str(onnx_yolu))
    giris = m.graph.input[0]
    eski = giris.name
    giris.name = eski + "_u8"
    giris.type.tensor_type.elem_type = TensorProto.UINT8
    m.graph.node.insert(0, helper.make_node("Cast", [giris.name], [eski], to=TensorProto.FLOAT, name="giris_u8_float"))
    onnx.checker.check_model(m)
    onnx.save(m, str(onnx_yolu))


def kalibrasyon_listesi(kaynaklar: list[str], adet: int, tohum: int) -> list[Path]:
    dosyalar = []
    for k in kaynaklar:
        p = Path(k)
        if p.suffix == ".json":
            veri = json.loads(p.read_text(encoding="utf-8"))
            dosyalar += [Path(im["file_name"]) for im in veri["images"]]
        elif p.is_dir():
            dosyalar += [f for f in sorted(p.rglob("*")) if f.suffix.lower() in UZANTILAR]
    if not dosyalar:
        raise SystemExit("Kalibrasyon görüntüsü bulunamadı")
    random.Random(tohum).shuffle(dosyalar)
    return dosyalar[:adet]


def _derle_alt(ayar_yolu: str) -> int:
    """Alt süreçte derleme (TIDL oturumu kapanırken çökebiliyor; ana süreç etkilenmesin)."""
    import cv2
    import onnx
    import onnxruntime as ort

    from ihatakip.onislem import letterbox, tensore_cevir

    a = json.loads(Path(ayar_yolu).read_text(encoding="utf-8"))
    onnx.shape_inference.infer_shapes_path(a["onnx"], a["onnx"])
    secenekler = a["secenekler"]
    sess = ort.InferenceSession(a["onnx"], providers=["TIDLCompilationProvider", "CPUExecutionProvider"],
                                provider_options=[secenekler, {}], sess_options=ort.SessionOptions())
    giris = sess.get_inputs()[0]
    _, _, h, w = giris.shape
    uint8 = giris.type == "tensor(uint8)"
    for yol in a["kalibrasyon"]:
        kare = cv2.imread(yol)
        tuval, _ = letterbox(kare, w, h, 114)
        x = tensore_cevir(tuval)
        sess.run(None, {giris.name: x.astype(np.uint8) if uint8 else x})
    del sess
    return 0


def netlog_ozeti(artifacts: Path) -> dict:
    ozet = {"katman_turleri": {}, "toplam_gmac": None}
    for log in sorted((artifacts / "tempDir").glob("*_netLog.txt")):
        turler = Counter()
        for satir in log.read_text(errors="replace").splitlines():
            m = re.match(r"\s*\d+\|(TIDL_\w+)\s*\|", satir)
            if m:
                turler[m.group(1)] += 1
            g = re.search(r"Total Giga Macs\s*:\s*([0-9.]+)", satir)
            if g:
                ozet["toplam_gmac"] = (ozet["toplam_gmac"] or 0) + float(g.group(1))
        for k, v in turler.items():
            ozet["katman_turleri"][k] = ozet["katman_turleri"].get(k, 0) + v
    return ozet


def yollari_gizle(rapor: Path, cikti: Path, araclar: str) -> None:
    """Repoya girecek rapor dosyalarındaki yerel mutlak yolları (kullanıcı adı içerir) yer tutucuyla değiştirir."""
    degisimler = [(str(cikti), "<model_klasoru>"), (str(Path(araclar).resolve()), "$TIDL_TOOLS_PATH"),
                  (araclar, "$TIDL_TOOLS_PATH")]
    if os.environ.get("IHA_CALISMA"):
        degisimler += [(str(Path(os.environ["IHA_CALISMA"]).resolve()), "$IHA_CALISMA"),
                       (os.environ["IHA_CALISMA"], "$IHA_CALISMA")]
    degisimler.append((str(Path.home()), "~"))
    for f in rapor.iterdir():
        if not f.is_file():
            continue
        metin = f.read_text(encoding="utf-8", errors="replace")
        for eski, yeni in degisimler:
            metin = metin.replace(eski, yeni)
        f.write_text(metin, encoding="utf-8")


def dugum_dagilimi(artifacts: Path) -> dict:
    meta = (artifacts / "onnxrtMetaData.txt").read_text()
    toplam = int(re.search(r"numGraphNodes=(\d+)", meta).group(1))
    sayilar = [int(x) for x in (artifacts / "allowedNode.txt").read_text().split()]
    alt_graf, i, aktarilan = sayilar[0], 1, 0
    for _ in range(alt_graf):
        n = sayilar[i]
        aktarilan += n
        i += 1 + n
    return {"toplam_dugum": toplam, "c7x_dugum": aktarilan, "cpu_dugum": toplam - aktarilan, "alt_graf": alt_graf}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--prototxt", required=True)
    ap.add_argument("--kalibrasyon", action="append", required=True, help="görüntü klasörü ya da COCO json")
    ap.add_argument("--adet", type=int, default=50, help="kalibrasyon kare sayısı")
    ap.add_argument("--iterasyon", type=int, default=10, help="kalibrasyon yinelemesi")
    ap.add_argument("--cikti", required=True, help="model klasörü (model.json, model.onnx, artifacts/)")
    ap.add_argument("--ad", default=None)
    ap.add_argument("--siniflar", default="iha")
    ap.add_argument("--bit", type=int, default=8, choices=[8, 16])
    ap.add_argument("--dogruluk-seviyesi", type=int, default=1, choices=[0, 1, 9])
    ap.add_argument("--niceleme-turu", type=int, default=4, choices=[0, 1, 4],
                    help="quantization_scale_type: 0 simetrik 2'nin kuvveti olmayan ölçek, 1 2'nin kuvveti, "
                         "4 asimetrik kanal başına (TI'nin AM67A için önerisi; varsayılan)")
    ap.add_argument("--otomatik-karisik", type=float, default=-1,
                    help="mixed_precision_factor: >1 ise TIDL 16 bit katmanları kendisi seçer (ör. 1.2 = %%20 ek süre payı)")
    ap.add_argument("--uint8-giris", action="store_true", default=True,
                    help="ONNX girişini uint8 yap (float dönüşümü C7x'te); A53 önişlem yükünü azaltır (varsayılan)")
    ap.add_argument("--float-giris", dest="uint8_giris", action="store_false")
    ap.add_argument("--karisik-hassasiyet", action="store_true", default=True)
    ap.add_argument("--karisik-hassasiyet-kapali", dest="karisik_hassasiyet", action="store_false")
    ap.add_argument("--cikarim-modu", type=int, default=0, choices=[0, 1, 2],
                    help="0: tek C7x, 1: yüksek verim (çoklu kare), 2: düşük gecikme (tek kare iki C7x'e bölünür)")
    ap.add_argument("--cekirdek-sayisi", type=int, default=1)
    ap.add_argument("--guven-esigi", type=float, default=0.05)
    ap.add_argument("--nms", type=float, default=0.45)
    ap.add_argument("--en-fazla-kutu", type=int, default=100)
    ap.add_argument("--canli-esik", type=float, default=0.35)
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--_alt", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args._alt:
        return _derle_alt(args._alt)

    araclar = os.environ.get("TIDL_TOOLS_PATH")
    if not araclar or not Path(araclar, "tidl_model_import.so").exists():
        raise SystemExit("TIDL_TOOLS_PATH tanımlı değil ya da geçersiz (source $IHA_CALISMA/ortam.env)")
    surum_dosyasi = Path(araclar) / "version.txt"

    cikti = Path(args.cikti).resolve()
    if cikti.exists():
        shutil.rmtree(cikti)
    (cikti / "artifacts").mkdir(parents=True)
    rapor = cikti / "derleme_raporu"
    rapor.mkdir()
    shutil.copy(args.onnx, cikti / "model.onnx")
    proto = prototxt_ayarla(Path(args.prototxt).read_text(), args.guven_esigi, args.nms, args.en_fazla_kutu)
    (cikti / "model.prototxt").write_text(proto)

    import onnx
    if args.uint8_giris:
        uint8_girise_cevir(cikti / "model.onnx")
    m = onnx.load(str(cikti / "model.onnx"))
    _, _, h, w = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim]
    del m

    kalib = kalibrasyon_listesi(args.kalibrasyon, args.adet, args.tohum)
    secenekler = {
        "tidl_tools_path": araclar,
        "artifacts_folder": str(cikti / "artifacts"),
        "platform": "J7", "version": "7.2",
        "tensor_bits": args.bit,
        "debug_level": 0,
        "max_num_subgraphs": 16,
        "accuracy_level": args.dogruluk_seviyesi,
        "advanced_options:calibration_frames": len(kalib),
        "advanced_options:calibration_iterations": args.iterasyon,
        "advanced_options:add_data_convert_ops": 3,
        "object_detection:meta_arch_type": 6,
        "object_detection:meta_layers_names_list": str(cikti / "model.prototxt"),
    }
    if args.niceleme_turu:
        secenekler["advanced_options:quantization_scale_type"] = args.niceleme_turu
    if args.otomatik_karisik > 0:
        secenekler["advanced_options:mixed_precision_factor"] = args.otomatik_karisik
    if args.cikarim_modu:
        secenekler["advanced_options:inference_mode"] = args.cikarim_modu
        secenekler["advanced_options:num_cores"] = args.cekirdek_sayisi
    karisik = []
    if args.karisik_hassasiyet and args.bit == 8:
        karisik = karisik_hassasiyet_katmanlari(cikti / "model.onnx", proto)
        secenekler["advanced_options:output_feature_16bit_names_list"] = ", ".join(karisik)

    ayar = {"onnx": str(cikti / "model.onnx"), "secenekler": secenekler, "kalibrasyon": [str(p) for p in kalib]}
    ayar_yolu = rapor / "derleme_ayari.json"
    ayar_yolu.write_text(json.dumps(ayar, ensure_ascii=False, indent=2), encoding="utf-8")

    t0 = time.time()
    with open(rapor / "derleme.log", "w") as log:
        kod = subprocess.call([sys.executable, __file__, "--onnx", "-", "--prototxt", "-", "--kalibrasyon", "-",
                               "--cikti", "-", "--_alt", str(ayar_yolu)], stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "LD_LIBRARY_PATH": araclar + ":" + os.environ.get("LD_LIBRARY_PATH", "")})
    sure = time.time() - t0
    art = cikti / "artifacts"
    if not list(art.glob("subgraph_*_tidl_net.bin")) or not (art / "allowedNode.txt").exists():
        print(f"HATA: derleme başarısız (çıkış kodu {kod}); ayrıntı: {rapor / 'derleme.log'}", file=sys.stderr)
        return 1
    if kod != 0:
        print(f"UYARI: derleme alt süreci {kod} koduyla bitti (oturum kapanışı); dosyalar üretildi, devam.")

    dagilim = dugum_dagilimi(art)
    katman = netlog_ozeti(art)
    for f in (art / "tempDir").glob("*_netLog.txt"):
        shutil.copy(f, rapor / f.name)
    for f in (art / "tempDir").glob("*layer_info.txt"):
        shutil.copy(f, rapor / f.name)
    shutil.rmtree(art / "tempDir", ignore_errors=True)
    yollari_gizle(rapor, cikti, araclar)

    meta = {
        "ad": args.ad or cikti.name,
        "onnx": "model.onnx",
        "artifacts": "artifacts",
        "giris": {"genislik": w, "yukseklik": h, "dolgu": 114, "kanal_sirasi": "BGR", "yerlesim": "NCHW",
                  "ortalama": [0.0, 0.0, 0.0], "olcek": [1.0, 1.0, 1.0]},
        "siniflar": args.siniflar.split(","),
        "degerlendirme_esigi": args.guven_esigi,
        "canli_esik": args.canli_esik,
        "derleme": {
            "edgeai_tidl_tools": TIDL_SURUMU,
            "tidl_tools_version_txt": surum_dosyasi.read_text().strip().replace("\n", " ") if surum_dosyasi.exists() else "",
            "hedef": "AM67A (J722S)",
            "tensor_bits": args.bit,
            "accuracy_level": args.dogruluk_seviyesi,
            "quantization_scale_type": args.niceleme_turu,
            "uint8_giris": args.uint8_giris,
            "mixed_precision_factor": args.otomatik_karisik,
            "kalibrasyon_kare": len(kalib),
            "kalibrasyon_iterasyon": args.iterasyon,
            "karisik_hassasiyet_16bit_tensorler": karisik,
            "inference_mode": args.cikarim_modu,
            "num_cores": args.cekirdek_sayisi,
            "prototxt": {"guven_esigi": args.guven_esigi, "nms": args.nms, "en_fazla_kutu": args.en_fazla_kutu},
            "sure_sn": round(sure, 1),
        },
        "hizlandirici": {**dagilim, **katman},
    }
    (cikti / "model.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    satirlar = [f"# Derleme raporu: {meta['ad']}", "",
                f"- Giriş: {w}x{h}, BGR, 0-255, NCHW (letterbox, dolgu 114)",
                f"- Derleyici: edgeai-tidl-tools {TIDL_SURUMU}, hedef AM67A, {args.bit} bit, accuracy_level "
                f"{args.dogruluk_seviyesi}, kalibrasyon {len(kalib)} kare x {args.iterasyon} yineleme",
                f"- 16 bit tutulan tensörler: {', '.join(karisik) if karisik else 'yok'}",
                f"- ONNX Runtime düğümleri: toplam {dagilim['toplam_dugum']}, C7x/MMA (TIDL) {dagilim['c7x_dugum']}, "
                f"CPU {dagilim['cpu_dugum']}, TIDL alt graf sayısı {dagilim['alt_graf']}",
                f"- Toplam işlem: {katman['toplam_gmac']} GMAC", "", "| TIDL katman türü | Adet |", "|---|---|"]
    satirlar += [f"| {k} | {v} |" for k, v in sorted(katman["katman_turleri"].items(), key=lambda x: -x[1])]
    (rapor / "OZET.md").write_text("\n".join(satirlar) + "\n", encoding="utf-8")
    print("\n".join(satirlar))
    print(f"\nderleme süresi {sure:.0f} sn -> {cikti}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
