#!/usr/bin/env bash
# Değerlendirme betiği. olcum/calistir.py bu dosyayı takım klasöründe, internetsiz ve root olarak çalıştırır.
#
# Tanımlı ortam değişkenleri (calistir.py verir):
#   TAKIM_DIZINI   takım klasörünün çalışma kopyası
#   TEST_SETI      test-seti/ klasörünün yolu
#   CIKTI_DIZINI   sonuçların yazılacağı klasör
#
# İsteğe bağlı:
#   IHA_MODEL           model klasörü (varsayılan: model/iha_yolox)
#   CPU_KARSILASTIRMA   ilk N test görüntüsünde aynı modeli CPU'da float32 çalıştırıp TIDL INT8 ile
#                       karşılaştırır (varsayılan 50; 0 = kapalı)
#
# Çıktılar: $CIKTI_DIZINI/sonuc.json (metrik_adi: mAP@0.5), tahminler.json, ayrintili_ozet.json
set -euo pipefail

cd "${TAKIM_DIZINI:-$(dirname "$0")}"
MODEL="${IHA_MODEL:-model/iha_yolox}"

python3 src/degerlendir.py \
    --test-seti "$TEST_SETI" \
    --model "$MODEL" \
    --sonuc "$CIKTI_DIZINI/sonuc.json" \
    --cpu-karsilastirma "${CPU_KARSILASTIRMA:-50}"
