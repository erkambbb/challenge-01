#!/usr/bin/env bash
# Model seçimi için hız ölçüm varyantlarını derler (COCO ağırlıklarıyla; yalnız gecikme ölçümü içindir,
# doğruluk için değil). Çıktı: $IHA_CALISMA/varyantlar/<ad>/ (model.json + model.onnx + artifacts/).
# Kartta: python3 src/hiz_olc.py --modeller varyantlar/* --rapor hiz_raporu.md
#
# Kullanım: source $IHA_CALISMA/ortam.env && bash varyant_derle.sh [kalibrasyon_klasoru]
set -euo pipefail

BURASI="$(cd "$(dirname "$0")" && pwd)"
KALIB="${1:-$IHA_CALISMA/veri/sentetik_duman/val/goruntuler}"
DISA="$IHA_CALISMA/disa/coco"
CIKTI="$IHA_CALISMA/varyantlar"
mkdir -p "$DISA" "$CIKTI"

declare -A AGIRLIK=(
  [nano]=yolox_nano_lite_416x416_20220214_checkpoint.pth
  [tiny]=yolox_tiny_lite_416x416_20220217_checkpoint.pth
  [s]=yolox_s_lite_640x640_20220221_checkpoint.pth
)

# model genislik yukseklik cikarim_modu cekirdek karisik(1/0)
VARYANTLAR=(
  "nano 416 416 0 1 1"
  "nano 640 640 0 1 1"
  "nano 960 544 0 1 1"
  "nano 960 544 0 1 0"
  "nano 960 544 2 2 1"
  "nano 1280 736 0 1 1"
  "tiny 640 640 0 1 1"
  "tiny 960 544 0 1 1"
  "tiny 960 544 2 2 1"
  "s 640 640 0 1 1"
  "s 960 544 0 1 1"
  "s 960 544 2 2 1"
)

for v in "${VARYANTLAR[@]}"; do
  read -r M W H IM C K <<<"$v"
  AD="coco_${M}_${W}x${H}_im${IM}$([ "$K" = 1 ] && echo _k16 || echo _int8)"
  if [ -f "$CIKTI/$AD/model.json" ]; then echo "var: $AD"; continue; fi
  E="$DISA/${M}_${W}x${H}"
  if [ ! -f "$E/model.onnx" ]; then
    "$IHA_EGITIM_PY" "$BURASI/disa_aktar.py" --ayar "$TI_MMDET/configs_edgeailite/yolox/yolox_${M}_lite.py" \
      --agirlik "$IHA_CALISMA/modelzoo/${AGIRLIK[$M]}" --genislik "$W" --yukseklik "$H" --cikti "$E" \
      > "$E.log" 2>&1 || { echo "dışa aktarma hatası: $E.log"; continue; }
  fi
  EK=()
  [ "$K" = 0 ] && EK+=(--karisik-hassasiyet-kapali)
  "$IHA_TIDL_PY" "$BURASI/tidl_derle.py" --onnx "$E/model.onnx" --prototxt "$E/model.prototxt" \
    --kalibrasyon "$KALIB" --adet 6 --iterasyon 2 --cikti "$CIKTI/$AD" --ad "$AD" --siniflar coco \
    --cikarim-modu "$IM" --cekirdek-sayisi "$C" "${EK[@]}" > "$CIKTI/$AD.log" 2>&1 \
    && echo "tamam: $AD" || echo "derleme hatası: $CIKTI/$AD.log"
done
