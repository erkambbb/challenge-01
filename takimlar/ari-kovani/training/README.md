# training/

Modeli eğitmek, ONNX'e dışa aktarmak ve AM67A (C7x/MMA) için TIDL ile derlemek için kullanılan betikler. Hepsi kartta değil, x86_64 bir Linux bilgisayarda çalışır (NVIDIA GPU ile eğitim). Eğitim verisi bu klasörde değildir.

## Ortam

```bash
IHA_CALISMA=~/iha_calisma bash ortam_kur.sh     # bir kez; Docker gerekmez, uv ile Python 3.10
source ~/iha_calisma/ortam.env
```

Sürümler bilerek sabittir:

| Bileşen | Sürüm | Neden |
|---|---|---|
| edgeai-tidl-tools | **11_00_06_00** | Kartın çalışma zamanı TIDL 11.0 (`t3-gem-o1-edgeai` 11.0, SDK 11_00_00). 11_00_08_00 ile derlenen dosyalar kartta sürüm uyuşmazlığı verir. |
| onnxruntime-tidl | 1.15.0 (cp310) | edgeai-tidl-tools 11_00_06_00 ile gelen derleme/emülasyon sağlayıcısı |
| PyTorch / CUDA | 2.1.2 / 12.1 | mmcv 2.1.0'ın hazır derlenmiş paketi olan en yeni eşleşme |
| mmcv / mmengine / mmdet | 2.1.0 / 0.10.7 / 3.3 (TI) | edgeai-mmdetection'ın istediği aralık |

## Akış

```
veri_hazirla.py  YOLO etiketli veri -> COCO JSON (+ kutu boyutu istatistiği)
egit.py          YOLOX-lite ince ayar (TI COCO ağırlığından; tek sınıf başlığı COCO "airplane"dan)
disa_aktar.py    .pth -> model.onnx + model.prototxt (TIDL algılama çıkış katmanı)
tidl_derle.py    model.onnx -> ../model/<ad>/ (artifacts/, model.json, derleme_raporu/)
```

Örnek (tam veri seti):

```bash
source ~/iha_calisma/ortam.env
$IHA_EGITIM_PY veri_hazirla.py --cikti $IHA_CALISMA/veri/coco \
    --kaynak train=/veri/xplane_v4:train --kaynak val=/veri/xplane_v4:valid \
    --kaynak train=/veri/gercek:train   --kaynak val=/veri/gercek:valid
$IHA_EGITIM_PY egit.py --ad nano960 --model nano --genislik 960 --yukseklik 544 \
    --train $IHA_CALISMA/veri/coco/train.json --val $IHA_CALISMA/veri/coco/val.json --epok 60 --batch 16
$IHA_EGITIM_PY disa_aktar.py --ayar $IHA_CALISMA/egitim/nano960/iha_ayar.py \
    --agirlik $IHA_CALISMA/egitim/nano960/best_coco_bbox_mAP_50_epoch_*.pth \
    --genislik 960 --yukseklik 544 --cikti $IHA_CALISMA/disa/nano960
$IHA_TIDL_PY tidl_derle.py --onnx $IHA_CALISMA/disa/nano960/model.onnx \
    --prototxt $IHA_CALISMA/disa/nano960/model.prototxt \
    --kalibrasyon $IHA_CALISMA/veri/coco/train.json --adet 50 \
    --cikti ../model/iha_yolox --ad iha_yolox_nano_lite_960x544
```

Derlenen model, kart olmadan da bilgisayarda TIDL emülasyonuyla denenebilir (aynı INT8 sonuçları):

```bash
TIDL_TOOLS_PATH=... $IHA_TIDL_PY ../src/degerlendir.py --test-seti ../test-seti --model ../model/iha_yolox \
    --sonuc /tmp/sonuc.json --mod tidl --cpu-karsilastirma 60
```

## Diğer betikler

- `sentetik_ornek_uret.py`: prosedürel sentetik İHA görüntüleri (gökyüzü, bulut, ufuk, şehir dokusu, sabit kanat / dört rotorlu / delta siluetleri). Yalnız zinciri uçtan uca sınamak (duman testi) içindir; asıl model bununla eğitilmez.
- `metrik_dogrula.py`: `src/ihatakip/metrikler.py`'nin AP hesabını pycocotools ile karşılaştırır.
- `varyant_derle.sh`: model seçimi için farklı boyut / çözünürlük / çekirdek modlarında hız ölçüm varyantları derler; kartta `src/hiz_olc.py` ile ölçülür.
- `configs/deploy_onnx_static.py`: dışa aktarma ayarı; ONNX içindeki son işlem eşikleri TIDL prototxt'iyle aynı tutulur ki CPU float32 ve TIDL INT8 sonuçları karşılaştırılabilsin.

## Eğitim ayrıntıları

- Başlangıç: TI edgeai-modelzoo YOLOX-lite COCO ağırlıkları. "lite" dönüşümü (SiLU -> ReLU vb.) eğitimden önce uygulanır, ağırlıklar ondan sonra yüklenir; böylece eğitilen ağ derlenen ağla birebir aynıdır.
- Tek sınıflı başlık, COCO'nun 80 sınıflı başlığındaki "airplane" (indeks 4) satırından başlatılır (`--coco-sinifi 4`).
- Artırmalar: mozaik, rastgele afin (ölçek 0,5-1,5; küçük hedefler kaybolmasın diye COCO'daki 0,1-2 yerine), HSV, yatay çevirme; son epoklarda artırmasız (YOLOXModeSwitchHook). Çok ölçekli eğitim yüksekliğin ±%25'i aralığında, en-boy oranı korunarak.
- Optimizasyon: SGD + Nesterov, öğrenme oranı 0,01 × batch / 64, kuadratik ısınma + kosinüs, EMA, AMP. En iyi ağırlık doğrulama mAP@0.5'e göre seçilir.
- İsteğe bağlı QAT: `egit.py --qat --baslangic <en_iyi.pth> --epok 10` (TI torchmodelopt).
