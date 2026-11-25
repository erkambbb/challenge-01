#!/usr/bin/env python3
"""YOLOX-lite İHA algılayıcısını TI edgeai-mmdetection ile eğitir (ince ayar).

Her çalıştırma için eksiksiz bir mmdetection yapılandırması üretir, iş klasörüne kaydeder ve
TI'nin tools/train.py'sini çağırır. "lite" dönüşümü (SiLU -> ReLU, Focus -> konvolüsyon, büyük
maxpool -> art arda 3x3) TIDL ile tam uyum içindir. Başlangıç ağırlıkları TI model zoo'daki COCO
eğitimli YOLOX-lite ağırlıklarıdır; sınıflandırma başlığı tek sınıfa (iha) yeniden başlatılır.

Ortam değişkenleri (training/ortam_kur.sh bunları $IHA_CALISMA/ortam.env'e yazar):
    TI_MMDET     edgeai-mmdetection klasörü
    IHA_CALISMA  ağırlıkların, veri JSON'larının ve çıktıların durduğu çalışma klasörü

Örnek (RTX 4050 6 GB):
    python3 egit.py --ad nano960 --model nano --genislik 960 --yukseklik 544 \\
        --train $IHA_CALISMA/veri/coco/train.json --val $IHA_CALISMA/veri/coco/val.json \\
        --epok 60 --batch 16
    python3 egit.py ... --qat --baslangic <onceki_en_iyi.pth> --epok 10      # niceleme farkındalıklı ince ayar
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

AGIRLIKLAR = {
    "nano": "yolox_nano_lite_416x416_20220214_checkpoint.pth",
    "tiny": "yolox_tiny_lite_416x416_20220217_checkpoint.pth",
    "s": "yolox_s_lite_640x640_20220221_checkpoint.pth",
}

SABLON = '''# egit.py tarafından üretildi; elle değiştirmeyin, egit.py argümanlarını değiştirin.
_base_ = '{ti_mmdet}/configs_edgeailite/yolox/yolox_{model}_lite.py'

img_scale = ({w}, {h})  # (genişlik, yükseklik)
siniflar = ('iha',)
metainfo = dict(classes=siniflar)

model = dict(
    data_preprocessor=dict(batch_augments=[
        dict(type='BatchSyncRandomResize', random_size_range=({rmin}, {rmax}), size_divisor=32, interval=10)
    ]),
    bbox_head=dict(num_classes=1),
    test_cfg=dict(score_thr=0.01, nms=dict(type='nms', iou_threshold=0.45)))

train_pipeline = [
    dict(type='Mosaic', img_scale=img_scale, pad_val=114.0, prob={mosaic}),
    dict(type='RandomAffine', scaling_ratio_range=({olcek_min}, {olcek_max}), max_rotate_degree=5.0,
         max_shear_degree=1.0, border=(-img_scale[0] // 2, -img_scale[1] // 2)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='Pad', size=img_scale, pad_val=dict(img=(114.0, 114.0, 114.0))),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size={batch}, num_workers={isci}, persistent_workers=True,
    dataset=dict(
        type='MultiImageMixDataset',
        dataset=dict(
            type='CocoDataset', data_root='', metainfo=metainfo,
            ann_file='{train}', data_prefix=dict(img=''),
            pipeline=[dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True)],
            filter_cfg=dict(filter_empty_gt=False, min_size=32)),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size={val_batch}, num_workers={isci}, persistent_workers=True,
    dataset=dict(type='CocoDataset', data_root='', metainfo=metainfo, ann_file='{val}',
                 data_prefix=dict(img=''), test_mode=True, pipeline=test_pipeline))
test_dataloader = val_dataloader
val_evaluator = dict(type='CocoMetric', ann_file='{val}', metric='bbox', classwise=False)
test_evaluator = val_evaluator

max_epochs = {epok}
num_last_epochs = {son_epok}
base_lr = {lr}
train_cfg = dict(max_epochs=max_epochs, val_interval={val_aralik})
optim_wrapper = dict(
    type='{optim}',
    optimizer=dict(type='SGD', lr=base_lr, momentum=0.9, weight_decay=5e-4, nesterov=True),
    paramwise_cfg=dict(norm_decay_mult=0., bias_decay_mult=0.))
param_scheduler = [
    dict(type='mmdet.QuadraticWarmupLR', by_epoch=True, begin=0, end={isinma}, convert_to_iter_based=True),
    dict(type='CosineAnnealingLR', eta_min=base_lr * 0.05, begin={isinma}, T_max=max_epochs - num_last_epochs,
         end=max_epochs - num_last_epochs, by_epoch=True, convert_to_iter_based=True),
    dict(type='ConstantLR', by_epoch=True, factor=1, begin=max_epochs - num_last_epochs, end=max_epochs),
]
default_hooks = dict(
    checkpoint=dict(interval={val_aralik}, max_keep_ckpts=2, save_best='coco/bbox_mAP_50', rule='greater'),
    logger=dict(interval=50))
custom_hooks = [
    dict(type='YOLOXModeSwitchHook', num_last_epochs=num_last_epochs, priority=48),
    dict(type='SyncNormHook', priority=48),
    dict(type='EMAHook', ema_type='ExpMomentumEMA', momentum=0.0001, update_buffers=True, priority=49)
]
load_from = '{baslangic}'
randomness = dict(seed={tohum})
'''


def coco_sinifindan_baslat(kaynak: Path, hedef: Path, sinif: int) -> None:
    """COCO (80 sınıf) ağırlığındaki sınıflandırma başlığının yalnız bir sınıfını (ör. 4 = airplane)
    tutarak tek sınıflı başlangıç ağırlığı yazar. Böylece 'iha' sınıfı rastgele değil, COCO'nun uçak
    sınıfı bilgisiyle başlar."""
    import torch

    # Yalnız TI model zoo'nun resmi checkpoint'i ya da kendi eğitimimizin çıktısı yüklenir. mmengine
    # checkpoint'leri tensör dışı meta veri içerdiğinden weights_only=True ile açılamaz; mmengine de
    # eğitimde aynı dosyayı aynı biçimde yükler.
    ck = torch.load(str(kaynak), map_location="cpu")
    sd = ck.get("state_dict", ck)
    degisen = 0
    for k in list(sd):
        if "multi_level_conv_cls" in k and sd[k].shape[0] == 80:
            sd[k] = sd[k][sinif:sinif + 1].clone()
            degisen += 1
    torch.save(ck, str(hedef))
    print(f"başlangıç ağırlığı: COCO sınıf {sinif} -> iha ({degisen} tensör), {hedef}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ad", required=True, help="çalıştırma adı (iş klasörü: $IHA_CALISMA/egitim/<ad>)")
    ap.add_argument("--model", default="nano", choices=list(AGIRLIKLAR))
    ap.add_argument("--genislik", type=int, default=960)
    ap.add_argument("--yukseklik", type=int, default=544)
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--epok", type=int, default=60)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--val-batch", type=int, default=8)
    ap.add_argument("--isci", type=int, default=6, help="veri yükleyici işçi sayısı")
    ap.add_argument("--lr", type=float, default=None, help="varsayılan 0.01 * batch / 64")
    ap.add_argument("--mosaic", type=float, default=1.0)
    ap.add_argument("--olcek-min", type=float, default=0.5, help="RandomAffine en küçük ölçek")
    ap.add_argument("--olcek-max", type=float, default=1.5)
    ap.add_argument("--val-aralik", type=int, default=2)
    ap.add_argument("--baslangic", default=None, help="başlangıç ağırlığı (varsayılan: TI COCO lite)")
    ap.add_argument("--coco-sinifi", type=int, default=4,
                    help="tek sınıflı başlığı bu COCO sınıfından başlat (4 = airplane, -1 = rastgele)")
    ap.add_argument("--qat", action="store_true", help="niceleme farkındalıklı eğitim (TI torchmodelopt QAT)")
    ap.add_argument("--amp", action="store_true", default=True)
    ap.add_argument("--amp-kapali", dest="amp", action="store_false")
    ap.add_argument("--tohum", type=int, default=0)
    ap.add_argument("--devam", action="store_true", help="iş klasöründeki son noktadan devam et")
    ap.add_argument("--yalniz-ayar", action="store_true", help="yalnız yapılandırmayı yaz, eğitme")
    args = ap.parse_args()

    ti_mmdet = Path(os.environ["TI_MMDET"]).resolve()
    calisma = Path(os.environ["IHA_CALISMA"]).resolve()
    is_klasoru = calisma / "egitim" / args.ad
    is_klasoru.mkdir(parents=True, exist_ok=True)
    baslangic = args.baslangic or str(calisma / "modelzoo" / AGIRLIKLAR[args.model])
    if args.baslangic is None and args.coco_sinifi >= 0 and not args.yalniz_ayar:
        hedef = is_klasoru / f"baslangic_coco{args.coco_sinifi}.pth"
        coco_sinifindan_baslat(Path(baslangic), hedef, args.coco_sinifi)
        baslangic = str(hedef)
    lr =args.lr if args.lr is not None else 0.01 * args.batch / 64
    son_epok = max(2, min(15, args.epok // 5))
    if args.qat:
        son_epok = args.epok  # QAT'te mozaik kapalı, düz veriyle ince ayar
    # çok ölçekli eğitim aralığı yükseklik üzerinden (en-boy oranı korunur), 32'nin katı
    rmin = max(32, int(args.yukseklik * 0.75) // 32 * 32)
    rmax = int(args.yukseklik * 1.25) // 32 * 32
    ayar = SABLON.format(
        ti_mmdet=ti_mmdet, model=args.model, w=args.genislik, h=args.yukseklik, rmin=rmin, rmax=rmax,
        mosaic=args.mosaic, olcek_min=args.olcek_min, olcek_max=args.olcek_max, batch=args.batch,
        val_batch=args.val_batch, isci=args.isci, train=Path(args.train).resolve(), val=Path(args.val).resolve(),
        epok=args.epok, son_epok=son_epok, lr=lr, val_aralik=args.val_aralik,
        optim="AmpOptimWrapper" if args.amp and not args.qat else "OptimWrapper",
        isinma=min(3, max(1, args.epok // 10)) if not args.qat else 0, baslangic=baslangic, tohum=args.tohum)
    if args.qat:
        # QAT: düşük öğrenme oranı, ısınma yok
        ayar = ayar.replace(f"base_lr = {lr}", f"base_lr = {lr * 0.1}")
        ayar = ayar.replace("    dict(type='mmdet.QuadraticWarmupLR', by_epoch=True, begin=0, end=0, "
                            "convert_to_iter_based=True),\n", "")
    ayar_yolu = is_klasoru / "iha_ayar.py"
    ayar_yolu.write_text(ayar, encoding="utf-8")
    print(f"yapılandırma: {ayar_yolu}")
    if args.yalniz_ayar:
        return 0

    komut = [sys.executable, "tools/train.py", str(ayar_yolu), "--work-dir", str(is_klasoru)]
    if args.qat:
        komut += ["--quantization", "1"]
    if args.devam:
        komut += ["--resume"]
    print(" ".join(komut), flush=True)
    return subprocess.call(komut, cwd=ti_mmdet)


if __name__ == "__main__":
    sys.exit(main())
