#!/usr/bin/env python3
"""src/ihatakip/metrikler.py'nin AP hesabını pycocotools ile karşılaştırır.

Rastgele üretilmiş gerçek kutular ve gürültülü tespitlerle (boş görüntüler, kaçırılan hedefler, yanlış
alarmlar dahil) birçok senaryo dener; AP@0.5 ve AP@[.5:.95] farkının 1e-6'dan küçük olmasını bekler.
Kartta değil, bilgisayarda çalışır (pycocotools gerekir).

    python3 metrik_dogrula.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from ihatakip.metrikler import IOU_ESIKLERI, TespitDegerlendirici  # noqa: E402


def senaryo(rng, n_goruntu=60):
    goruntuler = []
    for _ in range(n_goruntu):
        n = rng.choice([0, 0, 1, 1, 2, 3, 5])
        gercek = []
        for _ in range(n):
            w, h = rng.uniform(3, 120), rng.uniform(3, 80)
            x, y = rng.uniform(0, 900), rng.uniform(0, 500)
            gercek.append([x, y, x + w, y + h, 0])
        gercek = np.asarray(gercek, np.float64).reshape(-1, 5)
        tespit = []
        for g in gercek:
            if rng.random() < 0.8:  # bulunan
                j = rng.normal(0, 0.15 * (g[2] - g[0]), 4)
                tespit.append([g[0] + j[0], g[1] + j[1], g[2] + j[2], g[3] + j[3], rng.uniform(0.2, 1.0), 0])
            if rng.random() < 0.2:  # çift tespit
                tespit.append([*g[:4] + rng.normal(0, 3, 4), rng.uniform(0.01, 0.6), 0])
        for _ in range(rng.integers(0, 4)):  # yanlış alarm
            x, y = rng.uniform(0, 900), rng.uniform(0, 500)
            tespit.append([x, y, x + rng.uniform(3, 60), y + rng.uniform(3, 60), rng.uniform(0.01, 0.7), 0])
        tespit = np.asarray(tespit, np.float64).reshape(-1, 6)
        tespit[:, [0, 2]] = np.sort(tespit[:, [0, 2]], axis=1)
        tespit[:, [1, 3]] = np.sort(tespit[:, [1, 3]], axis=1)
        goruntuler.append((tespit, gercek))
    return goruntuler


def pycoco(goruntuler):
    import contextlib
    import io

    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    images, anns, dets = [], [], []
    aid = 1
    for i, (t, g) in enumerate(goruntuler, 1):
        images.append({"id": i, "width": 1000, "height": 600})
        for b in g:
            anns.append({"id": aid, "image_id": i, "category_id": 1, "bbox": [b[0], b[1], b[2] - b[0], b[3] - b[1]],
                         "area": (b[2] - b[0]) * (b[3] - b[1]), "iscrowd": 0})
            aid += 1
        for d in t:
            dets.append({"image_id": i, "category_id": 1, "bbox": [d[0], d[1], d[2] - d[0], d[3] - d[1]],
                         "score": float(d[4])})
    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": [{"id": 1, "name": "iha"}]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(dets) if dets else COCO()
        e = COCOeval(gt, dt, "bbox")
        e.params.areaRng = [[0, 1e10]]
        e.params.areaRngLbl = ["all"]
        e.evaluate()
        e.accumulate()
    p = e.eval["precision"][:, :, 0, 0, -1]  # [T, R]
    ap = [float(np.mean(p[t][p[t] > -1])) for t in range(len(IOU_ESIKLERI))]
    return ap[0], float(np.mean(ap))


def main():
    rng = np.random.default_rng(0)
    en_buyuk_fark = 0.0
    for s in range(20):
        goruntuler = senaryo(rng)
        d = TespitDegerlendirici()
        for t, g in goruntuler:
            d.ekle(t, g)
        biz50, biz_aralik = d.ap(0.5), float(np.mean([d.ap(float(e)) for e in IOU_ESIKLERI]))
        coco50, coco_aralik = pycoco(goruntuler)
        fark = max(abs(biz50 - coco50), abs(biz_aralik - coco_aralik))
        en_buyuk_fark = max(en_buyuk_fark, fark)
        print(f"senaryo {s:2d}: AP50 biz={biz50:.6f} coco={coco50:.6f} | AP[.5:.95] biz={biz_aralik:.6f} "
              f"coco={coco_aralik:.6f} | fark={fark:.2e}")
    print(f"en büyük fark: {en_buyuk_fark:.2e}")
    return 0 if en_buyuk_fark < 1e-6 else 1


if __name__ == "__main__":
    sys.exit(main())
