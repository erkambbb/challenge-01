# src/

Kart üzerinde çalışan kod. Bağımlılıklar referans ortamda hazır gelir: Python 3.10, NumPy, OpenCV, ONNX Runtime (TIDL sağlayıcısıyla), GStreamer + PyGObject. TIDL hızlandırıcısı root yetkisi ister: `sudo su` ve `source /opt/t3-edgeai-env`.

| Dosya | Görev |
|---|---|
| `degerlendir.py` | `test-seti/` üzerinde mAP@0.5 ve ek metrikleri hesaplar, `sonuc.json` yazar (`degerlendir.sh` çağırır) |
| `canli.py` | Canlı kamera / video: tespit + takip + çizim + H.264 RTP/UDP akışı ya da dosyaya kayıt |
| `hiz_olc.py` | Derlenmiş model klasörlerinin gecikmesini ölçer (model ve çözünürlük seçimi için) |
| `ihatakip/onislem.py` | Letterbox önişlemi (eğitim ve TIDL kalibrasyonuyla aynı) |
| `ihatakip/algilayici.py` | ONNX Runtime oturumu: TIDL (C7x/MMA, INT8) ya da CPU (float32) |
| `ihatakip/takipci.py` | Kalman filtreli, iki aşamalı (yüksek / düşük skor) çoklu hedef takipçisi |
| `ihatakip/metrikler.py` | AP@0.5, AP@[.5:.95], hassasiyet / duyarlılık / F1, hedef boyutuna göre duyarlılık (pycocotools ile aynı sonuç) |
| `ihatakip/veri.py` | Yarışmanın nesne tespiti test seti biçimini (YOLO etiketleri) okur |
| `ihatakip/kamera.py` | Kaynaklar (IMX219 donanım ISP hattı, V4L2/USB, video, GStreamer) ve çıkış (UDP, dosya, ekran) |
| `ihatakip/cizim.py` | Kutu, iz, kimlik ve durum çizimi |

`--mod cpu` ile aynı kod TIDL olmadan bilgisayarda da çalışır. x86 bilgisayarda `TIDL_TOOLS_PATH` tanımlıysa `--mod tidl` aynı derlenmiş dosyalarla TIDL emülasyonu yapar.
