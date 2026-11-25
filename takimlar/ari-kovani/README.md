# Havadan Havaya İHA Tespiti ve Takibi

T3 Gemstone O1'e bağlı kamerayla, havadaki başka İHA'ları gerçek zamanlı tespit edip izleyen; ağın tamamını (algılama başlığı ve NMS dahil) C7x/MMA hızlandırıcısında INT8 olarak çalıştıran uç yapay zekâ sistemi.

> **Durum (geliştirme sürüyor):** Bu klasördeki kod, derleme ve değerlendirme zinciri çalışır durumda. `model/iha_yolox` ve `test-seti/` şu an **geçici** içeriktir: zinciri uçtan uca sınamak için prosedürel sentetik verilerle eğitilmiş bir duman testi modeli ve sentetik bir test setidir. Asıl model (X-Plane sentetik + gerçek görüntülerle ince ayar) ve gerçek görüntülerden oluşan test seti teslimden önce bunların yerini alacak; sonuç tabloları da o zaman doldurulacak.

## Problem ve kullanım senaryosu

İnsansız hava araçlarının sayısı arttıkça, aynı hava sahasındaki başka bir İHA'yı zamanında görüp ondan kaçınmak (algıla ve kaçın) temel bir güvenlik ihtiyacı hâline geliyor. Küçük İHA'lar ADS-B gibi işbirlikçi yayın sistemleri taşımadığından, onları görmenin pratik yolu kameradır. Bu proje, uçuş halindeki bir platformun kamerasından gelen görüntüde başka İHA'ları tespit eder, her birine kalıcı bir kimlik vererek izler ve konumunu (kare merkezine göre normalleştirilmiş `dx`, `dy`) sürekli üretir.

Neden kart üzerinde (uçta)?

- **Gecikme:** Yaklaşan bir hedef için karar süresi saniyelerle ölçülür; görüntüyü buluta gönderip yanıt beklemek kabul edilemez.
- **Bağlantı:** Uçuş sırasında güvenilir ve yüksek bant genişlikli bir bağlantı yoktur.
- **Güç ve ağırlık:** İHA'ya ayrı bir bilgisayar ya da harici hızlandırıcı taşımak yerine, 4 TOPS'luk C7x/MMA hızlandırıcısı olan tek bir kart yeterli olmalıdır.

Zorluk: Uzaktaki bir İHA karede yalnızca birkaç piksel kaplar (ör. 2 m kanat açıklıklı bir İHA 300 m'de, 62° yatay görüş açılı 1280 piksellik bir karede yaklaşık 8 piksel). Bu yüzden giriş çözünürlüğü, küçük hedef performansı ve takip, model seçiminin merkezindedir.

## Donanım düzeneği

- **Kart:** T3 Gemstone O1 (TI AM67A: 4× Cortex-A53, 2× C7x DSP + MMA = 4 TOPS INT8), aktif soğutucu.
- **Kamera:** MIPI CSI kamera (CSI0 konnektörü). Görüntü, kartın donanım ISP'sinden (VPAC VISS) ve ölçekleyicisinden (MSC) geçer; renk dönüşümü de donanımda yapılır. Kamera modeli ve bağlantı fotoğrafı kart testinden sonra eklenecek.
- **Çıkış:** İşlenmiş görüntü kartın donanım H.264 kodlayıcısıyla sıkıştırılıp RTP/UDP ile yer bilgisayarına aktarılır; izlerin konumu terminale ve isteğe göre JSON Lines dosyasına yazılır.

Veri akışı ve donanım ayrımı:

```
CSI kamera ──> VPAC ISP (VISS) ──> MSC ölçekleyici ──> renk dönüşümü ──> A53: letterbox (önişlem)
                                                                              │
          C7x/MMA (TIDL, INT8): YOLOX-lite omurga + boyun + başlık + kutu çözme + NMS
                                                                              │
          A53: takipçi (Kalman + iki aşamalı eşleştirme) ──> çizim ──> H.264 kodlayıcı (donanım) ──> RTP/UDP
```

## Kurulum ve çalıştırma

Referans ortam: T3 Gemstone OS Minimal v2026.1 (Ubuntu 22.04) + t3-gem-o1-edgeai 11.0.

Kurulum tamamlandıktan sonra uygulama internet bağlantısı olmadan çalışır. Model dosyaları (ONNX ve derlenmiş TIDL dosyaları) `model/` altında repodadır; ek indirme gerekmez.

```bash
# Kurulum (kartta, repo kökünde)
sudo apt update && sudo apt install t3-gem-o1-edgeai git
# /boot/uEnv.txt -> overlays= satırının sonuna: k3-am67a-t3-gem-o1-edgeai-apps.dtbo ve kamera overlay'i; sonra reboot
sudo su
source /opt/t3-edgeai-env
pip3 install -r takimlar/ari-kovani/requirements.txt

# Uygulamayı çalıştırma (canlı kamera, akış yer bilgisayarına)
cd takimlar/ari-kovani
python3 src/canli.py --kaynak imx219:0 --cikis udp://192.168.7.1:5000
# Yer bilgisayarında izlemek için:
#   gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" \
#     ! rtpjitterbuffer latency=0 ! rtph264depay ! avdec_h264 ! autovideosink sync=false

# test-seti/ ile değerlendirme (repo kök dizininden; degerlendir.sh'ı çalıştırır)
sudo python3 olcum/calistir.py --takim ari-kovani
```

Diğer araçlar:

- `python3 src/hiz_olc.py --modeller model/iha_yolox` derlenmiş modellerin kart üzerindeki gecikmesini ölçer (`--cpu` ile aynı modelin float32 CPU süresi).
- `python3 src/canli.py --kaynak video.mp4 --cikis kayit.mp4 --iz-kaydi izler.jsonl` bir video dosyasını işler.

## Model

- **Mimari:** YOLOX-lite (TI edgeai-mmdetection). "lite" sürüm, TIDL ile tam uyum için SiLU yerine ReLU, Focus yerine konvolüsyon ve büyük maxpool yerine art arda 3×3 maxpool kullanır. Model boyutu (nano / tiny / s) ve giriş çözünürlüğü kartta yapılan hız ölçümleriyle seçilecek (ölçüm aracı: `src/hiz_olc.py`; aday varyantlar `training/varyant_derle.sh`). Geçici duman testi modeli: YOLOX-nano-lite, 960×544.
- **Model biçimi:** ONNX (opset 17) + AM67A için derlenmiş TIDL dosyaları (`model/iha_yolox/artifacts/`).
- **Model dosyaları:** `model/iha_yolox/` (model.onnx, model.prototxt, artifacts/, model.json, derleme_raporu/).
- **Görev türü:** Nesne tespiti (tek sınıf: `iha`) + çoklu hedef takibi.
- **Başarım metriği ve gerekçesi:** Birincil metrik **mAP@0.5** (tek sınıfta AP@0.5). Algıla-ve-kaçın görevinde önemli olan, hedefin karede *bulunup bulunmadığı* ve yaklaşık yeridir; birkaç piksellik uzak hedeflerde etiket kutusunun kendisi ±1-2 piksel belirsiz olduğundan, sıkı IoU eşikleri (0.75 ve üstü) modelin değil etiketin hatasını ölçer. AP@0.5, tüm güven eşikleri üzerinden hassasiyet-duyarlılık dengesini tek sayıda topladığı için eşik seçiminden bağımsızdır. Hesap pycocotools kurallarıyla birebir aynıdır (`training/metrik_dogrula.py`, en büyük fark 1e-16 düzeyinde). Ek metrikler `sonuc.json`'da raporlanır: AP@[.5:.95], AP@0.75, çalışma eşiğinde hassasiyet/duyarlılık/F1, **hedef boyutuna göre duyarlılık** (0-8, 8-16, 16-32, 32-64, 64+ piksel; uzak hedef performansı), **boş karede yanlış alarm sayısı** (gökyüzü ve bulutta hayalet tespit), uçtan uca gecikme ve C7x işlem süresi.
- **Eğitim verisi:** Yaklaşık 50.000 görüntü. 22.000'i X-Plane uçuş simülatöründe takım tarafından üretilen sentetik görüntülerdir: 165 farklı İHA modeli + 2 ve 3-4 İHA'lı gruplar + 2.000 boş arka plan karesi; hava (açık %40, ince bulut %10, bulutlu / yağmur / fırtına / sis %5'er, karışık %30), mesafe (0,25-25 m %10, 25-180 m %50, 180-300 m %40), 17 şehir, bina yoğunluğu, irtifa 150-1000 ft ve kamera eğimi kontrollü dağılımlarla örneklenmiştir; İHA'ların %15'i kare kenarında kesiktir. Kalan görüntülerin kaynağı ve lisansı bu bölüme eklenecek. Etiketleme biçimi YOLO (sınıf, merkez x, merkez y, genişlik, yükseklik; 0-1 arası). Veri seti repoda paylaşılmamaktadır.
- **Test verisi:** `test-seti/` yarışmanın "Nesne tespiti" biçimindedir (`siniflar.txt`, `goruntuler/`, `etiketler/`). Asıl test seti, eğitimde hiç kullanılmamış **gerçek** görüntülerden oluşacak ve eğitim/doğrulama verisinden çekim (video/oturum) düzeyinde ayrılacaktır (aynı videodan kareler hem eğitimde hem testte bulunmaz). Şu anki içerik geçicidir: `training/sentetik_ornek_uret.py` ile üretilmiş 60 sentetik görüntü.
- **Eğitim süreci:** COCO'da eğitilmiş TI YOLOX-lite ağırlıklarından ince ayar. Tek sınıflı sınıflandırma başlığı COCO'nun "airplane" sınıfının ağırlıklarından başlatılır. Mozaik, rastgele afin (ölçek 0,5-1,5), HSV ve yatay çevirme artırmaları, çok ölçekli eğitim (yüksekliğin ±%25'i), son epoklarda artırmasız eğitim, EMA, AMP. Ortam: Ubuntu tabanlı PC, NVIDIA RTX 4050 (6 GB), PyTorch 2.1.2 + CUDA 12.1, mmcv 2.1.0, edgeai-mmdetection 3.3. İsteğe bağlı ikinci aşama: TI torchmodelopt ile niceleme farkındalıklı eğitim (QAT); eğitim tarafı çalışıyor, TIDL'e aktarımı henüz denenmedi. Betikler: `training/egit.py`, `training/veri_hazirla.py`, ayrıntılar `training/README.md`.
- **Hızlandırıcıya derleme:** edgeai-tidl-tools **11_00_06_00** (kartın TIDL 11.0 çalışma zamanıyla uyumlu sürüm), hedef AM67A. ONNX dışa aktarımı TI torch2onnx ile; TIDL "meta-arch" (YOLOX) prototxt'i sayesinde kutu çözme ve NMS de C7x'te çalışır. INT8 niceleme (asimetrik, kanal başına; TI'nin AM67A önerisi), eğitim verisinden seçilen 50 kalibrasyon karesiyle; ilk konvolüsyon ve algılama başlığı çıkışları 16 bit (karışık hassasiyet). Duman testinde bu ayarlarla INT8 modelin AP@0.5'i float32 ile aynı çıktı (fark 0,0001; saf INT8'de AP@[.5:.95] 0,64'ten 0,47'ye düşüyordu); deney tablosu `training/README.md`'de. Kalibrasyon önişlemi kartta kullanılan kodla aynıdır (`src/ihatakip/onislem.py`). Betik: `training/tidl_derle.py`.
- **Hızlandırıcıda / CPU'da çalışan kısım:** ONNX Runtime grafındaki düğümlerin **tamamı** (duman testi modelinde 270/270) tek bir TIDL alt grafı olarak C7x/MMA'da çalışır: uint8 girişin float'a dönüşümü (`TIDL_DataConvertLayer`), omurga, boyun, başlık konvolüsyonları, yeniden boyutlandırma, birleştirme, havuzlama, kutu çözme ve NMS (`TIDL_DetectionOutputLayer`). Model girişi uint8 olduğundan A53 her karede float32 tensör üretmez; kareyi olduğu gibi (NCHW, uint8) verir. Ayrıntılı katman listesi ve MAC sayısı: `model/iha_yolox/derleme_raporu/OZET.md`. CPU'da (A53) kalanlar: görüntünün letterbox ile model boyutuna getirilmesi, tespitlerin özgün koordinata çevrilmesi, takipçi ve çizim. Kamera ISP'si, ölçekleme ve H.264 kodlama da kartın ayrı donanım bloklarındadır.
- **Hazır model kullanıldıysa:** Başlangıç ağırlıkları TI edgeai-modelzoo'daki COCO eğitimli YOLOX-lite ağırlıklarıdır (BSD-3-Clause). Bu ağırlıklar İHA verisiyle yeniden eğitilmiştir; hazır model olduğu gibi kullanılmamaktadır.

## Sonuçlar

Asıl model ve gerçek test setiyle kartta (`olcum/calistir.py`) elde edilecek sonuçlar teslimden önce bu tabloya yazılacak.

| | Değer |
|---|---|
| mAP@0.5 | kart testinden sonra |
| FPS | kart testinden sonra |
| Gecikme ortalama / p95 (ms) | kart testinden sonra |

Planlanan karşılaştırmalar:

- Aynı modelin CPU'da float32 ile C7x'te INT8 çalışması: doğruluk farkı ve hız kazancı (`CPU_KARSILASTIRMA` ile `sonuc.json`'a otomatik yazılır).
- Model boyutu ve giriş çözünürlüğüne göre gecikme ve küçük hedef duyarlılığı (model seçiminin gerekçesi).
- Tek C7x ile iki C7x'in birlikte çalıştığı düşük gecikme modu (`inference_mode=2`).

## Demo videosu

Bağlantı teslimden önce eklenecek. Çekim planı: kart, kamera ve yer bilgisayarındaki canlı akış aynı karede, kesintisiz; test setinde ve eğitimde bulunmayan yeni bir ortamda uçan gerçek bir İHA'nın tespiti ve takibi.

## Bu yarışmadaki katkı

Proje bu yarışma için geliştirilmiştir.

## Kullanılan yapay zekâ araçları

Claude Code (Anthropic): kod yazımı, derleme ve eğitim betiklerinin hazırlanması, belgelerin taslağı ve hata ayıklamada yardımcı olarak kullanıldı. Tüm çıktılar takım tarafından çalıştırılıp denetlenmiştir.

## Takım ve lisans

- **Takım:** Arı Kovanı
- **Takım Kaptanı:** erkambbb
- **Üyeler:** erkambbb (diğer üyelerin GitHub kullanıcı adları eklenecek)
- **Danışman:** belirtilecek
- **Lisans:** Bu repodaki çalışmalar referans repodaki lisans koşullarına tabidir; [LICENSE](LICENSE) dosyasına bakın.
- **Üçüncü taraf model, veri seti ve kütüphaneler:**
  - YOLOX-lite COCO başlangıç ağırlıkları ve yapılandırmaları: TI edgeai-modelzoo / edgeai-mmdetection, BSD-3-Clause (YOLOX: Megvii, Apache-2.0; MMDetection: OpenMMLab, Apache-2.0).
  - edgeai-modeloptimization (torchmodelopt) ve edgeai-mmdeploy: Texas Instruments, BSD-3-Clause. Yalnız eğitim/dışa aktarma aracı olarak kullanılır, repoya kopyalanmamıştır.
  - edgeai-tidl-tools 11_00_06_00 (TIDL derleyicisi ve ONNX Runtime TIDL sağlayıcısı): Texas Instruments, TI Text File License. Derlenmiş TIDL dosyaları bu araçla üretilmiştir.
  - ONNX Runtime (MIT), OpenCV (Apache-2.0), NumPy (BSD-3-Clause), GStreamer (LGPL-2.1, kartın sistem paketi).
  - IMX219 donanım boru hattı ayarları: t3gemstone/object-tracker-o1 ve TI edgeai-gst-apps (TI Text File License) referans alınarak yazılmıştır.
  - Takip yöntemi ByteTrack'in iki aşamalı eşleştirme fikrine dayanır (Zhang ve ark., "ByteTrack: Multi-Object Tracking by Associating Every Detection Box", ECCV 2022); kod takım tarafından yazılmıştır.
  - COCO veri seti (TI'nin ön eğitiminde kullanıldı): etiketler CC BY 4.0.
  - Eğitim verisi: X-Plane (Laminar Research) ile takım tarafından üretilen sentetik görüntüler; diğer kaynaklar ve lisansları eklenecek.
