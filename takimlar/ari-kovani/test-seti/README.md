# test-seti/

Yarışmanın "Nesne tespiti" biçimi: `siniflar.txt` (tek sınıf: `iha`), `goruntuler/*.jpg`, `etiketler/*.txt` (YOLO: `sinif x_merkez y_merkez genislik yukseklik`, 0-1 arası; boş dosya = karede İHA yok).

**Şu anki içerik geçicidir.** `training/sentetik_ornek_uret.py --tohum 3` ile üretilmiş 60 prosedürel sentetik görüntüdür; yalnız değerlendirme zincirini (`olcum/calistir.py`) uçtan uca sınamak içindir. Teslimden önce yerini, eğitimde kullanılmamış gerçek görüntülerden oluşan ve eğitim verisinden çekim (video/oturum) düzeyinde ayrılmış test seti alacak.

- Eğitimde kullanılan veri bu klasöre konmaz.
- Görüntülerde kişisel veri (yüz, plaka vb.) bulunmaz.
