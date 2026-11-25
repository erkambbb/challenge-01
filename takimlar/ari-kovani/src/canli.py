#!/usr/bin/env python3
"""Canlı İHA tespit ve takip uygulaması (T3 Gemstone O1).

Kamera karesi -> (ISP/ölçekleme donanımda) -> YOLOX-lite INT8 (C7x/MMA, TIDL) -> takipçi (A53) -> çizim ->
H.264 donanım kodlayıcı -> RTP/UDP (ya da dosya / ekran).

Örnekler (kartta, root ve 'source /opt/t3-edgeai-env' sonrası):
    python3 src/canli.py --kaynak imx219:0 --cikis udp://192.168.7.1:5000
    python3 src/canli.py --kaynak /dev/video2 --cikis kayit.mp4
    python3 src/canli.py --kaynak ucus.mp4 --cikis ekran --mod cpu      # bilgisayarda deneme

Bilgisayarda akışı izlemek için:
    gst-launch-1.0 udpsrc port=5000 caps="application/x-rtp,media=video,encoding-name=H264,payload=96,clock-rate=90000" \
        ! rtpjitterbuffer latency=0 ! rtph264depay ! avdec_h264 ! autovideosink sync=false

Her saniye terminale FPS, gecikme ve izlerin kare merkezine göre normalleştirilmiş konumu (dx, dy: -1..1) yazılır.
--iz-kaydi verilirse her karenin izleri JSON Lines olarak kaydedilir.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ihatakip.algilayici import Algilayici  # noqa: E402
from ihatakip.cizim import durum_ciz, izleri_ciz, tespitleri_ciz  # noqa: E402
from ihatakip.kamera import Cikis, kaynak_ac  # noqa: E402
from ihatakip.takipci import Takipci  # noqa: E402

VARSAYILAN_MODEL = Path(__file__).resolve().parent.parent / "model" / "iha_yolox"


class SonKareOkuyucu(threading.Thread):
    """Kaynaktan sürekli okuyup yalnız en son kareyi tutar; çıkarım ile yakalamayı üst üste bindirir."""

    def __init__(self, kaynak, dosya_hizi: float | None = None):
        super().__init__(daemon=True)
        self.kaynak = kaynak
        self.dosya_hizi = dosya_hizi  # video dosyasında gerçek zamanı taklit etmek için
        self.kilit = threading.Lock()
        self.kare = None
        self.sayac = 0
        self.bitti = False
        self.dur = False

    def run(self):
        aralik = 1.0 / self.dosya_hizi if self.dosya_hizi else 0.0
        while not self.dur:
            t = time.perf_counter()
            ok, kare = self.kaynak.oku()
            if not ok:
                self.bitti = True
                break
            with self.kilit:
                self.kare = kare
                self.sayac += 1
            if aralik:
                kalan = aralik - (time.perf_counter() - t)
                if kalan > 0:
                    time.sleep(kalan)

    def al(self):
        with self.kilit:
            return self.sayac, self.kare


class SiraliOkuyucu:
    """Video dosyasını kare atlamadan okur (çevrimdışı işleme)."""

    def __init__(self, kaynak):
        self.kaynak = kaynak
        self.sayac = 0
        self.bitti = False
        self.dur = False

    def al(self):
        ok, kare = self.kaynak.oku()
        if not ok:
            self.bitti = True
            return self.sayac, None
        self.sayac += 1
        return self.sayac, kare


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kaynak", default="imx219:0")
    ap.add_argument("--model", default=str(VARSAYILAN_MODEL))
    ap.add_argument("--mod", default="otomatik", choices=["otomatik", "tidl", "cpu"])
    ap.add_argument("--cikis", default="", help="udp://HOST:PORT | dosya.mp4 | ekran | boş (çıkış yok)")
    ap.add_argument("--genislik", type=int, default=1280)
    ap.add_argument("--yukseklik", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--esik", type=float, default=None, help="izlenecek en düşük tespit skoru")
    ap.add_argument("--sure", type=float, default=0, help="saniye; 0 = Ctrl+C'ye kadar")
    ap.add_argument("--iz-kaydi", default="", help="izleri JSON Lines olarak yaz")
    ap.add_argument("--gercek-zaman", action="store_true",
                    help="video dosyasını kendi FPS'iyle oynat (geç kalan kareler atlanır)")
    ap.add_argument("--ham-tespit", action="store_true", help="eşik üstü ham tespitleri de çiz")
    args = ap.parse_args()

    meta = json.loads((Path(args.model) / "model.json").read_text(encoding="utf-8"))
    esik = args.esik if args.esik is not None else float(meta.get("canli_esik", 0.3))
    algilayici = Algilayici(args.model, mod=args.mod, esik=min(0.1, esik))
    takipci = Takipci(yuksek_esik=esik, dusuk_esik=min(0.1, esik), yeni_iz_esigi=max(esik, 0.4))
    kaynak = kaynak_ac(args.kaynak, args.genislik, args.yukseklik, args.fps)
    dosya = getattr(kaynak, "dosya", False)
    if dosya and not args.gercek_zaman:
        okuyucu = SiraliOkuyucu(kaynak)  # video dosyası: her kare sırayla işlenir
    else:
        okuyucu = SonKareOkuyucu(kaynak, getattr(kaynak, "fps", None) if dosya else None)
        okuyucu.start()

    cikis = None
    iz_dosyasi = open(args.iz_kaydi, "w", encoding="utf-8") if args.iz_kaydi else None
    calisiyor = [True]
    signal.signal(signal.SIGINT, lambda *_: calisiyor.__setitem__(0, False))
    signal.signal(signal.SIGTERM, lambda *_: calisiyor.__setitem__(0, False))

    print(f"[canli] model={algilayici.bilgi.ad} mod={algilayici.mod} "
          f"giris={algilayici.bilgi.genislik}x{algilayici.bilgi.yukseklik} esik={esik}", flush=True)
    son_no, islenen, baslangic = 0, 0, time.time()
    gecikmeler, kareler = deque(maxlen=60), deque(maxlen=60)
    son_rapor = time.time()
    while calisiyor[0]:
        no, kare = okuyucu.al()
        if kare is None or no == son_no:
            if okuyucu.bitti:
                break
            time.sleep(0.001)
            continue
        son_no = no
        t0 = time.perf_counter()
        tespitler = algilayici.tespit_et(kare)
        izler = takipci.guncelle(tespitler)
        t1 = time.perf_counter()
        islenen += 1
        gecikmeler.append((t1 - t0) * 1e3)
        kareler.append(t1)
        fps = (len(kareler) - 1) / (kareler[-1] - kareler[0]) if len(kareler) > 1 else 0.0
        c7x = algilayici.ti_olcumleri().get("c7x_islem_ms")

        h, w = kare.shape[:2]
        if args.cikis:
            if args.ham_tespit:
                tespitleri_ciz(kare, tespitler, esik)
            izleri_ciz(kare, izler, takipci.tahmini_izler())
            durum = [f"T3 Gemstone O1 | {algilayici.mod.upper()} | {fps:4.1f} FPS | {gecikmeler[-1]:5.1f} ms"
                     + (f" | C7x {c7x:4.1f} ms" if c7x else ""),
                     f"IHA: {len(izler)}"]
            durum_ciz(kare, durum)
            if cikis is None:
                cikis = Cikis(args.cikis, w, h, fps=args.fps)
            cikis.yaz(kare)

        if iz_dosyasi:
            iz_dosyasi.write(json.dumps({
                "kare": no, "zaman": round(time.time() - baslangic, 3),
                "izler": [{"id": iz.kimlik, "kutu": [round(float(v), 1) for v in iz.kutu],
                           "skor": round(iz.skor, 3)} for iz in izler]}) + "\n")

        if time.time() - son_rapor >= 1.0:
            son_rapor = time.time()
            konumlar = " ".join(
                f"#{iz.kimlik}(dx={((iz.kutu[0] + iz.kutu[2]) / 2 - w / 2) / (w / 2):+.2f},"
                f"dy={((iz.kutu[1] + iz.kutu[3]) / 2 - h / 2) / (h / 2):+.2f})" for iz in izler)
            ort = sum(gecikmeler) / len(gecikmeler)
            print(f"[canli] {fps:5.1f} FPS | gecikme {ort:5.1f} ms" + (f" | C7x {c7x:4.1f} ms" if c7x else "")
                  + f" | iz {len(izler)} {konumlar}", flush=True)
        if args.sure and time.time() - baslangic > args.sure:
            break

    okuyucu.dur = True
    if cikis:
        cikis.kapat()
    kaynak.kapat()
    if iz_dosyasi:
        iz_dosyasi.close()
    sure = time.time() - baslangic
    print(f"[canli] bitti: {islenen} kare, {sure:.1f} s, ortalama {islenen / max(sure, 1e-6):.1f} FPS", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
