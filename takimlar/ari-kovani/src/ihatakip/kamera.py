"""Görüntü kaynakları ve çıkış akışı.

Kaynak biçimleri:
    imx219[:0|1]        CSI kamera, donanım ISP'si (VPAC VISS) + ölçekleyici (MSC) + renk dönüşümü GStreamer'da.
                        Kartta gem-camera-setup çıktısındaki aygıt adları kullanılır.
    /dev/videoN | N     V4L2 / USB kamera (OpenCV)
    gst:<boru hattı>    appsink ile biten herhangi bir GStreamer boru hattı (BGR ya da RGB çıkış)
    dosya.mp4 / .avi    video dosyası (OpenCV)

Çıkış biçimleri:
    udp://HOST:PORT     H.264 (donanım kodlayıcı v4l2h264enc) RTP/UDP akışı; bilgisayarda gst-launch ile izlenir
    dosya.mp4           video dosyası
    ekran               cv2.imshow (masaüstü imajı)
"""

from __future__ import annotations

import time

import numpy as np

import cv2

try:
    import gi

    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

    Gst.init(None)
except Exception:  # GStreamer Python bağları yoksa yalnız OpenCV kaynakları çalışır
    Gst = None


def imx219_boru_hatti(kamera_no: int = 0, genislik: int = 1280, yukseklik: int = 720, fps: int = 30) -> str:
    """IMX219 için TI donanım hattı: v4l2src (bayer) -> tiovxisp -> tiovxmultiscaler -> tiovxdlcolorconvert.

    Kaynak: t3gemstone/object-tracker-o1 (TI edgeai-gst-apps) içindeki boru hattı kurulumu.
    """
    aygit = f"/dev/video-imx219-cam{kamera_no}"
    alt_aygit = f"/dev/v4l-imx219-subdev{kamera_no}"
    dcc = "/opt/imaging/imx219/linear"
    return (
        f"v4l2src device={aygit} io-mode=5 ! queue leaky=2 ! "
        f"video/x-bayer,width=1920,height=1080,format=rggb,framerate={fps}/1 ! "
        f"tiovxisp sensor-name=SENSOR_SONY_IMX219_RPI dcc-isp-file={dcc}/dcc_viss.bin format-msb=7 "
        f"sink_0::dcc-2a-file={dcc}/dcc_2a.bin sink_0::device={alt_aygit} ! video/x-raw,format=NV12 ! "
        f"tiovxmultiscaler ! video/x-raw,format=NV12,width={genislik},height={yukseklik} ! "
        f"tiovxdlcolorconvert ! video/x-raw,format=RGB"
    )


class GstKaynak:
    def __init__(self, boru_hatti: str):
        if Gst is None:
            raise RuntimeError("GStreamer Python bağları (gi) yok")
        self.boru = Gst.parse_launch(boru_hatti + " ! appsink name=cikis max-buffers=1 drop=true sync=false")
        self.sink = self.boru.get_by_name("cikis")
        self.boru.set_state(Gst.State.PLAYING)

    def oku(self):
        ornek = self.sink.emit("try-pull-sample", 2 * Gst.SECOND)
        if ornek is None:
            return False, None
        yapi = ornek.get_caps().get_structure(0)
        w, h, bicim = yapi.get_value("width"), yapi.get_value("height"), yapi.get_value("format")
        tampon = ornek.get_buffer()
        ok, bilgi = tampon.map(Gst.MapFlags.READ)
        if not ok:
            return False, None
        try:
            dizi = np.frombuffer(bilgi.data, np.uint8)
            adim = dizi.size // h  # satır adımı dolgulu olabilir
            kare = dizi.reshape(h, adim)[:, : w * 3].reshape(h, w, 3)
            kare = kare[:, :, ::-1].copy() if bicim == "RGB" else kare.copy()
        finally:
            tampon.unmap(bilgi)
        return True, kare

    def kapat(self):
        self.boru.set_state(Gst.State.NULL)


class CvKaynak:
    def __init__(self, kaynak, genislik=None, yukseklik=None):
        self.cap = cv2.VideoCapture(kaynak)
        if genislik:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, genislik)
        if yukseklik:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, yukseklik)
        if not self.cap.isOpened():
            raise RuntimeError(f"Kaynak açılamadı: {kaynak}")
        self.dosya = isinstance(kaynak, str) and not kaynak.startswith("/dev/")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0

    def oku(self):
        return self.cap.read()

    def kapat(self):
        self.cap.release()


def kaynak_ac(kaynak: str, genislik: int = 1280, yukseklik: int = 720, fps: int = 30):
    if kaynak.startswith("imx219"):
        no = int(kaynak.split(":")[1]) if ":" in kaynak else 0
        return GstKaynak(imx219_boru_hatti(no, genislik, yukseklik, fps))
    if kaynak.startswith("gst:"):
        return GstKaynak(kaynak[4:])
    if kaynak.isdigit():
        return CvKaynak(int(kaynak), genislik, yukseklik)
    if kaynak.startswith("/dev/video"):
        return CvKaynak(kaynak, genislik, yukseklik)
    return CvKaynak(kaynak)


class Cikis:
    def __init__(self, hedef: str, genislik: int, yukseklik: int, fps: float = 30.0, kodlayici: str = "v4l2h264enc"):
        self.hedef = hedef
        self.tur = None
        if hedef == "ekran":
            self.tur = "ekran"
        elif hedef.startswith("udp://"):
            if Gst is None:
                raise RuntimeError("UDP akışı için GStreamer Python bağları (gi) gerekli")
            host, port = hedef[6:].rsplit(":", 1)
            boru = (
                f"appsrc name=giris is-live=true do-timestamp=true format=time "
                f"caps=video/x-raw,format=BGR,width={genislik},height={yukseklik},framerate={int(fps)}/1 ! "
                f"videoconvert ! video/x-raw,format=NV12 ! {kodlayici} ! h264parse ! "
                f"rtph264pay config-interval=1 pt=96 ! udpsink host={host} port={port} sync=false"
            )
            self.boru = Gst.parse_launch(boru)
            self.appsrc = self.boru.get_by_name("giris")
            self.boru.set_state(Gst.State.PLAYING)
            self.tur = "udp"
        else:
            dortcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.yazici = cv2.VideoWriter(hedef, dortcc, fps, (genislik, yukseklik))
            self.tur = "dosya"

    def yaz(self, kare: np.ndarray) -> None:
        if self.tur == "ekran":
            cv2.imshow("IHA takip", kare)
            cv2.waitKey(1)
        elif self.tur == "udp":
            tampon = Gst.Buffer.new_wrapped(kare.tobytes())
            self.appsrc.emit("push-buffer", tampon)
        else:
            self.yazici.write(kare)

    def kapat(self) -> None:
        if self.tur == "udp":
            self.appsrc.emit("end-of-stream")
            time.sleep(0.2)
            self.boru.set_state(Gst.State.NULL)
        elif self.tur == "dosya":
            self.yazici.release()
