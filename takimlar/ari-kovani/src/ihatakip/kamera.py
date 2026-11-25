"""Görüntü kaynakları ve çıkış akışı.

Kaynak biçimleri:
    imx477[:0|1]        CSI kamera (Raspberry Pi HQ Camera; imx219, imx230, ov5647 de olur). Donanım ISP'si
                        (VPAC VISS) + ölçekleyici (MSC) + renk dönüşümü GStreamer'da. Aygıt adları
                        'camera-isp-sdk setup' (imx219 için gem-camera-setup) çıktısındakilerdir.
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


# Sensör -> (tiovxisp sensor-name, bayer biçimi, format-msb, kare hızı).
# imx219: TI'nin kendi yolu (object-tracker-o1 / edgeai-gst-apps). Diğerleri: T3 Kamera ISP SDK tanımlayıcıları
# (t3gemstone/camera-isp-sdk, sensors/<id>/sensor.conf); bu yol için kartta camera-isp-sdk paketi kurulu ve
# tiovx eklentisi yamalı olmalıdır.
SENSORLER = {
    "imx219": ("SENSOR_SONY_IMX219_RPI", "rggb", 7, 30),
    "imx477": ("imx477", "rggb12", 11, 50),   # Raspberry Pi HQ Camera, 1920x1080 RAW12
    "imx230": ("imx230", "rggb10", 9, 30),
    "ov5647": ("ov5647", "bggr10", 9, 30),
}


def sensor_boru_hatti(sensor: str = "imx477", kamera_no: int = 0, genislik: int = 1280, yukseklik: int = 720,
                      fps: int | None = None, isp_ek: str = "") -> str:
    """CSI kamera için donanım hattı: v4l2src (bayer) -> tiovxisp (VPAC VISS + 2A) -> tiovxmultiscaler (MSC)
    -> tiovxdlcolorconvert -> RGB. Görüntü işleme ve ölçekleme donanımda yapılır; A53 yalnız 2A'yı çalıştırır.

    isp_ek: tiovxisp'e eklenecek ayarlar, ör. "sink_0::awb-mode=0" (IMX477'de AE ve AWB birlikte açıkken görüntü
    kararıyor; bkz. camera-isp-sdk/sensors/imx477/README.md).
    """
    ad, bicim, msb, varsayilan_fps = SENSORLER[sensor]
    fps = fps or varsayilan_fps
    aygit = f"/dev/video-{sensor}-cam{kamera_no}"
    alt_aygit = f"/dev/v4l-{sensor}-subdev{kamera_no}"
    dcc = f"/opt/imaging/{sensor}/linear"
    return (
        f"v4l2src device={aygit} io-mode=dmabuf-import ! queue leaky=2 ! "
        f"video/x-bayer,width=1920,height=1080,format={bicim},framerate={fps}/1 ! "
        f"tiovxisp sensor-name={ad} dcc-isp-file={dcc}/dcc_viss.bin format-msb={msb} "
        f"sink_0::dcc-2a-file={dcc}/dcc_2a.bin sink_0::device={alt_aygit} {isp_ek} ! video/x-raw,format=NV12 ! "
        f"tiovxmultiscaler ! video/x-raw,format=NV12,width={genislik},height={yukseklik} ! "
        f"tiovxdlcolorconvert ! video/x-raw,format=RGB"
    )


def imx219_boru_hatti(kamera_no: int = 0, genislik: int = 1280, yukseklik: int = 720, fps: int = 30) -> str:
    return sensor_boru_hatti("imx219", kamera_no, genislik, yukseklik, fps)


def gamma_tablosu(gamma: float) -> np.ndarray:
    """Çıkış = 255 * (giriş/255)^(1/gamma). gamma > 1 karanlık görüntüyü aydınlatır."""
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.round(255.0 * x ** (1.0 / gamma)), 0, 255).astype(np.uint8)


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


class GammaKaynak:
    """Başka bir kaynağın karelerine gamma düzeltmesi uygular (cv2.LUT, 1280x720'de ~1-2 ms)."""

    def __init__(self, kaynak, gamma: float):
        self.kaynak = kaynak
        self.tablo = gamma_tablosu(gamma)
        self.dosya = getattr(kaynak, "dosya", False)
        self.fps = getattr(kaynak, "fps", None)

    def oku(self):
        ok, kare = self.kaynak.oku()
        if ok:
            kare = cv2.LUT(kare, self.tablo)
        return ok, kare

    def kapat(self):
        self.kaynak.kapat()


def kaynak_ac(kaynak: str, genislik: int = 1280, yukseklik: int = 720, fps: int | None = None,
              gamma: float = 1.0, isp_ek: str = ""):
    k = _kaynak_ac(kaynak, genislik, yukseklik, fps, isp_ek)
    return GammaKaynak(k, gamma) if gamma and abs(gamma - 1.0) > 1e-3 else k


def _kaynak_ac(kaynak: str, genislik: int, yukseklik: int, fps: int | None, isp_ek: str):
    sensor = kaynak.split(":")[0]
    if sensor in SENSORLER:
        no = int(kaynak.split(":")[1]) if ":" in kaynak else 0
        return GstKaynak(sensor_boru_hatti(sensor, no, genislik, yukseklik, fps, isp_ek))
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
