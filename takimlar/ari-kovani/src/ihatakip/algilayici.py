"""ONNX Runtime ile İHA algılayıcı.

İki çalışma biçimi vardır:
    tidl  ONNX Runtime + TIDLExecutionProvider. Ağın tamamı (algılama başlığı ve NMS dahil, TIDL
          "meta-arch" katmanı sayesinde) C7x/MMA hızlandırıcısında INT8 olarak çalışır. Derlenmiş
          TIDL dosyaları model/<ad>/artifacts/ altındadır. Kartta çalışır; x86 bilgisayarda
          TIDL_TOOLS_PATH tanımlıysa aynı dosyalarla emülasyon yapılır.
    cpu   Aynı ONNX modeli yalnız CPUExecutionProvider ile float32 olarak çalıştırır (karşılaştırma için).

model/<ad>/model.json alanları:
    onnx, artifacts, giris{genislik, yukseklik, dolgu, ortalama, olcek}, siniflar[]
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import onnxruntime as ort

from .onislem import letterbox, tensore_cevir


@dataclass
class ModelBilgisi:
    dizin: Path
    onnx: Path
    artifacts: Path
    genislik: int
    yukseklik: int
    dolgu: int = 114
    ortalama: tuple = (0.0, 0.0, 0.0)
    olcek: tuple = (1.0, 1.0, 1.0)
    siniflar: list = field(default_factory=lambda: ["iha"])
    ad: str = ""
    cikarim_modu: int = 0

    @classmethod
    def oku(cls, dizin: str | Path) -> "ModelBilgisi":
        dizin = Path(dizin)
        meta = json.loads((dizin / "model.json").read_text(encoding="utf-8"))
        g = meta["giris"]
        return cls(
            dizin=dizin,
            onnx=dizin / meta.get("onnx", "model.onnx"),
            artifacts=dizin / meta.get("artifacts", "artifacts"),
            genislik=int(g["genislik"]),
            yukseklik=int(g["yukseklik"]),
            dolgu=int(g.get("dolgu", 114)),
            ortalama=tuple(g.get("ortalama", (0.0, 0.0, 0.0))),
            olcek=tuple(g.get("olcek", (1.0, 1.0, 1.0))),
            siniflar=list(meta.get("siniflar", ["iha"])),
            ad=meta.get("ad", dizin.name),
            cikarim_modu=int(meta.get("derleme", {}).get("inference_mode", 0)),
        )


def tidl_var_mi() -> bool:
    return "TIDLExecutionProvider" in ort.get_available_providers()


class Algilayici:
    """Tek karede İHA tespiti. tespit_et() özgün görüntü koordinatlarında (N, 6) dizi döndürür:
    x1, y1, x2, y2, skor, sinif."""

    def __init__(self, model_dizini: str | Path, mod: str = "otomatik", cekirdek: int = 1,
                 esik: float = 0.01, cpu_is_parcacigi: int = 0):
        self.bilgi = ModelBilgisi.oku(model_dizini)
        if mod == "otomatik":
            mod = "tidl" if tidl_var_mi() and self.bilgi.artifacts.is_dir() else "cpu"
        if mod == "tidl" and not tidl_var_mi():
            raise RuntimeError("TIDLExecutionProvider yok. Kartta 'source /opt/t3-edgeai-env' yapıldı mı, "
                               "root olarak mı çalışıyor?")
        if mod == "tidl" and not self.bilgi.artifacts.is_dir():
            raise RuntimeError(f"TIDL dosyaları bulunamadı: {self.bilgi.artifacts}")
        self.mod = mod
        self.esik = esik

        so = ort.SessionOptions()
        if cpu_is_parcacigi:
            so.intra_op_num_threads = cpu_is_parcacigi
        if mod == "tidl":
            secenekler = {"artifacts_folder": str(self.bilgi.artifacts), "debug_level": 0}
            if self.bilgi.cikarim_modu:
                # çok çekirdekli derleme (1: yüksek verim, 2: düşük gecikme): C7x_1'den başla
                secenekler["core_start_idx"] = int(cekirdek)
            else:
                secenekler["core_number"] = int(cekirdek)
            araclar = os.environ.get("TIDL_TOOLS_PATH")
            if araclar:  # yalnız x86 emülasyonunda gerekir; kartta yok sayılır
                secenekler["tidl_tools_path"] = araclar
            self.oturum = ort.InferenceSession(
                str(self.bilgi.onnx),
                providers=["TIDLExecutionProvider", "CPUExecutionProvider"],
                provider_options=[secenekler, {}],
                sess_options=so,
            )
        else:
            self.oturum = ort.InferenceSession(str(self.bilgi.onnx), providers=["CPUExecutionProvider"],
                                               sess_options=so)
        self.giris_adi = self.oturum.get_inputs()[0].name
        self.cikis_adlari = [o.name for o in self.oturum.get_outputs()]
        self.son_sureler = {}

    # ------------------------------------------------------------------
    def _cozumle(self, ciktilar: dict, r: float, h: int, w: int) -> np.ndarray:
        if "dets" in ciktilar:
            dets = np.asarray(ciktilar["dets"], np.float32).reshape(-1, 5)
            if "labels" in ciktilar:
                labels = np.asarray(ciktilar["labels"]).reshape(-1).astype(np.float32)
            else:
                labels = np.zeros(len(dets), np.float32)
        else:
            raise RuntimeError(f"Beklenmeyen model çıkışları: {list(ciktilar)}")
        tut = dets[:, 4] >= self.esik
        dets, labels = dets[tut], labels[tut]
        if len(dets) == 0:
            return np.zeros((0, 6), np.float32)
        kutular = dets[:, :4] / r
        kutular[:, [0, 2]] = kutular[:, [0, 2]].clip(0, w)
        kutular[:, [1, 3]] = kutular[:, [1, 3]].clip(0, h)
        sonuc = np.concatenate([kutular, dets[:, 4:5], labels[:, None]], axis=1)
        gecerli = (sonuc[:, 2] - sonuc[:, 0] > 0.5) & (sonuc[:, 3] - sonuc[:, 1] > 0.5)
        sonuc = sonuc[gecerli]
        return sonuc[np.argsort(-sonuc[:, 4])]

    def tespit_et(self, bgr: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        b = self.bilgi
        tuval, r = letterbox(bgr, b.genislik, b.yukseklik, b.dolgu)
        x = tensore_cevir(tuval, b.ortalama, b.olcek)
        t1 = time.perf_counter()
        cikti = self.oturum.run(None, {self.giris_adi: x})
        t2 = time.perf_counter()
        sonuc = self._cozumle(dict(zip(self.cikis_adlari, cikti)), r, bgr.shape[0], bgr.shape[1])
        t3 = time.perf_counter()
        self.son_sureler = {"onislem_ms": (t1 - t0) * 1e3, "cikarim_ms": (t2 - t1) * 1e3,
                            "sonislem_ms": (t3 - t2) * 1e3, "toplam_ms": (t3 - t0) * 1e3}
        return sonuc

    def ti_olcumleri(self) -> dict:
        """TIDL oturumunda son çalıştırmanın C7x üzerindeki süreleri (ms). CPU modunda boş döner."""
        if self.mod != "tidl" or not hasattr(self.oturum, "get_TI_benchmark_data"):
            return {}
        try:
            v = self.oturum.get_TI_benchmark_data()
        except Exception:
            return {}
        sonuc = {}
        alt_graflar = sorted({k.split("ts:subgraph_")[1].split("_proc_start")[0]
                              for k in v if "proc_start" in k})
        islem = sum(v[f"ts:subgraph_{s}_proc_end"] - v[f"ts:subgraph_{s}_proc_start"] for s in alt_graflar)
        sonuc["c7x_islem_ms"] = islem / 1e6
        if "ts:run_start" in v and "ts:run_end" in v:
            sonuc["oturum_toplam_ms"] = (v["ts:run_end"] - v["ts:run_start"]) / 1e6
        sonuc["alt_graf_sayisi"] = len(alt_graflar)
        return sonuc
