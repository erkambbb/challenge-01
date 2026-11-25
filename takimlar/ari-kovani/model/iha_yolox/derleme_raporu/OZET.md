# Derleme raporu: iha_yolox_nano_lite_960x544_duman

- Giriş: 960x544, BGR, 0-255, NCHW (letterbox, dolgu 114)
- Derleyici: edgeai-tidl-tools 11_00_06_00, hedef AM67A, 8 bit, accuracy_level 1, kalibrasyon 50 kare x 10 yineleme
- 16 bit tutulan tensörler: 1, 165, 166, 164, 179, 180, 178, 192, 193, 191
- ONNX Runtime düğümleri: toplam 270, C7x/MMA (TIDL) 270, CPU 0, TIDL alt graf sayısı 1
- Toplam işlem: 4.4 GMAC

| TIDL katman türü | Adet |
|---|---|
| TIDL_ConvolutionLayer | 84 |
| TIDL_ConcatLayer | 16 |
| TIDL_EltWiseLayer | 7 |
| TIDL_PoolingLayer | 6 |
| TIDL_DataLayer | 3 |
| TIDL_ResizeLayer | 2 |
| TIDL_OdOutputReformatLayer | 2 |
| TIDL_DataConvertLayer | 1 |
| TIDL_DetectionOutputLayer | 1 |
