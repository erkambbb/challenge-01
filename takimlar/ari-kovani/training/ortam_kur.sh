#!/usr/bin/env bash
# Eğitim ve TIDL derleme ortamını kurar (x86_64 Linux, NVIDIA GPU; Docker gerekmez).
# Ubuntu 22.04/24.04 ve türevlerinde denendi (Linux Mint 22.3). uv (https://docs.astral.sh/uv/) ile
# Python 3.10 sanal ortamları açar; sistem Python'una dokunmaz.
#
#   IHA_CALISMA=~/iha_calisma bash ortam_kur.sh
#
# Kurulanlar:
#   $IHA_CALISMA/venv-egitim   PyTorch 2.1.2+cu121, mmcv 2.1.0, mmengine 0.10.7, TI edgeai-mmdetection 3.3,
#                              TI edgeai-modeloptimization (torchmodelopt), TI edgeai-mmdeploy
#   $IHA_CALISMA/venv-tidl     onnxruntime-tidl 1.15.0 (edgeai-tidl-tools 11_00_06_00), onnx 1.14.0
#   $IHA_CALISMA/edgeai-tidl-tools/tools/AM67A/tidl_tools   TIDL derleyicisi (AM67A, 11_00_06_00)
#   $IHA_CALISMA/ti/                                         TI eğitim depoları (t3gemstone/edgeai-tensorlab)
#   $IHA_CALISMA/modelzoo/                                   TI YOLOX-lite COCO ağırlıkları, ONNX, prototxt
#   $IHA_CALISMA/ortam.env                                   ortam değişkenleri (source ile yükleyin)
set -euo pipefail

IHA_CALISMA="${IHA_CALISMA:-$HOME/iha_calisma}"
TIDL_REL=11_00_06_00
mkdir -p "$IHA_CALISMA" && cd "$IHA_CALISMA"
command -v uv >/dev/null || { echo "uv gerekli: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# 1) TI eğitim depoları (T3 Gemstone'un edgeai-tensorlab fork'u)
if [ ! -d ti/edgeai-mmdetection ]; then
  git clone --depth 1 https://github.com/t3gemstone/edgeai-tensorlab.git ti-tensorlab
  mkdir -p ti
  mv ti-tensorlab/edgeai-mmdetection ti-tensorlab/edgeai-modeloptimization ti-tensorlab/edgeai-mmdeploy ti/
  rm -rf ti-tensorlab
fi

# 2) Eğitim ortamı
if [ ! -x venv-egitim/bin/python ]; then
  uv venv --python 3.10 venv-egitim
  P=venv-egitim/bin/python
  uv pip install --python $P torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu121
  uv pip install --python $P numpy==1.26.4 mmengine==0.10.7
  uv pip install --python $P mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.1/index.html
  uv pip install --python $P cython wheel setuptools==73.0.0 pycocotools terminaltables shapely scipy matplotlib \
    six tqdm onnx==1.14.0 protobuf==3.20.2 onnxsim torchinfo pydot progressbar colorama pyyaml \
    opencv-python-headless==4.8.1.78 aenum grpcio multiprocess prettytable
  (cd ti/edgeai-mmdetection && uv pip install --python ../../$P --no-build-isolation -e .)
  (cd ti/edgeai-modeloptimization/torchmodelopt && uv pip install --python ../../../$P --no-build-isolation -e .)
  (cd ti/edgeai-mmdeploy && uv pip install --python ../../$P --no-build-isolation --no-deps -e .)
fi

# 3) TIDL derleme ortamı (kartın TIDL 11.0 çalışma zamanıyla uyumlu sürüm)
if [ ! -d edgeai-tidl-tools/tools/AM67A/tidl_tools ]; then
  git clone --depth 1 --branch $TIDL_REL https://github.com/TexasInstruments/edgeai-tidl-tools.git
  mkdir -p edgeai-tidl-tools/tools/AM67A
  (cd edgeai-tidl-tools/tools/AM67A && \
   wget -q https://software-dl.ti.com/jacinto7/esd/tidl-tools/$TIDL_REL/TIDL_TOOLS/AM67A/tidl_tools.tar.gz && \
   tar -xzf tidl_tools.tar.gz && rm tidl_tools.tar.gz)
fi
if [ ! -x venv-tidl/bin/python ]; then
  uv venv --python 3.10 venv-tidl
  uv pip install --python venv-tidl/bin/python numpy==1.23.0 onnx==1.14.0 protobuf==3.20.3 pyyaml pillow \
    opencv-python-headless==4.8.1.78
  uv pip install --python venv-tidl/bin/python \
    https://software-dl.ti.com/jacinto7/esd/tidl-tools/$TIDL_REL/OSRT_TOOLS/X86_64_LINUX/UBUNTU_22_04/onnxruntime_tidl-1.15.0-cp310-cp310-linux_x86_64.whl
fi

# 4) TI model zoo: YOLOX-lite COCO ağırlıkları (başlangıç noktası)
mkdir -p modelzoo
B=https://software-dl.ti.com/jacinto7/esd/modelzoo/11_01_00/models/vision/detection/coco/edgeai-mmdet
for m in yolox_nano_lite_416x416_20220214 yolox_tiny_lite_416x416_20220217 yolox_s_lite_640x640_20220221; do
  for s in _checkpoint.pth _model.onnx _model.prototxt; do
    [ -f modelzoo/$m$s ] || wget -q -O modelzoo/$m$s $B/$m$s
  done
done

cat > ortam.env <<EOF
export IHA_CALISMA=$IHA_CALISMA
export TI_MMDET=\$IHA_CALISMA/ti/edgeai-mmdetection
export TIDL_TOOLS_PATH=\$IHA_CALISMA/edgeai-tidl-tools/tools/AM67A/tidl_tools
export IHA_EGITIM_PY=\$IHA_CALISMA/venv-egitim/bin/python
export IHA_TIDL_PY=\$IHA_CALISMA/venv-tidl/bin/python
EOF
venv-egitim/bin/python -c "import torch, mmcv, mmdet; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'mmdet', mmdet.__version__)"
venv-tidl/bin/python -c "import onnxruntime as o; print('onnxruntime-tidl', o.__version__, o.get_available_providers())"
echo "Hazır. Kullanım: source $IHA_CALISMA/ortam.env"
