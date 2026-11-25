# mmdeploy dışa aktarma ayarı (sabit giriş boyutu, ONNX Runtime).
# ONNX içindeki son işlem yalnızca CPU (float32) karşılaştırmasında kullanılır; TIDL'de bu kısım
# yerine TIDL'in algılama çıkış katmanı (meta-arch, prototxt) C7x'te çalışır. Eşikler TIDL
# prototxt'indekilerle aynı tutulur ki iki yolun sonuçları karşılaştırılabilsin.
onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=17,
    save_file='end2end.onnx',
    input_names=['input'],
    output_names=['dets', 'labels'],
    input_shape=None,
    optimize=True)
codebase_config = dict(
    type='mmdet',
    task='ObjectDetection',
    model_type='end2end',
    post_processing=dict(
        score_threshold=0.01,
        confidence_threshold=0.005,
        iou_threshold=0.45,
        max_output_boxes_per_class=200,
        pre_top_k=1000,
        keep_top_k=100,
        background_label_id=-1,
    ))
backend_config = dict(type='onnxruntime')
