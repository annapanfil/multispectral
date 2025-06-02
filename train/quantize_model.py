import onnx
from onnxconverter_common import float16
from ultralytics import YOLO

def test_improvement(model_fp32, model_fp16):
    metrics_fp32 = model_fp32.val(data='/home/anna/Datasets/created/sea_form8_sea_aug-random/sea_form8_sea_aug-random.yaml', split='val')
    metrics_fp16 = model_fp16.val(data='/home/anna/Datasets/created/sea_form8_sea_aug-random/sea_form8_sea_aug-random.yaml', split='val')
        
    for k in metrics_fp16.results_dict.keys():
        print(f"FP32 {k}: {metrics_fp32.results_dict[k]}")
        print(f"FP16 {k}: {metrics_fp16.results_dict[k]}")

    print("FP32 time:", metrics_fp32.speed)
    print("FP16 time:", metrics_fp16.speed)

def time_model(model):
    from timeit import timeit
    import cv2

    img = cv2.imread('/home/anna/Datasets/created/sea_form8_sea_aug-random/images/train/mandrac-beach_11_0079_10_aug0_N_G_(N-(E-N)).png')
    # FP32
    model.float()
    t_fp32 = timeit(lambda: model.predict(img, half=False), number=50)

    # FP16 
    model.half()
    t_fp16 = timeit(lambda: model.predict(img, half=True), number=50)

    print(f"FP32: {t_fp32/50:.4f}s/img | FP16: {t_fp16/50:.4f}s/img")

if __name__ == "__main__":
    # pt to onnx conversion: from onnxruntime.quantization import quantize_dynamic, QuantType
    model = YOLO('../models/sea-form8_sea_aug-random_best.pt')
    # model.export(format='onnx', dynamic=True, simplify=True)
    quantized_model_path = "../models/sea-form8_sea_aug-random_best_int8.onnx"

    # model_onnx = onnx.load("../models/sea-form8_sea_aug-random_best.onnx")
    # model_fp16 = float16.convert_float_to_float16(model_onnx)
    # onnx.save(model_fp16, quantized_model_path)

    # test_improvement(model, YOLO(quantized_model_path))
    time_model(model)
