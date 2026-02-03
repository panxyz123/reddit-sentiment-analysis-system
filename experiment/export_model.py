from optimum.onnxruntime import ORTModelForSequenceClassification, ORTQuantizer
from optimum.onnxruntime.configuration import AutoQuantizationConfig
from transformers import AutoTokenizer
import os

model_path = "./model"
export_path = "./model_onnx_quantized"

# 1. Output ONNX (FP32)
model = ORTModelForSequenceClassification.from_pretrained(model_path, export=True)
tokenizer = AutoTokenizer.from_pretrained(model_path)
model.save_pretrained(export_path)

# 2. Quantization (INT8)
quantizer = ORTQuantizer.from_pretrained(export_path)
dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)

# Save quantized model
quantizer.quantize(
    save_dir=export_path,
    quantization_config=dqconfig,
)

print("\n--- Final Size Check ---")
original_size = os.path.getsize(os.path.join(export_path, "model.onnx")) / 1e6
quant_size = os.path.getsize(os.path.join(export_path, "model_quantized.onnx")) / 1e6

print(f"📦 FP32 ONNX Size: {original_size:.2f} MB")
print(f"⚡ INT8 Quantized Size: {quant_size:.2f} MB")