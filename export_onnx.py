from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer

model_id = "Dongjin-kr/ko-reranker"
save_dir = "./models/onnx_reranker"

print("🚀 ONNX 변환 및 다운로드 시작... (시간이 조금 소요됩니다)")
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForSequenceClassification.from_pretrained(model_id, export=True)

tokenizer.save_pretrained(save_dir)
model.save_pretrained(save_dir)
print("✅ ONNX 변환 및 저장 완료!")
