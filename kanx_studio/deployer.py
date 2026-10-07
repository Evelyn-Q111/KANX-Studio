import os


class ModelDeployer:
    def __init__(self, output_dir="deployment_package", backend="tensorflow"):
        self.output_dir = output_dir
        self.backend = backend.lower()
        os.makedirs(self.output_dir, exist_ok=True)

    def export_onnx(self, model, input_dim, model_name="kan_model"):
        onnx_path = os.path.join(self.output_dir, f"{model_name}.onnx")

        if self.backend == "tensorflow":
            import tensorflow as tf
            import tf2onnx
            @tf.function(input_signature=[tf.TensorSpec([None, input_dim], tf.float32, name='features')])
            def inference_func(inputs):
                return model(inputs, training=False)

            tf2onnx.convert.from_function(
                inference_func,
                input_signature=[tf.TensorSpec([None, input_dim], tf.float32, name='features')],
                output_path=onnx_path
            )

        elif self.backend == "pytorch":
            import torch
            from kanx.torch import export_onnx as kanx_export_onnx

            dummy_input = torch.zeros(1, input_dim, dtype=torch.float32)

            kanx_export_onnx(
                model=model,
                path=onnx_path,
                sample_input=dummy_input,
                input_name="features",
                output_name="output"
            )

        print(f"✅ ONNX 模型已成功导出至: {onnx_path}")
        return onnx_path

    def generate_api_and_docker(self, model_name="kan_model"):
        # 1. 生成 FastAPI 脚本
        app_code = f"""from fastapi import FastAPI
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np

app = FastAPI(title="KANX-Studio Production API")
session = ort.InferenceSession("{model_name}.onnx")

class InputData(BaseModel):
    features: list

@app.post("/predict")
def predict(data: InputData):
    input_arr = np.array(data.features, dtype=np.float32).reshape(1, -1)
    outputs = session.run(None, {{"features": input_arr}})
    return {{"prediction": outputs[0].tolist()}}

@app.get("/health")
def health():
    return {{"status": "Healthy", "model": "{model_name}.onnx"}}
"""
        with open(os.path.join(self.output_dir, "app.py"), "w", encoding="utf-8") as f:
            f.write(app_code)

        # 2. 生成 Dockerfile
        docker_code = """FROM python:3.10-slim
WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir fastapi uvicorn onnxruntime numpy pydantic
EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""
        with open(os.path.join(self.output_dir, "Dockerfile"), "w", encoding="utf-8") as f:
            f.write(docker_code)

        return ["app.py", "Dockerfile", f"{model_name}.onnx"]