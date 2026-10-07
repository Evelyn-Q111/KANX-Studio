# 🌌 KANX-Studio
首个为 Kolmogorov-Arnold Networks (KAN) 打造的生产级网络工程与智能诊断平台，让 KAN 从“研究玩具”真正变成“可用工具”。

## 📖 目录
- [🎯 核心特性](#-核心特性)
- [📂 项目结构](#-项目结构)
- [🚀 快速开始](#-快速开始)
- [📚 核心 API 文档](#-核心-api-文档)

## 🎯 核心特性
1. **🤖 智能元学习诊断**: 上传 CSV，系统基于特征平滑度、本征维度等自动诊断是否建议使用 KAN。
2. **⚖️ 自动化基准评测**: 一键跑通 KAN 与 MLP 的精度、训练速度、推理延迟 (Latency) 对比。
3. **🔄 双后端统一支持**: `tensorflow` 和 `pytorch` 引擎无缝切换。
4. **📦 工业级一键部署**: 自动将模型转换为 ONNX，生成配套的 `FastAPI` (RESTful) 和 `Dockerfile`。

## 📂 项目结构
```text
KANX-Studio/
├── app.py                      # Streamlit 可视化交互前端入口
├── requirements.txt            # 项目依赖
├── README.md                   # 官方文档
├── kanx_studio_diagnoser.json  # XGBoost 预训练元专家大脑权重
└── kanx_studio/                # 核心处理引擎库
    ├── __init__.py
    ├── benchmark.py            # TF/PyTorch 双后端对照实验模块
    ├── deployer.py             # ONNX 导出与容器化组装模块
    └── diagnoser.py            # 提取元特征并推理适用性的模块
```

## 🚀 快速开始
1. 环境安装
```Bash
pip install -r requirements.txt
# 确保你安装了对应的后端，如: pip install tensorflow torch onnxruntime tf2onnx
```
2. 启动可视化 Studio 控制台
```Bash
streamlit run app.py
```
启动后，浏览器将自动打开 WebUI 页面。依次体验：智能诊断 -> 自动化对比 -> 一键生产部署。
## 📚 核心 API 文档 (支持代码调用)
除了使用 WebUI，你也可以在 Python 代码中直接调用：
1. 适用性诊断 (KANDiagnoser)
```Python
from kanx_studio.diagnoser import KANDiagnoser
diagnoser = KANDiagnoser(model_path="kanx_studio_diagnoser.json")
# X 为 numpy array, 自动处理 nan
label, advice, features = diagnoser.evaluate(X, task_type="regression")
print(advice['status']) # 输出推荐建议
```
2. 双后端基准测试 (KANBenchmark)
```Python
from kanx_studio.benchmark import KANBenchmark
# backend 支持 "tensorflow" 或 "pytorch"
benchmark = KANBenchmark(epochs=50, backend="pytorch")
results = benchmark.run(X, y, task_type="classification")
```
3. 一键生产环境打包 (ModelDeployer)
```Python
from kanx_studio.deployer import ModelDeployer
deployer = ModelDeployer(output_dir="./dist", backend="tensorflow")
deployer.export_onnx(model, input_dim=10)
deployer.generate_api_and_docker()
```