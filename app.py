import streamlit as st
import pandas as pd
import numpy as np
import io
import zipfile
import os
from sklearn.impute import SimpleImputer
from kanx_studio.diagnoser import KANDiagnoser
from kanx_studio.benchmark import KANBenchmark
from kanx_studio.deployer import ModelDeployer

st.set_page_config(page_title="KANX-Studio", page_icon="🌌", layout="wide")

# ================= 侧边栏导航 =================
st.sidebar.title("🌌 KANX-Studio")
st.sidebar.markdown("生产级 KAN 网络工程与诊断平台")
page = st.sidebar.radio("导航菜单", ["🏠 主页", "🧠 智能适用性诊断", "⚖️ 自动化基准对比", "📦 一键生产部署"])

# ================= 状态管理 =================
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "input_dim" not in st.session_state:
    st.session_state.input_dim = 0
if "backend_choice" not in st.session_state:
    st.session_state.backend_choice = "tensorflow"

# ================= 页面逻辑 =================
if page == "🏠 主页":
    st.title("欢迎来到 KANX-Studio 🚀")
    st.markdown("""
    这是首个为 **Kolmogorov-Arnold Networks (KAN)** 打造的工程化诊断与部署工具包。
    - **智能诊断**: 不必盲目训练，上传特征让预训练的诊断器告诉你是否该用 KAN。
    - **基准对比**: 一键跑通 TF/PyTorch 双后端基准测试。
    - **一键部署**: 自动生成 ONNX、FastAPI 与 Docker 容器镜像。
    """)

elif page == "🧠 智能适用性诊断":
    st.title("🔍 KAN 适用性诊断器")
    uploaded_file = st.file_uploader("上传 CSV 数据集 (仅需特征 X)", type=["csv"])
    task_type = st.selectbox("选择任务类型", ["regression", "classification", "time-series"])

    if uploaded_file is not None and st.button("🚀 开始极速诊断"):
        df = pd.read_csv(uploaded_file)
        X_raw = df.select_dtypes(include=[np.number]).values

        # 【修复 NaN 报错】使用均值填补缺失值
        X = SimpleImputer(strategy='mean').fit_transform(X_raw)

        with st.spinner("调用 XGBoost 专家大脑..."):
            diagnoser = KANDiagnoser("kanx_studio_diagnoser.json")
            label, result_dict, features = diagnoser.evaluate(X, task_type=task_type)

            st.subheader("📊 诊断结果")
            col1, col2, col3 = st.columns(3)
            col1.metric("样本量 (N)", int(features['n_samples']))
            col2.metric("特征维度 (D)", int(features['n_features']))
            col3.metric("平滑度估计", round(features['smoothness_proxy'], 4))

            if label == 0:
                st.success(f"🎯 **{result_dict['status']}**\n\n{result_dict['reason']}")
            elif label == 1:
                st.warning(f"⚠️ **{result_dict['status']}**\n\n{result_dict['reason']}")
            elif label == 2:
                st.error(f"🚫 **{result_dict['status']}**\n\n{result_dict['reason']}")
            else:
                st.info(f"💡 **{result_dict['status']}**\n\n{result_dict['reason']}")

elif page == "⚖️ 自动化基准对比":
    st.title("⚖️ KAN vs MLP 性能大比拼")
    uploaded_file = st.file_uploader("上传带标签的 CSV 数据集", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        target_col = st.selectbox("请选择目标标签列 (y)", df.columns)
        task_type = st.radio("任务类型", ["regression", "classification"])
        backend = st.selectbox("计算后端", ["tensorflow", "pytorch"])

        if st.button("⚡ 开始训练与对比"):
            X_raw = df.drop(columns=[target_col]).select_dtypes(include=[np.number]).values
            y = df[target_col].values
            X = SimpleImputer(strategy='mean').fit_transform(X_raw)

            with st.spinner(f"正在使用 {backend.upper()} 训练双模型... (这可能需要几分钟)"):
                benchmark = KANBenchmark(epochs=20, backend=backend)
                res_dict = benchmark.run(X, y, task_type=task_type)

                # 保存模型以便后续部署
                st.session_state.trained_model = res_dict.pop("trained_kan_model")
                st.session_state.input_dim = X.shape[1]
                st.session_state.backend_choice = backend

                # 展示结果
                res_df = pd.DataFrame(res_dict).set_index("Model")
                st.success("✅ 训练与评估完成！")
                st.dataframe(res_df, use_container_width=True)

                metric_name = "MSE" if task_type == "regression" else "Accuracy"
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**🎯 精度对比 ({metric_name})** (越{'低' if task_type == 'regression' else '高'}越好)")
                    st.bar_chart(res_df[[metric_name]])
                with col2:
                    st.write("**⏱️ 延迟对比 (ms/batch)** (越低越好)")
                    st.bar_chart(res_df[["Latency (ms/batch)"]])

elif page == "📦 一键生产部署":
    st.title("📦 生产环境打包与导出")

    if st.session_state.trained_model is None:
        st.warning("⚠️ 尚未检测到训练好的 KAN 模型。请先前往 **『⚖️ 自动化基准对比』** 页面完成训练。")
    else:
        st.success(f"✅ 已检测到来自 {st.session_state.backend_choice.upper()} 的 KAN 模型，准备就绪。")

        if st.button("🛠️ 生成生产级部署包"):
            with st.spinner("正在转换为 ONNX 并生成接口代码..."):
                deployer = ModelDeployer(output_dir="deployment_package", backend=st.session_state.backend_choice)
                deployer.export_onnx(st.session_state.trained_model, st.session_state.input_dim)
                deployer.generate_api_and_docker()

                # 将文件夹打包为 ZIP (内存中，避免弄脏用户文件系统)
                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
                    for root, dirs, files in os.walk("deployment_package"):
                        for file in files:
                            file_path = os.path.join(root, file)
                            zip_file.write(file_path, arcname=file)

                st.success("🎉 生产包构建成功！点击下方按钮下载。")
                st.download_button(
                    label="⬇️ 下载部署包 (ZIP)",
                    data=zip_buffer.getvalue(),
                    file_name="KAN_Deployment_Package.zip",
                    mime="application/zip"
                )