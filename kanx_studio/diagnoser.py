import os
import xgboost as xgb
import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.decomposition import PCA


class KANDiagnoser:
    def __init__(self, model_path="kanx_studio_diagnoser.json"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"未找到诊断器权重文件: {model_path}")
        self.model = xgb.XGBClassifier()
        self.model.load_model(model_path)

        self.label_map = {
            0: {"status": "强烈推荐 KAN", "reason": "精度更高，且延迟在可接受范围内 (<= 5倍)。"},
            1: {"status": "建议改用 MLP (延迟敏感)",
                "reason": "尽管 KAN 精度更高，但推理耗时超 MLP 5 倍以上，存在工程瓶颈。"},
            2: {"status": "强烈推荐 MLP", "reason": "MLP 在此数据集上精度持平或更高，且耗时正常。高维/非平滑场景。"},
            3: {"status": "建议使用 KAN (罕见)", "reason": "MLP 精度虽高，但耗时异常巨大，退而求其次选择 KAN。"}
        }

    def _extract_meta_features(self, X, task_type):
        """复用你之前写的特征提取逻辑"""
        n_samples, n_features = X.shape
        sparsity = float(np.mean(X == 0))
        skew_val = np.mean(np.abs(skew(X, axis=0, nan_policy='omit')))
        kurt_val = np.mean(np.abs(kurtosis(X, axis=0, nan_policy='omit')))

        try:
            pca = PCA(n_components=0.95)
            pca.fit(X)
            intrinsic_dim = pca.n_components_ / max(1, n_features)
        except:
            intrinsic_dim = 1.0

        smoothness = float(np.nanvar(np.diff(X, axis=0))) if n_samples > 1 else 1.0
        task_code = {"regression": 0, "classification": 1, "time-series": 2}.get(task_type, 0)

        return {
            "n_samples": n_samples, "n_features": n_features,
            "samples_per_feature": n_samples / max(1, n_features),
            "sparsity": sparsity, "skewness": np.nan_to_num(skew_val),
            "kurtosis": np.nan_to_num(kurt_val), "intrinsic_dim": intrinsic_dim,
            "smoothness_proxy": min(smoothness, 10.0), "task_type_code": task_code
        }

    def evaluate(self, X, task_type="regression"):
        """一键诊断接口"""
        features = self._extract_meta_features(X, task_type)
        df_features = pd.DataFrame([features])

        pred_class = int(self.model.predict(df_features)[0])
        return pred_class, self.label_map[pred_class], features