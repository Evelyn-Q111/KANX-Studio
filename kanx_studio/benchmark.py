import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, accuracy_score


class KANBenchmark:
    def __init__(self, epochs=50, batch_size=128, backend="tensorflow"):
        self.epochs = epochs
        self.batch_size = batch_size
        self.backend = backend.lower()

    def _measure_latency(self, infer_func, dummy_input, runs=10):
        """通用延迟测试 (ms)"""
        for _ in range(3): infer_func(dummy_input)  # 预热
        start = time.perf_counter()
        for _ in range(runs): infer_func(dummy_input)
        return ((time.perf_counter() - start) / runs) * 1000.0

    def run(self, X, y, task_type="regression", hidden_dim=32):
        print(f"🚀 启动 {self.backend.upper()} 基准测试 (任务: {task_type})")
        is_reg = (task_type in ["regression", "time-series"])
        if not is_reg:
            label_encoder = LabelEncoder()
            y = label_encoder.fit_transform(y)
            out_dim = len(np.unique(y))
        else:
            out_dim = 1
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train = StandardScaler().fit_transform(X_train)
        X_test = StandardScaler().fit(X_train).transform(X_test)

        in_dim = X_train.shape[1]

        out_dim = 1 if is_reg else int(np.max(y)) + 1


        if is_reg:
            scaler_y = StandardScaler()
            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

        if self.backend == "tensorflow":
            return self._run_tensorflow(X_train, X_test, y_train, y_test, in_dim, out_dim, hidden_dim, is_reg)
        elif self.backend == "pytorch":
            return self._run_pytorch(X_train, X_test, y_train, y_test, in_dim, out_dim, hidden_dim, is_reg)
        else:
            raise ValueError("不支持的后端，请选择 'tensorflow' 或 'pytorch'")

    def _run_tensorflow(self, X_train, X_test, y_train, y_test, in_dim, out_dim, hidden_dim, is_reg):
        import tensorflow as tf
        from kanx import KAN
        tf.keras.backend.clear_session()
        loss_fn = "mse" if is_reg else "sparse_categorical_crossentropy"

        # 构建模型
        kan_model = tf.keras.Sequential([KAN([in_dim, hidden_dim, out_dim])])
        if not is_reg: kan_model.add(tf.keras.layers.Softmax())
        kan_model.build((None, in_dim))
        kan_params = int(sum(np.prod(v.shape) for v in kan_model.trainable_variables))

        mlp_h = max(4, int(np.sqrt(kan_params)))
        mlp_model = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_h, activation="silu", input_shape=(in_dim,)),
            tf.keras.layers.Dense(mlp_h, activation="silu"),
            tf.keras.layers.Dense(out_dim, activation=None if is_reg else "softmax")
        ])
        mlp_model.build((None, in_dim))
        mlp_params = int(sum(np.prod(v.shape) for v in mlp_model.trainable_variables))

        # 训练
        t0 = time.perf_counter()
        kan_model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss=loss_fn)
        kan_model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        kan_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        mlp_model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss=loss_fn)
        mlp_model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        mlp_time = time.perf_counter() - t0

        # 推理与评测
        kan_pred = kan_model(X_test, training=False).numpy()
        mlp_pred = mlp_model(X_test, training=False).numpy()

        metric_name = "MSE" if is_reg else "Accuracy"
        if is_reg:
            kan_metric, mlp_metric = float(mean_squared_error(y_test, kan_pred)), float(
                mean_squared_error(y_test, mlp_pred))
        else:
            kan_metric, mlp_metric = float(accuracy_score(y_test, np.argmax(kan_pred, axis=1))), float(
                accuracy_score(y_test, np.argmax(mlp_pred, axis=1)))

        kan_lat = self._measure_latency(lambda x: kan_model(x, training=False), X_test[:32])
        mlp_lat = self._measure_latency(lambda x: mlp_model(x, training=False), X_test[:32])

        return {
            "Model": ["KAN", "MLP (Baseline)"],
            "Params": [kan_params, mlp_params],
            metric_name: [kan_metric, mlp_metric],
            "Train Time (s)": [kan_time, mlp_time],
            "Latency (ms/batch)": [kan_lat, mlp_lat],
            "trained_kan_model": kan_model  # 用于传递给后续部署
        }

    def _run_tensorflow(self, X_train, X_test, y_train, y_test, in_dim, out_dim, hidden_dim, is_reg):
        import tensorflow as tf
        from kanx import KAN
        tf.keras.backend.clear_session()
        loss_fn = "mse" if is_reg else "sparse_categorical_crossentropy"

        kan_model = tf.keras.Sequential([KAN([in_dim, hidden_dim, out_dim])])
        if not is_reg: kan_model.add(tf.keras.layers.Softmax())
        kan_model.build((None, in_dim))
        kan_params = int(sum(np.prod(v.shape) for v in kan_model.trainable_variables))

        mlp_h = max(4, int(np.sqrt(kan_params)))
        mlp_model = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_h, activation="silu", input_shape=(in_dim,)),
            tf.keras.layers.Dense(mlp_h, activation="silu"),
            tf.keras.layers.Dense(out_dim, activation=None if is_reg else "softmax")
        ])
        mlp_model.build((None, in_dim))
        mlp_params = int(sum(np.prod(v.shape) for v in mlp_model.trainable_variables))

        t0 = time.perf_counter()
        kan_model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss=loss_fn)
        kan_model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        kan_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        mlp_model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss=loss_fn)
        mlp_model.fit(X_train, y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=0)
        mlp_time = time.perf_counter() - t0

        kan_pred = kan_model(X_test, training=False).numpy()
        mlp_pred = mlp_model(X_test, training=False).numpy()

        metric_name = "MSE" if is_reg else "Accuracy"
        if is_reg:
            kan_metric, mlp_metric = float(mean_squared_error(y_test, kan_pred)), float(
                mean_squared_error(y_test, mlp_pred))
        else:
            kan_metric, mlp_metric = float(accuracy_score(y_test, np.argmax(kan_pred, axis=1))), float(
                accuracy_score(y_test, np.argmax(mlp_pred, axis=1)))

        kan_lat = self._measure_latency(lambda x: kan_model(x, training=False), X_test[:32])
        mlp_lat = self._measure_latency(lambda x: mlp_model(x, training=False), X_test[:32])

        return {
            "Model": ["KAN", "MLP (Baseline)"], "Params": [kan_params, mlp_params],
            metric_name: [kan_metric, mlp_metric], "Train Time (s)": [kan_time, mlp_time],
            "Latency (ms/batch)": [kan_lat, mlp_lat], "trained_kan_model": kan_model
        }

    def _run_pytorch(self, X_train, X_test, y_train, y_test, in_dim, out_dim, hidden_dim, is_reg):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        from kanx.torch import KAN as PyTorchKAN

        X_train_t = torch.tensor(X_train, dtype=torch.float32)
        y_train_t = torch.tensor(y_train, dtype=torch.float32 if is_reg else torch.long)
        X_test_t = torch.tensor(X_test, dtype=torch.float32)
        loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.batch_size, shuffle=True)

        # 1. 构建官方 PyTorch KAN
        kan_model = PyTorchKAN([in_dim, hidden_dim, out_dim])
        kan_params = sum(p.numel() for p in kan_model.parameters() if p.requires_grad)

        # 2. 构建参数量匹配的 MLP 基线
        mlp_h = max(4, int(np.sqrt(kan_params)))
        mlp_model = nn.Sequential(
            nn.Linear(in_dim, mlp_h), nn.SiLU(),
            nn.Linear(mlp_h, mlp_h), nn.SiLU(),
            nn.Linear(mlp_h, out_dim)
        )
        mlp_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)

        criterion = nn.MSELoss() if is_reg else nn.CrossEntropyLoss()

        # 3. 训练循环 (保持公平的手动控制循环)
        def train_model(model):
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
            model.train()
            t0 = time.perf_counter()
            for _ in range(self.epochs):
                for b_X, b_y in loader:
                    optimizer.zero_grad()
                    # ⚠️ 注意: 回归任务需要 squeeze，分类任务输出原始 logits
                    pred = model(b_X).squeeze(-1) if is_reg else model(b_X)
                    loss = criterion(pred, b_y)
                    loss.backward()
                    optimizer.step()
            return time.perf_counter() - t0

        kan_time = train_model(kan_model)
        mlp_time = train_model(mlp_model)

        # 4. 推理评估
        kan_model.eval();
        mlp_model.eval()
        with torch.no_grad():
            kan_pred = kan_model(X_test_t).numpy()
            mlp_pred = mlp_model(X_test_t).numpy()

        metric_name = "MSE" if is_reg else "Accuracy"
        if is_reg:
            kan_metric = float(mean_squared_error(y_test, kan_pred))
            mlp_metric = float(mean_squared_error(y_test, mlp_pred))
        else:
            kan_metric = float(accuracy_score(y_test, np.argmax(kan_pred, axis=1)))
            mlp_metric = float(accuracy_score(y_test, np.argmax(mlp_pred, axis=1)))

        # 5. 推理延迟测试
        dummy_t = X_test_t[:32]
        kan_lat = self._measure_latency(lambda x: kan_model(x), dummy_t)
        mlp_lat = self._measure_latency(lambda x: mlp_model(x), dummy_t)

        return {
            "Model": ["KAN (kanx.torch)", "MLP (Baseline)"], "Params": [kan_params, mlp_params],
            metric_name: [kan_metric, mlp_metric], "Train Time (s)": [kan_time, mlp_time],
            "Latency (ms/batch)": [kan_lat, mlp_lat], "trained_kan_model": kan_model
        }