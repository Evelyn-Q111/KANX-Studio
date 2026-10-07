import os
import gc
import time
import random
import itertools
import warnings
import traceback
from collections import Counter

# ==========================================
# 0. 环境优化配置
# ==========================================
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, accuracy_score
from scipy.stats import skew, kurtosis
from sklearn.datasets import fetch_california_housing, fetch_covtype, load_breast_cancer, load_diabetes, make_classification, make_regression, make_moons, make_circles, make_friedman1

import openml
import pmlb
import yfinance as yf
import urllib.request
from io import StringIO
import xgboost as xgb

from kanx import KAN

# 限制显存，防止多尺度并发时爆显存
try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

OLD_CSV = "meta_dataset.csv"
NEW_CSV = "meta_dataset_v2.csv"
MAX_SAMPLES = 3000


# ==========================================
# 1. 业务导向：四象限智能标签逻辑
# ==========================================
def get_recommendation_label(kan_mse, mlp_mse, kan_time, mlp_time):
    """
    四象限业务逻辑：
    0: 强烈推荐 KAN (KAN精度更高，且耗时在MLP的5倍以内)
    1: 建议用 MLP (KAN精度虽高，但耗时 > 5倍，延迟敏感)
    2: 强烈推荐 MLP (MLP精度更高或持平，且KAN耗时没比MLP快多少)
    3: 建议用 KAN (MLP精度虽高，但耗时极长，KAN比MLP快5倍以上)
    """
    # 注：MSE/Error越小精度越高
    time_ratio = kan_time / (mlp_time + 1e-9)

    if kan_mse < mlp_mse:
        # KAN 精度更高
        if time_ratio <= 5.0:
            return 0  # 类别0: 推荐 KAN
        else:
            return 1  # 类别1: 延迟过大，退而求其次用 MLP
    else:
        # MLP 精度更高 (或相等)
        if time_ratio >= 0.2:
            return 2  # 类别2: 推荐 MLP
        else:
            return 3  # 类别3: MLP太慢(KAN比它快5倍以上)，退而求其次用 KAN


# ==========================================
# 2. 特征提取模块 (兼容时序)
# ==========================================
def extract_meta_features(X: np.ndarray, y: np.ndarray, task_type: str) -> dict:
    n_samples, n_features = X.shape
    sparsity = float(np.mean(X == 0))

    skew_val = np.mean(np.abs(skew(X, axis=0, nan_policy='omit')))
    kurt_val = np.mean(np.abs(kurtosis(X, axis=0, nan_policy='omit')))
    if np.isnan(skew_val): skew_val = 0
    if np.isnan(kurt_val): kurt_val = 0

    try:
        pca = PCA(n_components=0.95)
        pca.fit(X)
        intrinsic_dim = pca.n_components_ / max(1, n_features)
    except:
        intrinsic_dim = 1.0

    task_code = {"regression": 0, "classification": 1, "time-series": 2}.get(task_type, 0)

    # 简化版的平滑度和可分离性(为了加速数据收集)
    # 此处省略复杂的随机森林提取，用数据本身的方差特性代替
    smoothness = float(np.var(np.diff(X, axis=0))) if n_samples > 1 else 1.0

    return {
        "n_samples": n_samples, "n_features": n_features,
        "samples_per_feature": n_samples / max(1, n_features),
        "sparsity": sparsity, "skewness": skew_val, "kurtosis": kurt_val,
        "intrinsic_dim": intrinsic_dim, "smoothness_proxy": min(smoothness, 10.0),
        "task_type_code": task_code
    }


# ==========================================
# 3. 多尺度基准测试引擎 (Multi-scale Benchmarking)
# ==========================================
def count_params(model: tf.keras.Model) -> int:
    return int(sum(np.prod(v.shape) for v in model.trainable_variables))


def run_multiscale_benchmark(X, y, task_type):
    """
    核心升级：在 16, 32, 64 三个尺度下测试，取最佳表现，记录平均时间。
    采用 ReduceLROnPlateau 和稍长的 EarlyStopping 进行折中。
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit(X_train).transform(X_test)

    in_dim = X_train.shape[1]
    is_reg = (task_type in ["regression", "time-series"])
    out_dim = 1 if is_reg else int(np.max(y)) + 1
    loss_fn = "mse" if is_reg else "sparse_categorical_crossentropy"

    # 【折中训练策略】：最高100轮，连续8轮不降则减小学习率，连续15轮不降则停止
    lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=0)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)

    kan_best_err, mlp_best_err = float('inf'), float('inf')
    kan_total_time, mlp_total_time = 0.0, 0.0
    scales = [16, 32, 64]  # 多尺度

    for hidden in scales:
        tf.keras.backend.clear_session()
        # --- KAN 构建 ---
        kan_model = tf.keras.Sequential([KAN([in_dim, hidden, out_dim])])
        if not is_reg: kan_model.add(tf.keras.layers.Softmax())
        kan_model.build((None, in_dim))

        # --- 对齐 MLP 构建 ---
        mlp_h = max(4, int(np.sqrt(count_params(kan_model))))
        mlp_model = tf.keras.Sequential([
            tf.keras.layers.Dense(mlp_h, activation="silu", input_shape=(in_dim,)),
            tf.keras.layers.Dense(mlp_h, activation="silu"),
            tf.keras.layers.Dense(out_dim, activation=None if is_reg else "softmax")
        ])

        # --- KAN 训练 ---
        kan_model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss=loss_fn)
        t0 = time.perf_counter()
        kan_model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=128,
                      verbose=0, callbacks=[lr_scheduler, early_stop])
        kan_total_time += (time.perf_counter() - t0)

        # --- MLP 训练 ---
        mlp_model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss=loss_fn)
        t0 = time.perf_counter()
        mlp_model.fit(X_train, y_train, validation_split=0.2, epochs=100, batch_size=128,
                      verbose=0, callbacks=[lr_scheduler, early_stop])
        mlp_total_time += (time.perf_counter() - t0)

        # --- 评估 ---
        if is_reg:
            k_err = float(mean_squared_error(y_test, kan_model.predict(X_test, verbose=0)))
            m_err = float(mean_squared_error(y_test, mlp_model.predict(X_test, verbose=0)))
        else:
            k_err = float(1.0 - accuracy_score(y_test, np.argmax(kan_model.predict(X_test, verbose=0), axis=1)))
            m_err = float(1.0 - accuracy_score(y_test, np.argmax(mlp_model.predict(X_test, verbose=0), axis=1)))

        kan_best_err = min(kan_best_err, k_err)
        mlp_best_err = min(mlp_best_err, m_err)

        del kan_model, mlp_model
        tf.keras.backend.clear_session()
        gc.collect()

    # 取平均时间，最佳误差
    return kan_best_err, mlp_best_err, kan_total_time / 3.0, mlp_total_time / 3.0


# ==========================================
# 4. 数据采集管道 (旧数据重构 + 海量时序数据)
# ==========================================
def preprocess_tabular(X, y, task_type):
    if X is None or y is None or X.shape[1] == 0: return None, None
    X = SimpleImputer(strategy='median').fit_transform(X)

    if task_type == "regression":
        y = pd.to_numeric(pd.Series(y.ravel()), errors='coerce').values
    else:
        y_flat = y.ravel()
        # 强制过滤掉连续数值型（浮点数）被误当成分类任务的情况
        if np.issubdtype(y_flat.dtype, np.floating) and len(np.unique(y_flat)) > 20:
            return None, None
        y = LabelEncoder().fit_transform(y.ravel())
        num_classes = len(np.unique(y))

        if num_classes < 2 or num_classes > 50:
            return None, None

        if num_classes > 20 and (len(y) / num_classes) < 10:
            return None, None

    valid_mask = ~pd.isna(y)
    X, y = X[valid_mask], y[valid_mask]

    if len(y) > MAX_SAMPLES:
        idx = np.random.choice(len(y), MAX_SAMPLES, replace=False)
        X, y = X[idx], y[idx]

    if len(y) < 100: return None, None
    return X, y


def ts_sliding_window(series, window_size):
    """将一维时序数据转化为 (样本数, window_size) 的监督学习格式"""
    series = series[~np.isnan(series)]
    if len(series) < window_size + 50: return None, None
    X, y = [], []
    for i in range(len(series) - window_size):
        X.append(series[i: i + window_size])
        y.append(series[i + window_size])
    return np.array(X), np.array(y)


def fetch_academic_real_ts(processed_set, target_count, current_count):
    """
    【学术界真实时序挖掘引擎】: 零 API 限制，100% 真实世界数据
    数据来源: M4 预测竞赛 (宏观经济、金融、工业)、清华 ETTh1 电力数据集
    """
    if current_count >= target_count: return
    print(f"\n📈 [均衡器] 启动【学术界真实时序库】挖掘 ({current_count}/{target_count})...")

    yielded_count = current_count
    window_sizes = [10, 20, 30]  # 使用不同滑窗生成不同的特征维度

    # ---------------------------------------------------------
    # 来源 1：顶级多变量真实数据集 (ETT 电力变压器温度, 气象数据)
    # 这些是单文件多列的数据，每一列代表一个真实的物理传感器序列
    # ---------------------------------------------------------
    multivariate_sources = {
        "ETTh1": "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv",
        "ExchangeRate": "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz"
    }

    for ds_name, url in multivariate_sources.items():
        if yielded_count >= target_count: return
        print(f"正在拉取 {ds_name} 真实数据集...")
        try:
            if ds_name == "ExchangeRate":
                df = pd.read_csv(url, header=None)
                cols = df.columns
            else:
                df = pd.read_csv(url)
                cols = [c for c in df.columns if c != 'date']

            for col in cols:
                series = df[col].dropna().values.astype(float)
                # 随机采样以防止序列过长导致训练过慢
                if len(series) > 5000:
                    start_idx = np.random.randint(0, len(series) - 3000)
                    series = series[start_idx:start_idx + 3000]

                for w in window_sizes:
                    if yielded_count >= target_count: return
                    name_tag = f"RealTS_{ds_name}_Feature{col}_w{w}"
                    if name_tag in processed_set: continue

                    X, y = ts_sliding_window(series, w)
                    if X is not None:
                        yield name_tag, X, y, "time-series"
                        yielded_count += 1
        except Exception as e:
            print(f"  -> {ds_name} 拉取失败，跳过。({str(e)[:50]})")

    # ---------------------------------------------------------
    # 来源 2：M4 预测竞赛数据集 (Hourly & Daily)
    # 这是单文件多行的极品数据集，每行是一条独立的真实金融/经济序列
    # ---------------------------------------------------------
    m4_sources = {
        "M4_Hourly": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Hourly-train.csv",
        # 414 条序列
        "M4_Daily": "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/Dataset/Train/Daily-train.csv"
        # 4227 条序列
    }

    for ds_name, url in m4_sources.items():
        if yielded_count >= target_count: return
        print(f"正在拉取 {ds_name} 宏观真实数据集 (这可能需要十几秒下载)...")
        try:
            # 读取在线 CSV
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            response = urllib.request.urlopen(req)
            csv_data = response.read().decode('utf-8')
            df = pd.read_csv(StringIO(csv_data))

            # M4 的格式：第一列是序列名(V1)，后面是时间序列数据(包含NaN)
            for index, row in df.iterrows():
                if yielded_count >= target_count: return

                series_name = row.iloc[0]
                # 提取数值并去除末尾的 NaN
                series = row.iloc[1:].dropna().values.astype(float)

                if len(series) < 100: continue
                if len(series) > MAX_SAMPLES: series = series[-MAX_SAMPLES:]

                for w in window_sizes:
                    if yielded_count >= target_count: return
                    name_tag = f"RealTS_{ds_name}_{series_name}_w{w}"
                    if name_tag in processed_set: continue

                    X, y = ts_sliding_window(series, w)
                    if X is not None:
                        yield name_tag, X, y, "time-series"
                        yielded_count += 1
        except Exception as e:
            print(f"  -> {ds_name} 拉取失败，跳过。({str(e)[:50]})")


def fetch_openml_tasks(processed_set, target_count, current_count, task_type):
    """
    统一的 OpenML 挖掘引擎 (支持分类/回归)
    突破了原有的严格限制，通过本地降采样解锁数千个新数据集
    """
    if current_count >= target_count: return

    task_enum = openml.tasks.TaskType.SUPERVISED_CLASSIFICATION if task_type == "classification" \
        else openml.tasks.TaskType.SUPERVISED_REGRESSION

    task_name_zh = "分类" if task_type == "classification" else "回归"
    print(f"\n🔍 [均衡器] 当前{task_name_zh}数据 ({current_count}/{target_count})，启动 OpenML 深水区挖掘...")

    try:
        tasks = openml.tasks.list_tasks(task_type=task_enum, output_format='dataframe')

        # 【核心突破】：允许最高 200,000 行的超大数据集，允许最多 300 个特征！
        valid_tasks = tasks[(tasks['NumberOfInstances'] >= 50) &
                            (tasks['NumberOfInstances'] <= 250000) &
                            (tasks['NumberOfFeatures'] <= 500)].sort_values('NumberOfInstances')

        # 去重：OpenML 有时会对同一个 dataset_id (did) 创建多个交叉验证的 task
        # 为了保证数据的绝对多样性，我们每个 did 只取一个
        valid_tasks = valid_tasks.drop_duplicates(subset=['did'])

        yielded_count = current_count
        for _, task in valid_tasks.iterrows():
            if yielded_count >= target_count: break
            name_tag = f"OpenML_{task_type}_{task['name']}_{task['tid']}"
            if name_tag in processed_set: continue

            try:
                # 随机休眠防封锁
                time.sleep(np.random.uniform(0.1, 0.5))
                dataset = openml.datasets.get_dataset(task['did'], download_data=True,
                                                      download_qualities=False, download_features_meta_data=False)
                X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
                X = X.select_dtypes(include=[np.number]).values

                # 【安全锁】：如果特征依然超过 150 (可能是独热编码展开导致的)，直接截断保留前 150 个特征
                if X.shape[1] > 150:
                    X = X[:, :150]

                X, y = preprocess_tabular(X, y.values if hasattr(y, 'values') else y, task_type)

                if X is not None:
                    yield name_tag, X, y, task_type
                    yielded_count += 1
            except Exception:
                continue
    except Exception as e:
        print(f"OpenML 抓取异常: {e}")


def fetch_historical_dataset(name_tag):
    """解析旧CSV的名字，重新下载数据"""
    try:
        if name_tag.startswith("UCI_"):
            parts = name_tag.split("_")
            task_type = parts[1]
            dataset_name = "_".join(parts[2:])
            X, y = pmlb.fetch_data(dataset_name, return_X_y=True)
            return preprocess_tabular(X, y, task_type) + (task_type,)

        elif name_tag.startswith("OpenML_"):
            parts = name_tag.split("_")
            task_type = parts[1]
            tid = int(parts[-1])
            task = openml.tasks.get_task(tid)
            dataset = task.get_dataset()
            X, y, _, _ = dataset.get_data(dataset_format="dataframe", target=dataset.default_target_attribute)
            X = X.select_dtypes(include=[np.number]).values
            return preprocess_tabular(X, y.values, task_type) + (task_type,)
    except Exception:
        return None, None, None
    return None, None, None


def fetch_sklearn_real_data(processed_set, target_count, current_count, task_type):
    """提取 Scikit-Learn 内置的顶级经典真实数据集"""
    if current_count >= target_count: return
    print(f"\n💎 [均衡器] 启动 Scikit-Learn 经典真实数据挖掘 ({task_type})...")

    yielded_count = current_count

    datasets = {
        "regression": [("SK_California", fetch_california_housing), ("SK_Diabetes", load_diabetes)],
        "classification": [("SK_CovType", fetch_covtype), ("SK_BreastCancer", load_breast_cancer)]
    }

    for name_tag, loader in datasets.get(task_type, []):
        if yielded_count >= target_count: break
        if name_tag in processed_set: continue
        try:
            data = loader()
            X, y = data.data, data.target
            X, y = preprocess_tabular(X, y, task_type)
            if X is not None:
                yield name_tag, X, y, task_type
                yielded_count += 1
        except Exception:
            pass


def fetch_pmlb_tasks(processed_set, target_count, current_count, task_type):
    """提取宾夕法尼亚大学维护的 PMLB (Penn Machine Learning Benchmarks) 纯正真实数据"""
    if current_count >= target_count: return

    task_name_zh = "分类" if task_type == "classification" else "回归"
    print(f"\n🏛️ [均衡器] 启动 PMLB 宾大真实{task_name_zh}数据库挖掘...")

    yielded_count = current_count
    dataset_names = pmlb.classification_dataset_names if task_type == "classification" else pmlb.regression_dataset_names

    for ds_name in dataset_names:
        if yielded_count >= target_count: break
        name_tag = f"PMLB_{task_type}_{ds_name}"
        if name_tag in processed_set: continue

        try:
            X, y = pmlb.fetch_data(ds_name, return_X_y=True)
            X, y = preprocess_tabular(X, y, task_type)
            if X is not None:
                yield name_tag, X, y, task_type
                yielded_count += 1
        except Exception:
            pass


def fetch_synthetic_classification(processed_set, target_count, current_count):
    """【无限引擎】生成高价值合成连续分类流形 (契合 KAN 对非线性边界的学习)"""
    if current_count >= target_count: return
    print(f"\n🧬 [兜底均衡器] 启动【分类】合成流形生成引擎，补齐剩余 {target_count - current_count} 个...")

    yielded_count = current_count
    while yielded_count < target_count:
        n_samples = np.random.randint(500, 3000)
        n_features = np.random.randint(4, 100)
        algo = np.random.choice(['classification', 'moons', 'circles'])

        name_tag = f"Synthetic_Clf_{algo}_{yielded_count}_{n_features}F"
        if name_tag in processed_set:
            yielded_count += 1
            continue

        if algo == 'classification':
            n_inf = np.random.randint(2, max(3, n_features // 2))
            X, y = make_classification(n_samples=n_samples, n_features=n_features, n_informative=n_inf,
                                       random_state=yielded_count)
        elif algo == 'moons':
            X, y = make_moons(n_samples=n_samples, noise=np.random.uniform(0.1, 0.3), random_state=yielded_count)
        else:
            X, y = make_circles(n_samples=n_samples, noise=np.random.uniform(0.05, 0.2),
                                factor=np.random.uniform(0.1, 0.8), random_state=yielded_count)

        yield name_tag, X, y, "classification"
        yielded_count += 1


def fetch_synthetic_regression(processed_set, target_count, current_count):
    """【无限引擎】生成高难度数学回归函数 (原版 KAN 论文最常用的验证方式)"""
    if current_count >= target_count: return
    print(f"\n🧬 [兜底均衡器] 启动【回归】数学合成引擎，补齐剩余 {target_count - current_count} 个...")

    yielded_count = current_count
    while yielded_count < target_count:
        n_samples = np.random.randint(500, 3000)
        n_features = np.random.randint(4, 100)
        algo = np.random.choice(['regression', 'friedman1'])

        name_tag = f"Synthetic_Reg_{algo}_{yielded_count}_{n_features}F"
        if name_tag in processed_set:
            yielded_count += 1
            continue

        if algo == 'regression':
            X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=np.random.uniform(0.1, 2.0),
                                   random_state=yielded_count)
        else:
            X, y = make_friedman1(n_samples=n_samples, n_features=max(5, n_features), noise=np.random.uniform(0.1, 2.0),
                                  random_state=yielded_count)

        yield name_tag, X, y, "regression"
        yielded_count += 1

def dataset_generator():
    """无限数据流：优先重跑旧CSV，然后抓取海量新时序"""
    processed_set = set()
    clf_count, reg_count, ts_count = 0, 0, 0

    # 阶段1：读取并重算旧数据
    if os.path.exists(NEW_CSV):
        df_new = pd.read_csv(NEW_CSV)
        processed_set = set(pd.read_csv(NEW_CSV)['dataset'].unique())

        task_counts = df_new['task_type_code'].value_counts().to_dict()
        reg_count = task_counts.get(0, 0)
        clf_count = task_counts.get(1, 0)
        ts_count = task_counts.get(2, 0)
        print(f"找到 V2 存档，已跳过 {len(processed_set)} 个已重算的数据集。")

    if os.path.exists(OLD_CSV):
        old_df = pd.read_csv(OLD_CSV)
        old_names = old_df['dataset'].unique()
        print(f"检测到旧版 V1 记录 {len(old_names)} 条，开始进行【多尺度安全升级迁移】...")
        for name in old_names:
            if name in processed_set: continue
            X, y, task_type = fetch_historical_dataset(name)
            if X is not None:
                yield name, X, y, task_type

    TARGET = 2500

    print(f"\n📊 现状统计：分类={clf_count}, 回归={reg_count}, 时序={ts_count} | 各自目标量: {TARGET}")

    # 1. 疯狂补充【分类】数据
    if clf_count < TARGET:
        for item in fetch_sklearn_real_data(processed_set, TARGET, clf_count, "classification"):
            yield item
        for item in fetch_pmlb_tasks(processed_set, TARGET, clf_count, "classification"):
            yield item
        for item in fetch_openml_tasks(processed_set, TARGET, clf_count, "classification"):
            yield item
        for item in fetch_synthetic_classification(processed_set, TARGET, clf_count):
            yield item

    # 2. 疯狂补充【回归】数据
    if reg_count < TARGET:
        for item in fetch_sklearn_real_data(processed_set, TARGET, reg_count, "regression"):
            yield item
        for item in fetch_pmlb_tasks(processed_set, TARGET, reg_count, "regression"):
            yield item
        for item in fetch_openml_tasks(processed_set, TARGET, reg_count, "regression"):
            yield item
        for item in fetch_synthetic_regression(processed_set, TARGET, reg_count):
            yield item

    # 3. 疯狂补充【时序】数据
    if ts_count < TARGET:
        # 这里调用你上一次替换的学术时序库函数
        for item in fetch_academic_real_ts(processed_set, TARGET, ts_count):
            yield item


# ==========================================
# 5. 核心循环与大脑训练
# ==========================================
def main_loop():
    print("====== 🚀 KANX-Studio 数据收集与多尺度评测引擎 ======")
    print("提示：程序将无限运行并收集数据。随时按 [Ctrl+C] 中断收集，")
    print("程序会自动保存进度，并立刻进入【四象限分类器】的训练阶段！\n")

    count = 0
    try:
        for name, X, y, task_type in dataset_generator():
            try:
                # 提取特征
                features = extract_meta_features(X, y, task_type)

                # 运行多尺度基准
                k_err, m_err, k_time, m_time = run_multiscale_benchmark(X, y, task_type)

                # 四象限标签
                label = get_recommendation_label(k_err, m_err, k_time, m_time)

                record = {
                    "dataset": name, **features,
                    "kan_mse": k_err, "mlp_mse": m_err,
                    "kan_time": k_time, "mlp_time": m_time,
                    "recommendation_class": label
                }

                df = pd.DataFrame([record])
                df.to_csv(NEW_CSV, mode='a', header=not os.path.exists(NEW_CSV), index=False)

                count += 1
                print(f"[{count}] 评估完成: {name} | KAN_err={k_err:.4f}, MLP_err={m_err:.4f} -> 归类于象限 {label}")

            except Exception as e:
                # print(f"评估失败 {name}: {e}")
                pass
            finally:
                gc.collect()

    except KeyboardInterrupt:
        print("\n\n🛑 收到中断信号！停止数据采集。")
        print("====== 🧠 准备进入 KANX-Studio 大脑训练阶段 ======")


def train_diagnoser():
    if not os.path.exists(NEW_CSV):
        print("未找到 V2 数据集，无法训练。")
        return

    df = pd.read_csv(NEW_CSV)
    if len(df) < 50:
        print(f"数据量过少 ({len(df)}条)，建议多收集一些再训练。")
        return

    print(f"\n📊 开始训练四象限诊断大脑 (总数据量: {len(df)} 条)")

    # 打印标签分布映射
    label_map = {
        0: "强烈推荐 KAN (精度高，延迟<=5倍)",
        1: "推荐 MLP (尽管KAN精度高，但由于延迟>5倍过于敏感)",
        2: "强烈推荐 MLP (MLP精度占优，耗时正常)",
        3: "推荐 KAN (尽管MLP精度占优，但耗时太久)"
    }

    print("\n当前各象限数据分布：")
    distribution = df['recommendation_class'].value_counts().to_dict()
    for k, v in sorted(distribution.items()):
        print(f" - [象限 {k}] {label_map[k]}: {v} 条")

    # 特征准备
    feature_cols = ["n_samples", "n_features", "samples_per_feature",
                    "sparsity", "skewness", "kurtosis", "intrinsic_dim",
                    "smoothness_proxy", "task_type_code"]
    X = df[feature_cols].fillna(0)
    y = df["recommendation_class"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 使用 XGBoost 训练多分类模型
    print("\n训练 XGBoost (multi:softmax) 模型中...")
    clf = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=4,
        max_depth=5,
        learning_rate=0.05,
        n_estimators=150,
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42
    )

    clf.fit(X_train, y_train)

    # 评估
    preds = clf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n✅ 诊断器训练完成！测试集准确率: {acc * 100:.2f}%")

    print("\n🔍 诊断大脑决策依据 (Feature Importance):")
    importances = clf.feature_importances_
    # 将特征名和重要性打包并按重要性降序排序
    feat_imps = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    for feat, imp in feat_imps:
        print(f"  - {feat:20s}: {imp * 100:.2f}%")

    # 保存为 JSON 供 kanx_studio 调用
    model_path = "kanx_studio_diagnoser.json"
    clf.save_model(model_path)
    print(f"💾 大脑权重已保存至: {model_path}")
    print("\n🎉 现在，你可以将此文件放入 kanx_studio 目录下，完成作业的闭环！")


if __name__ == "__main__":
    # 1. 无限循环收集数据 & 重写旧数据
    main_loop()

    # 2. 捕捉到 Ctrl+C 后，立刻训练分类器
    train_diagnoser()