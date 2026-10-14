"""§5–§8 共用：讀檔、特徵工程、特徵組、評估工具。由 01_training.ipynb 與 02_analysis.ipynb 匯入。

資料來源採「hybrid」：
- 目標 y：pure_all.csv 的 target_PM2.5（原始單位 μg/m³）
- 原始欄位（z_ 開頭）：standardized_all.csv 的 z-score（已用 train 平均數補值、train 統計量標準化）
- 衍生特徵（eng_ 開頭）：從 pure_all.csv 的原始值計算，在 Pipeline 內 median 補值 + StandardScaler
- 原始值（r_ 開頭）：pure_all.csv 的原始 lag 欄位，給 persistence、HistGradientBoosting 與穩健性檢查使用
所有衍生特徵只用到 t−1 以前的觀測與目標時間的日曆資訊，預測時都已知。
"""

from pathlib import Path
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).resolve().parent
ROOT = MODEL_DIR.parent
DATA = ROOT / "Dataset" / "All_Metrics"
FIG = MODEL_DIR / "figures"
TAB = MODEL_DIR / "tables"
OUT = MODEL_DIR / "outputs"

SEED = 42
N_SPLITS = 5
GAP = 9  # 與 lag 窗口同長，驗證列的歷史窗口不會和訓練列的目標重疊
METRICS = [
    "AMB_TEMP", "CH4", "CO", "NMHC", "NO", "NO2", "NOx", "O3", "PM10", "PM2.5",
    "RAINFALL", "RH", "SO2", "THC", "WD_HR", "WIND_DIREC", "WIND_SPEED", "WS_HR",
]
LAGS = range(1, 10)
LAG_COLS = [f"{m}_lag_{lag}h" for lag in LAGS for m in METRICS]

# dataviz 參考色盤（與 EDA/eda.py 一致）
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
RED = "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"


def setup_plotting():
    plt.rcParams.update({
        "figure.dpi": 100, "savefig.dpi": 200, "font.size": 9, "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
        "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2,
    })
    for folder in (FIG, TAB, OUT):
        folder.mkdir(exist_ok=True)


def save_fig(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", facecolor="white")


def save_table(frame, name, index=False):
    frame.to_csv(TAB / f"{name}.csv", index=index, encoding="utf-8-sig")


# ---------------------------------------------------------------- 資料


def load_hybrid():
    """讀兩個檔並確認逐列對齊，回傳 y（pure）、Z（standardized）、R（pure 原始值）、缺值旗標、時間、切分。"""
    pure = pd.read_csv(DATA / "pure_all.csv", encoding="utf-8-sig", low_memory=False)
    std = pd.read_csv(DATA / "standardized_all.csv", encoding="utf-8-sig", low_memory=False)
    assert len(pure) == len(std), "兩檔列數不同"
    assert (pure["target_datetime"] == std["target_datetime"]).all(), "兩檔時間沒有逐列對齊"
    assert (pure["dataset_split"] == std["dataset_split"]).all(), "兩檔 train/test 標記不同"
    assert std[LAG_COLS].notna().all().all(), "standardized 檔應已補值"

    return SimpleNamespace(
        y=pure["target_PM2.5"].astype(float),
        Z=std[LAG_COLS].astype(float),
        R=pure[LAG_COLS].apply(pd.to_numeric, errors="coerce"),
        missing=pure[[f"{c}_missing" for c in LAG_COLS]].astype(int),
        time=pd.to_datetime(pure["target_datetime"]),
        is_train=pure["dataset_split"].eq("train").to_numpy(),
    )


def _flag(condition, source):
    """布林旗標，來源缺值時保留 NaN（交給 Pipeline 內的 imputer）。"""
    return condition.astype(float).where(source.notna())


def engineer(data):
    """從 pure 原始值產生衍生特徵（eng_ 開頭）。"""
    R, time = data.R, data.time
    pm = R[[f"PM2.5_lag_{lag}h" for lag in LAGS]]
    ws, wd, rain = R["WIND_SPEED_lag_1h"], np.deg2rad(R["WIND_DIREC_lag_1h"]), R["RAINFALL_lag_1h"]
    eng = pd.DataFrame(index=R.index)
    # 聚合：取代 PM2.5 lag 2–9h。lag1、trend、mean3 為 lag1–3 的三個獨立組合，mean6/mean9 再加入較遠的 lag，
    # 這樣與 lag1 不會完全共線。
    eng["eng_pm25_trend"] = pm.iloc[:, 0] - pm.iloc[:, 1]
    for k in (3, 6, 9):
        eng[f"eng_pm25_mean{k}h"] = pm.iloc[:, :k].mean(axis=1)
    eng["eng_pm25_std9h"] = pm.std(axis=1)
    # 風：角度轉 sin/cos，再乘上風速得到風向量。風向是「風吹來的方向」，依氣象慣例
    # u = −ws·sin(wd)（> 0 為向東吹）、v = −ws·cos(wd)（> 0 為向北吹）
    eng["eng_wd_sin"], eng["eng_wd_cos"] = np.sin(wd), np.cos(wd)
    eng["eng_wind_u"], eng["eng_wind_v"] = -ws * np.sin(wd), -ws * np.cos(wd)
    # 風速分段（Fig 4：<1 m/s 與 ≥6 m/s 的中位數明顯不同），基準為 1–6 m/s
    eng["eng_ws_lt1"] = _flag(ws < 1, ws)
    eng["eng_ws_ge6"] = _flag(ws >= 6, ws)
    eng["eng_rain_flag"] = _flag(rain > 0, rain)
    # 交互作用：風大時擴散較強，PM2.5 的持續性應該較弱
    eng["eng_pm25_x_ws"] = pm.iloc[:, 0] * ws
    # 日曆：目標時間的小時與週末（預測時已知）；不用月份，train 沒有 11–12 月
    hour = time.dt.hour
    eng["eng_hour_sin"], eng["eng_hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    eng["eng_weekend"] = (time.dt.dayofweek >= 5).astype(float)
    eng["eng_pm25_lag1_missing"] = data.missing["PM2.5_lag_1h_missing"].astype(float)
    return eng


def build_frame(data):
    """所有候選欄位放在同一張表：z_（standardized）、r_（pure 原始值）、eng_（衍生）。"""
    return pd.concat([
        data.Z.add_prefix("z_"),
        data.R.add_prefix("r_"),
        engineer(data),
    ], axis=1)


# ---------------------------------------------------------------- 特徵組

# Set C：依 EDA 的結論分組；共線組合擇一（刪除 NOx、THC、WS_HR、WD_HR）
GROUPS = {
    "persistence": ["z_PM2.5_lag_1h"],
    "pm25_aggregates": ["eng_pm25_trend", "eng_pm25_mean3h", "eng_pm25_mean6h", "eng_pm25_mean9h", "eng_pm25_std9h"],
    "co_pollutants": [f"z_{m}_lag_{lag}h" for m in ("PM10", "CO") for lag in (1, 2, 3)]
    + [f"z_{m}_lag_1h" for m in ("NO", "NO2", "O3", "SO2", "CH4", "NMHC")],
    "meteorology": ["z_AMB_TEMP_lag_1h", "z_RH_lag_1h", "z_RAINFALL_lag_1h", "eng_rain_flag"],
    "wind": ["z_WIND_SPEED_lag_1h", "eng_wd_sin", "eng_wd_cos", "eng_wind_u", "eng_wind_v"],
    "wind_bins": ["eng_ws_lt1", "eng_ws_ge6"],
    "interaction": ["eng_pm25_x_ws"],
    "calendar": ["eng_hour_sin", "eng_hour_cos", "eng_weekend"],
    "missing_flag": ["eng_pm25_lag1_missing"],
}
SET_A = [f"z_{c}" for c in LAG_COLS]  # 全部 162 個原始欄位（standardized）
SET_C = [c for cols in GROUPS.values() for c in cols]
RAW_ALL = [f"r_{c}" for c in LAG_COLS]  # HistGradientBoosting 用：原始單位、保留 NaN
PM25_RAW_LAGS = [f"z_PM2.5_lag_{lag}h" for lag in LAGS]


def set_c_without(group):
    return [c for c in SET_C if c not in GROUPS[group]]


def set_c_with_raw_lags():
    """H5 對照組：PM2.5 聚合特徵換回 lag 1–9h 原始欄位。"""
    return [c for c in SET_C if c not in GROUPS["pm25_aggregates"] + GROUPS["persistence"]] + PM25_RAW_LAGS


def to_raw_units(columns):
    """穩健性檢查：z_ 欄改用 pure 原始值（r_），在 fold 內補值與標準化。"""
    return [f"r_{c[2:]}" if c.startswith("z_") else c for c in columns]


def make_preprocessor(columns):
    """z_ 欄已經標準化 → passthrough；其餘欄（eng_、r_）在 fold 內 median 補值 + 標準化。"""
    z_cols = [c for c in columns if c.startswith("z_")]
    other = [c for c in columns if not c.startswith("z_")]
    transformers = []
    if z_cols:
        transformers.append(("z", "passthrough", z_cols))
    if other:
        transformers.append(("fit_in_fold", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), other))
    return ColumnTransformer(transformers, verbose_feature_names_out=False)


def linear_pipeline(columns, model):
    return Pipeline([("prep", make_preprocessor(columns)), ("model", model)])


class TopKSpearman(BaseEstimator, TransformerMixin):
    """Set B：在每個訓練 fold 內，依 |Spearman ρ| 取前 k 個特徵（不看驗證/測試資料）。"""

    def __init__(self, k=10):
        self.k = k

    def fit(self, X, y):
        ranks = pd.DataFrame(np.asarray(X, dtype=float)).rank()
        rho = ranks.corrwith(pd.Series(np.asarray(y, dtype=float)).rank()).abs().fillna(0.0)
        self.support_ = np.sort(np.argsort(-rho.to_numpy(), kind="stable")[: self.k])
        return self

    def transform(self, X):
        return np.asarray(X, dtype=float)[:, self.support_]


def topk_pipeline(columns, model):
    return Pipeline([("prep", make_preprocessor(columns)), ("select", TopKSpearman()), ("model", model)])


class Persistence(BaseEstimator, RegressorMixin):
    """Persistence baseline：預測值 = 上一小時的 PM2.5；缺值時往前找 lag2、lag3，仍缺則用 train 平均。"""

    def __init__(self, columns=("r_PM2.5_lag_1h", "r_PM2.5_lag_2h", "r_PM2.5_lag_3h")):
        self.columns = columns

    def fit(self, X, y):
        self.fallback_ = float(np.mean(y))
        return self

    def predict(self, X):
        values = X[list(self.columns)].bfill(axis=1).iloc[:, 0]
        return values.fillna(self.fallback_).to_numpy()


# ---------------------------------------------------------------- 驗證與指標


def make_folds(n_train):
    """TimeSeriesSplit：每個驗證 fold 都在其訓練資料之後，所有方法共用同一組索引。"""
    return list(TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP).split(np.arange(n_train)))


def scores(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": r2_score(y_true, y_pred),
    }


def cv_scores(estimator, X, y, folds):
    """逐 fold 重新 fit，回傳每個 fold 的 MAE / RMSE / R²。"""
    rows = []
    for fold, (train_idx, val_idx) in enumerate(folds, start=1):
        model = clone(estimator).fit(X.iloc[train_idx], y.iloc[train_idx])
        rows.append({"fold": fold, **scores(y.iloc[val_idx], model.predict(X.iloc[val_idx]))})
    return pd.DataFrame(rows)


def summarize(per_fold):
    """per_fold（含 method 欄）→ 每個方法的 mean ± std 表。"""
    stats = per_fold.groupby("method", sort=False)[["mae", "rmse", "r2"]].agg(["mean", "std"])
    stats.columns = [f"cv_{metric}_{stat}" for metric, stat in stats.columns]
    return stats.reset_index()


def vif(matrix):
    """VIF_j = (R⁻¹)_jj，R 為特徵相關矩陣（與 EDA/eda.py 相同的算法）。"""
    frame = pd.DataFrame(matrix)
    return pd.Series(np.diag(np.linalg.pinv(frame.corr().to_numpy())), index=frame.columns)


def block_bootstrap_indices(n, block, rng):
    """Moving-block bootstrap：隨機取起點、連續 block 列，接到長度 n，保留時間相依。"""
    starts = rng.integers(0, n - block + 1, size=int(np.ceil(n / block)))
    return np.concatenate([np.arange(s, s + block) for s in starts])[:n]
