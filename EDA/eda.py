"""§4 Exploratory Feature Analysis（作業 PDF p.9），並順帶產出 §3 的統計表（p.8）。

所有統計與圖表只用 train（dataset_split == "train"），test 不會被讀進分析。
執行：python EDA/eda.py   （套件見 requirements.txt）
輸出：EDA/figures/*.png|pdf、EDA/tables/*.csv
"""

import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "Dataset"
FIG = Path(__file__).resolve().parent / "figures"
TAB = Path(__file__).resolve().parent / "tables"

# 環境部測項單位（圖表軸標籤需附單位，p.4）
UNITS = {
    "AMB_TEMP": "°C", "CH4": "ppm", "CO": "ppm", "NMHC": "ppm", "NO": "ppb", "NO2": "ppb",
    "NOx": "ppb", "O3": "ppb", "PM10": "μg/m³", "PM2.5": "μg/m³", "RAINFALL": "mm", "RH": "%",
    "SO2": "ppb", "THC": "ppm", "WD_HR": "deg", "WIND_DIREC": "deg", "WIND_SPEED": "m/s", "WS_HR": "m/s",
}
ANGLES = ["WIND_DIREC", "WD_HR"]  # 0–360 度，線性模型前要轉 sin/cos

# dataviz 參考色盤（light）
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN]
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
DIVERGING = LinearSegmentedColormap.from_list("blue_gray_red", [BLUE, "#f0efec", "#e34948"])

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 200, "font.size": 9, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True, "lines.linewidth": 2,
})


def load_parser():
    """重用 preprocessing_all.py 的原始資料解析（含缺值原因），避免重寫一份。"""
    path = DATA / "All_Metrics" / "preprocessing_all.py"
    spec = importlib.util.spec_from_file_location("preprocessing_all", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def save(fig, name):
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIG / f"{name}.{ext}", facecolor="white")
    plt.close(fig)


def corr_pair(x, y):
    """成對刪除缺值後的 Pearson、Spearman（rank 後的 Pearson，不需要 scipy）。"""
    mask = x.notna() & y.notna()
    x, y = x[mask], y[mask]
    return x.corr(y), x.rank().corr(y.rank()), int(mask.sum())


def with_angles(frame):
    frame = frame.copy()
    for angle in ANGLES:
        rad = np.deg2rad(frame.pop(angle))
        frame[f"{angle}_sin"], frame[f"{angle}_cos"] = np.sin(rad), np.cos(rad)
    return frame


def vif(frame):
    """VIF_j = (R⁻¹)_jj，R 為完整觀測列的相關矩陣。"""
    complete = frame.dropna()
    values = np.diag(np.linalg.inv(complete.corr().to_numpy()))
    return pd.Series(values, index=frame.columns, name="VIF"), len(complete)


def main():
    FIG.mkdir(exist_ok=True)
    TAB.mkdir(exist_ok=True)

    df = pd.read_csv(DATA / "All_Metrics" / "pure_all.csv", encoding="utf-8-sig",
                     parse_dates=["target_datetime"], low_memory=False)
    split_time = df.loc[df["dataset_split"] == "test", "target_datetime"].min()
    train = df[df["dataset_split"] == "train"]
    y = train["target_PM2.5"]
    metrics = [c[: -len("_lag_1h")] for c in df.columns if c.endswith("_lag_1h")]
    print(f"train：{len(train)} 列（目標時間 < {split_time}）")

    # ---------- §3：原始每小時資料（只取 train 期間）----------
    prep = load_parser()
    _, measurements = prep.read_measurements(DATA / "新竹_2025.csv")
    hourly = pd.DataFrame({m: {t: v for t, (v, _) in d.items()} for m, d in measurements.items()}).sort_index()
    reasons = pd.DataFrame({m: {t: r for t, (_, r) in d.items()} for m, d in measurements.items()}).sort_index()
    hourly, reasons = hourly[hourly.index < split_time].astype(float), reasons[reasons.index < split_time]

    summary = pd.DataFrame({
        "unit": pd.Series(UNITS),
        "n_obs": hourly.notna().sum(),
        "n_missing": hourly.isna().sum(),
        "missing_%": (hourly.isna().mean() * 100).round(2),
        "mean": hourly.mean(), "std": hourly.std(ddof=0), "min": hourly.min(),
        "p50": hourly.median(), "max": hourly.max(),
    }).loc[metrics].round(3)
    summary.to_csv(TAB / "summary_stats.csv", encoding="utf-8-sig")

    missing = reasons.melt(var_name="metric", value_name="reason").query("reason != ''")
    pd.crosstab(missing["metric"], missing["reason"], margins=True, margins_name="total").to_csv(
        TAB / "missing_by_reason.csv", encoding="utf-8-sig")

    target_range = y.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).round(2)
    target_range.to_csv(TAB / "target_range.csv", header=["target_PM2.5 (μg/m³)"], encoding="utf-8-sig")

    # ---------- §4 C：與目標的關聯排行（lag 1h）----------
    lag1 = with_angles(train[[f"{m}_lag_1h" for m in metrics]].set_axis(metrics, axis=1))
    rows = []
    for name in lag1.columns:
        pearson, spearman, n = corr_pair(lag1[name], y)
        rows.append({"feature": f"{name}_lag_1h", "pearson_r": pearson, "spearman_r": spearman, "n": n})
    ranking = pd.DataFrame(rows).sort_values("spearman_r", key=abs, ascending=False).round(3)
    ranking.to_csv(TAB / "correlation_ranking.csv", index=False, encoding="utf-8-sig")

    decay = pd.DataFrame(
        {lag: {m: corr_pair(train[f"{m}_lag_{lag}h"], y)[0] for m in metrics} for lag in range(1, 10)})
    decay.columns.name = "lag_h"
    decay.round(3).to_csv(TAB / "lag_correlation_decay.csv", encoding="utf-8-sig")

    # ---------- §4 D：冗餘檢查 ----------
    vif_lag1, n_lag1 = vif(lag1)
    vif_pm25, n_pm25 = vif(train[[f"PM2.5_lag_{lag}h" for lag in range(1, 10)]])
    pd.concat([
        vif_lag1.to_frame().assign(set=f"18 metrics @ lag 1h (n={n_lag1})"),
        vif_pm25.to_frame().assign(set=f"PM2.5 lag 1–9h (n={n_pm25})"),
    ]).round(2).to_csv(TAB / "vif.csv", encoding="utf-8-sig")

    # ---------- 圖 ----------
    # Fig 1：PM2.5(t−1) vs PM2.5(t)
    x = train["PM2.5_lag_1h"]
    pearson, spearman, n = corr_pair(x, y)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.scatter(x, y, s=6, alpha=0.25, color=BLUE, linewidths=0)
    top = np.nanmax([x.max(), y.max()])
    ax.plot([0, top], [0, top], ls="--", lw=1, color=MUTED)
    ax.text(0.04, 0.96, f"Pearson r = {pearson:.3f}\nSpearman ρ = {spearman:.3f}\nn = {n}",
            transform=ax.transAxes, va="top", color=INK2)
    ax.set(xlabel="PM2.5 at t−1 (μg/m³)", ylabel="PM2.5 at t (μg/m³)", aspect="equal")
    save(fig, "fig1_pm25_lag1_scatter")

    # Fig 2：相關係數隨 lag 衰減（取 lag 1h |r| 最大的 6 個非角度測項）
    top6 = decay.drop(index=ANGLES)[1].abs().nlargest(6).index
    fig, ax = plt.subplots(figsize=(5, 3.4))
    for color, metric in zip(SERIES, top6):
        ax.plot(decay.columns, decay.loc[metric], color=color, marker="o", ms=4, label=metric)
    ax.axhline(0, color="#c3c2b7", lw=1)
    ax.set(xlabel="Lag (hours before target)", ylabel="Pearson r with PM2.5 at t", xticks=range(1, 10))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    save(fig, "fig2_lag_correlation_decay")

    # Fig 3：依小時、月份的 PM2.5 分布
    box_style = dict(patch_artist=True, showfliers=False, widths=0.6,
                     boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE),
                     medianprops=dict(color="#184f95", lw=1.5),
                     whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3), gridspec_kw={"width_ratios": [2.2, 1]})
    hours = train["target_datetime"].dt.hour
    ax1.boxplot([y[hours == h] for h in range(24)], tick_labels=range(24), **box_style)
    ax1.set(xlabel="Hour of day", ylabel="PM2.5 (μg/m³)")
    months = train["target_datetime"].dt.month
    month_ids = sorted(months.unique())
    ax2.boxplot([y[months == m] for m in month_ids], tick_labels=month_ids, **box_style)
    ax2.set(xlabel="Month (2025)", ylabel="PM2.5 (μg/m³)")
    save(fig, "fig3_pm25_by_hour_month")

    # Fig 4：風向分區、風速分段的 PM2.5 分布
    sectors = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    direction = train["WIND_DIREC_lag_1h"]
    sector = ((direction + 22.5) // 45 % 8).map(dict(enumerate(sectors)))
    speed_bins = [0, 1, 2, 3, 4, 6, np.inf]
    speed_labels = ["0–1", "1–2", "2–3", "3–4", "4–6", "≥6"]
    speed = pd.cut(train["WIND_SPEED_lag_1h"], speed_bins, labels=speed_labels, right=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3))
    ax1.boxplot([y[sector == s] for s in sectors], tick_labels=sectors, **box_style)
    ax1.set(xlabel="Wind direction at t−1 (sector)", ylabel="PM2.5 at t (μg/m³)")
    ax2.boxplot([y[speed == s] for s in speed_labels], tick_labels=speed_labels, **box_style)
    ax2.set(xlabel="Wind speed at t−1 (m/s)", ylabel="PM2.5 at t (μg/m³)")
    save(fig, "fig4_pm25_by_wind")

    # Fig 5：lag 1h 特徵間的相關矩陣
    corr = lag1.corr()
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(corr, cmap=DIVERGING, vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.grid(False)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iat[i, j]:.2f}", ha="center", va="center", fontsize=5,
                    color="white" if abs(corr.iat[i, j]) > 0.7 else INK)
    fig.colorbar(image, ax=ax, fraction=0.046, label="Pearson r")
    save(fig, "fig5_feature_correlation_heatmap")

    # ---------- 摘要 ----------
    print("\n目標範圍（μg/m³）：\n" + target_range.to_string())
    print("\n與目標的關聯（lag 1h，依 |Spearman| 排序）：\n" + ranking.to_string(index=False))
    print(f"\nVIF（18 測項 @ lag 1h，n={n_lag1}）：\n" + vif_lag1.sort_values(ascending=False).round(1).to_string())
    print(f"\nVIF（PM2.5 lag 1–9h，n={n_pm25}）：\n" + vif_pm25.round(1).to_string())
    print(f"\n輸出：{FIG}、{TAB}")


if __name__ == "__main__":
    main()
