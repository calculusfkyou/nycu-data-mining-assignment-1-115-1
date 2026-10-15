"""論文用的兩張圖，以論文版面的實際大小繪製。內容與 notebook 的 Fig 0、Fig 4 相同，只重新排版。

- fig7_paper_pipeline：pipeline 圖（全寬 6.3 吋），上泳道為訓練、下泳道為只評估一次的 test
- fig8_paper_coefficients：Set C 全部 34 個特徵的 β、95% HAC CI、p 值、VIF（單欄 3.15 吋）

只讀 tables/coefficients.csv（02_analysis.ipynb 的輸出），不重跑模型。樣式見 repo 根目錄的 paper_style.py。
執行：python Model/paper_figures.py
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from features import FIG, TAB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paper_style as ps  # noqa: E402

NAMES = {
    "z_PM2.5_lag_1h": "PM2.5 ($t-1$)", "z_PM10_lag_1h": "PM10 ($t-1$)", "z_PM10_lag_2h": "PM10 ($t-2$)",
    "z_PM10_lag_3h": "PM10 ($t-3$)", "z_CO_lag_1h": "CO ($t-1$)", "z_CO_lag_2h": "CO ($t-2$)",
    "z_CO_lag_3h": "CO ($t-3$)", "z_NO_lag_1h": "NO ($t-1$)", "z_NO2_lag_1h": "NO$_2$ ($t-1$)",
    "z_O3_lag_1h": "O$_3$ ($t-1$)", "z_SO2_lag_1h": "SO$_2$ ($t-1$)", "z_CH4_lag_1h": "CH$_4$ ($t-1$)",
    "z_NMHC_lag_1h": "NMHC ($t-1$)", "z_AMB_TEMP_lag_1h": "Temperature ($t-1$)", "z_RH_lag_1h": "RH ($t-1$)",
    "z_RAINFALL_lag_1h": "Rainfall ($t-1$)", "z_WIND_SPEED_lag_1h": "Wind speed ($t-1$)",
    "eng_pm25_mean3h": "PM2.5 3-h mean", "eng_pm25_mean6h": "PM2.5 6-h mean", "eng_pm25_mean9h": "PM2.5 9-h mean",
    "eng_pm25_trend": "PM2.5 trend", "eng_pm25_std9h": "PM2.5 9-h SD", "eng_rain_flag": "Rain flag",
    "eng_wd_sin": "Wind dir. sin", "eng_wd_cos": "Wind dir. cos", "eng_wind_u": "Wind $u$", "eng_wind_v": "Wind $v$",
    "eng_ws_lt1": "Wind < 1 m/s", "eng_ws_ge6": "Wind ≥ 6 m/s", "eng_pm25_x_ws": "PM2.5 × wind speed",
    "eng_hour_sin": "Hour sin", "eng_hour_cos": "Hour cos", "eng_weekend": "Weekend",
    "eng_pm25_lag1_missing": "PM2.5 imputed flag",
}

# 每類階段一個顏色（邊框, 淡底）：資料、轉換、模型（主角）、評估
KINDS = {
    "data": (ps.REF_GRAY, "#F2F2F2"),
    "transform": (ps.BLUE_MAIN, "#E6EDF6"),
    "model": (ps.VIOLET, "#F3EAF2"),
    "eval": (ps.TEAL, "#E5F1F2"),
}


def _box(ax, x, y, w, h, kind, title, body, dashed=False):
    edge, face = KINDS[kind]
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.05", facecolor=face,
                                edgecolor=edge, lw=0.9, ls=(0, (3, 2)) if dashed else "-"))
    ax.text(x + w / 2, y + h - 0.09, title, ha="center", va="top", fontsize=7.5, weight="bold")
    ax.text(x + w / 2, y + h - 0.27, body, ha="center", va="top", fontsize=6.3, color=ps.INK2, linespacing=1.2)


def _arrow(ax, start, end, label=None, label_xy=None, ha="center"):
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(
        arrowstyle="-|>", color=ps.INK2, lw=0.8, mutation_scale=7, shrinkA=0, shrinkB=0))
    if label:
        ax.text(*label_xy, label, ha=ha, va="center", fontsize=7, style="italic", color=ps.REF_GRAY)


def pipeline():
    fig, ax = plt.subplots(figsize=(6.3, 1.55))
    ax.set(xlim=(0, 6.3), ylim=(0, 1.55))
    ax.axis("off")
    h, top, bottom = 0.62, 0.86, 0.07  # 兩條泳道：上 = 訓練，下 = 測試

    _box(ax, 0.0, 0.46, 1.12, h, "data", "Raw data", "Hsinchu station, 2025\n18 variables × 8,760 h")
    _box(ax, 1.26, 0.46, 1.14, h, "transform", "Lag windows", "lags $t-1,\\dots,t-9$\n162 features, 8,663 rows")
    _box(ax, 2.66, top, 1.08, h, "transform", "Preprocessing", "mean imputation and\nz-score, train statistics")
    _box(ax, 3.9, top, 1.06, h, "transform", "Set C features", "34 features (Table 4)\naggregates, wind bins")
    _box(ax, 5.12, top, 1.17, h, "model", "Ridge regression", "$\\alpha = 31.6$ by 5-fold\nforward-chaining CV")
    _box(ax, 2.66, bottom, 2.3, h, "data", "Held-out test set",
         "19 Oct–31 Dec, 1,733 rows\ntransformed with training statistics only", dashed=True)
    _box(ax, 5.12, bottom, 1.17, h, "eval", "Evaluate once", "MAE, RMSE, $R^2$\nvs. 3 µg/m³ threshold")

    mid = 0.46 + h / 2
    _arrow(ax, (1.12, mid), (1.26, mid))
    _arrow(ax, (2.4, mid + 0.08), (2.66, top + h / 2), "80% train", (2.39, 1.3), ha="right")
    _arrow(ax, (2.4, mid - 0.08), (2.66, bottom + h / 2), "20% test", (2.39, 0.25), ha="right")
    _arrow(ax, (3.74, top + h / 2), (3.9, top + h / 2))
    _arrow(ax, (4.96, top + h / 2), (5.12, top + h / 2))
    _arrow(ax, (4.96, bottom + h / 2), (5.12, bottom + h / 2))
    _arrow(ax, (5.705, top), (5.705, bottom + h), "refit on full training set", (5.64, 0.775), ha="right")
    ps.save(fig, FIG / "fig7_paper_pipeline")


def coefficients():
    coef = pd.read_csv(TAB / "coefficients.csv", encoding="utf-8-sig").iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(3.15, 4.2))
    for i, row in coef.iterrows():
        color = ps.RED if row["vif"] >= 10 else ps.BLUE_MAIN
        ax.errorbar(row["beta_ols"], i, xerr=[[row["beta_ols"] - row["ci_low"]], [row["ci_high"] - row["beta_ols"]]],
                    fmt="o", ms=3.5, color=color, elinewidth=1.2, capsize=0)
    ax.axvline(0, color=ps.REF_GRAY, lw=1.0, ls="--", zorder=0)
    ax.set_yticks(range(len(coef)), [NAMES[f] for f in coef["feature"]], fontsize=6.5)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.7, len(coef) - 0.3)
    ax.set_xlabel("$\\beta$ (µg/m³ per SD), 95% HAC CI")

    # 右側兩欄：p 值與 VIF（VIF ≥ 10 標紅，與點的顏色一致）
    for i, row in coef.iterrows():
        p = "<.001" if row["p_value"] < 0.001 else f"{row['p_value']:.3f}".lstrip("0")
        vif = f"{row['vif']:.2f}" if 9.95 <= row["vif"] < 10 else f"{row['vif']:.1f}"  # 9.999 不要顯示成 10.0
        ax.text(1.04, i, p, transform=ax.get_yaxis_transform(), fontsize=6.5, va="center", color=ps.INK2)
        ax.text(1.27, i, vif, transform=ax.get_yaxis_transform(), fontsize=6.5, va="center",
                color=ps.RED if row["vif"] >= 10 else ps.INK2)
    header = len(coef) - 0.2
    ax.text(1.04, header, "$p$", transform=ax.get_yaxis_transform(), fontsize=7, va="bottom", weight="bold")
    ax.text(1.27, header, "VIF", transform=ax.get_yaxis_transform(), fontsize=7, va="bottom", weight="bold")
    fig.tight_layout(pad=0.4)
    fig.subplots_adjust(right=0.75)
    ps.save(fig, FIG / "fig8_paper_coefficients")


if __name__ == "__main__":
    ps.apply()
    pipeline()
    coefficients()
