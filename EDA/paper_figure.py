"""論文 Figure 1：把 eda.py 的 Fig 1、Fig 2、Fig 4（風速）合成一張，以論文全寬（6.3 吋）的實際大小繪製。

只用 train。沿用 eda.py 的分段與相關係數計算；樣式見 repo 根目錄的 paper_style.py。
執行：python EDA/paper_figure.py
輸出：EDA/figures/fig6_paper_eda.png|pdf
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eda import DATA, FIG, corr_pair

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import paper_style as ps  # noqa: E402

# lag 衰減圖：PM2.5 為主角，其餘依相關強度由深到淺
LAG_COLORS = [ps.BLUE_MAIN, ps.BLUE_2, ps.TEAL, ps.VIOLET, ps.REF_GRAY, ps.LIGHT_GRAY]


def main():
    ps.apply()
    df = pd.read_csv(DATA / "All_Metrics" / "pure_all.csv", encoding="utf-8-sig", low_memory=False)
    train = df[df["dataset_split"] == "train"]
    y = train["target_PM2.5"]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(6.3, 2.15), gridspec_kw={"width_ratios": [1, 1.35, 1.2]})

    # a：PM2.5(t−1) vs PM2.5(t)
    x = train["PM2.5_lag_1h"]
    pearson, spearman, _ = corr_pair(x, y)
    ax1.scatter(x, y, s=2, alpha=0.2, color=ps.BLUE_MAIN, linewidths=0, rasterized=True)
    top = np.nanmax([x.max(), y.max()])
    ax1.plot([0, top], [0, top], ls="--", lw=1.0, color=ps.REF_GRAY)
    ax1.text(0.05, 0.97, f"$r$ = {pearson:.2f}\n$\\rho$ = {spearman:.2f}", transform=ax1.transAxes, va="top",
             fontsize=7)
    ax1.set(xlabel="PM2.5 at $t-1$ (µg/m³)", ylabel="PM2.5 at $t$ (µg/m³)", aspect="equal")

    # b：相關係數隨 lag 衰減（lag 1h |r| 最大的 6 個非角度測項）
    metrics = [c[: -len("_lag_1h")] for c in df.columns if c.endswith("_lag_1h")]
    metrics = [m for m in metrics if m not in ("WIND_DIREC", "WD_HR")]
    decay = pd.DataFrame({lag: {m: corr_pair(train[f"{m}_lag_{lag}h"], y)[0] for m in metrics} for lag in range(1, 10)})
    for color, metric in zip(LAG_COLORS, decay[1].abs().nlargest(6).index):
        ax2.plot(decay.columns, decay.loc[metric], color=color, marker="o", ms=3, lw=1.3,
                 label=metric.replace("NO2", "NO$_2$"))
    ax2.set(xlabel="Lag (hours before target)", ylabel="Pearson $r$ with PM2.5 at $t$", xticks=range(1, 10),
            ylim=(0, 1.05))
    ax2.legend(ncol=3, loc="upper right", handlelength=1.2, columnspacing=0.8, borderaxespad=0.1)

    # c：依 t−1 風速分段的 PM2.5 分布
    speed_bins = [0, 1, 2, 3, 4, 6, np.inf]
    speed_labels = ["0–1", "1–2", "2–3", "3–4", "4–6", "≥6"]
    speed = pd.cut(train["WIND_SPEED_lag_1h"], speed_bins, labels=speed_labels, right=False)
    ax3.boxplot([y[speed == s] for s in speed_labels], tick_labels=speed_labels, patch_artist=True,
                showfliers=False, widths=0.6, boxprops=dict(facecolor="#DCE6F1", edgecolor=ps.BLUE_MAIN, lw=0.8),
                medianprops=dict(color=ps.BLUE_MAIN, lw=1.5), whiskerprops=dict(color=ps.BLUE_MAIN, lw=0.8),
                capprops=dict(color=ps.BLUE_MAIN, lw=0.8))
    ax3.set(xlabel="Wind speed at $t-1$ (m/s)", ylabel="PM2.5 at $t$ (µg/m³)")

    fig.tight_layout(pad=0.4, w_pad=1.5)
    ps.panel_labels_row(fig, [ax1, ax2, ax3], "abc")
    ps.save(fig, FIG / "fig6_paper_eda")


if __name__ == "__main__":
    main()
