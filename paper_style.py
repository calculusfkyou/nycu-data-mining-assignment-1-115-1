"""論文圖的共用樣式。

結構依 nature-figure 指南（https://github.com/Yuan1z0825/nature-skills）：8 pt 標籤、7 pt 刻度、0.8 pt 軸線、
粗體子圖標籤、虛線參考線、無圖框圖例、所有文字不小於 5 pt。
字型改用 Times，配合 ACL 模板「圖的字型盡量與內文一致」的要求。
由 EDA/paper_figure.py 與 Model/paper_figures.py 匯入。
"""

import matplotlib.pyplot as plt

# nature-figure PALETTE
BLUE_MAIN, BLUE_2, TEAL, VIOLET, RED = "#0F4D92", "#3775BA", "#42949E", "#9A4D8E", "#B64342"
REF_GRAY, LIGHT_GRAY = "#767676", "#A9A9A9"
INK, INK2 = "#1a1a1a", "#4d4d4d"

RC = {
    "font.family": "serif", "font.serif": ["Times New Roman", "STIXGeneral"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
    "xtick.major.size": 3, "ytick.major.size": 3,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": False, "legend.frameon": False,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2, "ytick.color": INK2, "text.color": INK,
    "lines.linewidth": 1.5, "savefig.dpi": 300, "pdf.fonttype": 42,
}


def apply():
    plt.rcParams.update(RC)


def panel_labels_row(fig, axes, labels):
    """粗體子圖標籤，在軸框左上角外側（−4 pt, +3 pt）；同一列取最高的軸框上緣，讓標籤對齊。"""
    fig.canvas.draw()  # 先套用 aspect 等版面設定，get_position 才是最終位置
    boxes = [ax.get_position() for ax in axes]
    top = max(b.y1 for b in boxes) + 3 / 72 / fig.get_figheight()
    for box, label in zip(boxes, labels):
        fig.text(box.x0 - 4 / 72 / fig.get_figwidth(), top, label, fontsize=8, weight="bold", va="bottom", ha="right")


def save(fig, path_without_suffix):
    for ext in ("pdf", "png"):
        fig.savefig(f"{path_without_suffix}.{ext}", facecolor="white", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
