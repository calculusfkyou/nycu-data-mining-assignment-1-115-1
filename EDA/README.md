# EDA：探索性特徵分析

對應作業 §4 Exploratory Feature Analysis（作業 PDF p.9），並同時產出 §3 需要的統計表（p.8）。

目標是用過去 9 小時（lag 1–9h）的 18 個測項，預測新竹測站下一小時的 PM2.5。
這個資料夾不訓練模型，只回答「哪些特徵有用、哪些重複、需要什麼轉換」。

## 資料夾結構

```
EDA/
├── README.md        本說明
├── eda.py           EDA 腳本，產出下面所有圖表
├── figures/         5 張圖，每張都有 .png 與 .pdf
│   ├── fig1_pm25_lag1_scatter
│   ├── fig2_lag_correlation_decay
│   ├── fig3_pm25_by_hour_month
│   ├── fig4_pm25_by_wind
│   └── fig5_feature_correlation_heatmap
└── tables/          6 張統計表（UTF-8 with BOM，Excel 可直接開）
    ├── target_range.csv            §3
    ├── summary_stats.csv           §3
    ├── missing_by_reason.csv       §3
    ├── correlation_ranking.csv     §4 C
    ├── lag_correlation_decay.csv   §4 C
    └── vif.csv                     §4 D
```

## 資料來源與範圍

| 項目 | 內容 |
|---|---|
| 監督式資料 | `Dataset/All_Metrics/pure_all.csv`（原始單位、未補值、未標準化） |
| 原始每小時資料 | `Dataset/新竹_2025.csv`，用 `Dataset/All_Metrics/preprocessing_all.py` 的 `read_measurements` 解析，缺值原因的定義與前處理一致 |
| 使用範圍 | **只用 train**：6930 列，目標時間 2025-01-01 09:00 至 2025-10-19 20:00 |
| 不使用 | test：1733 列，目標時間 2025-10-19 21:00 至 2025-12-31 23:00，腳本不會讀進任何統計 |

train / test 是依時間先後 80% / 20% 切分（見 `preprocessing_all.py`），EDA 只看 train，避免用到未來資訊。

## 執行方式

需要 Python 3.10 以上（`preprocessing_all.py` 用了 `float | None` 語法），以及 matplotlib 3.9 以上（`boxplot(tick_labels=...)`）。

```bash
uv run --no-project --with-requirements requirements.txt python EDA/eda.py
```

或在已安裝 pandas、numpy、matplotlib 的環境中執行：

```bash
python EDA/eda.py
```

每次執行都會覆寫 `figures/` 與 `tables/`，並在終端印出主要數字摘要。

## 統計表

### §3 資料描述

| 檔案 | 內容 | 欄位 |
|---|---|---|
| `target_range.csv` | 目標 `target_PM2.5` 的分布 | count、mean、std、min、1/5/25/50/75/95/99 百分位數、max |
| `summary_stats.csv` | 18 個測項在 train 期間的原始每小時統計 | unit、n_obs、n_missing、missing_%、mean、std（母體，ddof=0）、min、p50、max |
| `missing_by_reason.csv` | 各測項依缺值原因的次數交叉表 | 列為測項，欄為缺值原因，最後一列和最後一欄是合計 |

缺值原因代碼（來自原始資料的標記）：

| 原因 | 原始標記 | 意義 |
|---|---|---|
| `instrument_check_invalid` | `#` | 儀器檢核無效 |
| `program_check_invalid` | `*` | 程式檢核無效 |
| `manual_check_invalid` | `X` | 人工檢核無效 |
| `instrument_fault_alarm` | `A` | 儀器異常警報 |
| `other_invalid:NA` | `NA` | 其他無效標記 |

### §4 特徵分析

| 檔案 | 內容 |
|---|---|
| `correlation_ranking.csv` | 各特徵（lag 1h）與目標的 Pearson r、Spearman ρ 與有效樣本數 n，依 \|ρ\| 由大到小排序。風向已轉成 sin/cos |
| `lag_correlation_decay.csv` | 18 個測項在 lag 1–9h 與目標的 Pearson r（列：測項，欄：lag 小時數）。注意：`WIND_DIREC`、`WD_HR` 兩列是原始角度直接算的，數值沒有意義，判讀時應忽略 |
| `vif.csv` | 變異數膨脹因子（VIF）。分兩組：18 測項 @ lag 1h（風向轉 sin/cos 後共 20 欄，n = 6402），以及 PM2.5 lag 1–9h（n = 6497）。只用沒有缺值的完整列計算 |

相關係數一律採成對刪除缺值；Spearman ρ 以 rank 後的 Pearson 計算。

## 圖表

| 圖 | 內容 | 回答的問題 |
|---|---|---|
| Fig 1 `fig1_pm25_lag1_scatter` | PM2.5(t−1) 對 PM2.5(t) 散佈圖，虛線為 y = x | 最強的單一特徵是什麼？「沿用上一小時的值」這個 baseline 有多強？ |
| Fig 2 `fig2_lag_correlation_decay` | lag 1h 相關最強的 6 個測項，相關係數隨 lag 1–9h 的變化 | 需要多長的歷史窗口？ |
| Fig 3 `fig3_pm25_by_hour_month` | 依一天中的小時（0–23）與月份（1–10）的 PM2.5 箱形圖 | 時間特徵值不值得加？ |
| Fig 4 `fig4_pm25_by_wind` | 依 t−1 風向（8 方位）與風速（6 段）的 PM2.5 箱形圖 | 風的影響是不是非線性的？ |
| Fig 5 `fig5_feature_correlation_heatmap` | lag 1h 特徵兩兩之間的 Pearson r 熱圖（特徵對特徵，不含目標） | 哪些特徵彼此重複？ |

所有軸標籤都附單位。箱形圖不畫離群點（`showfliers=False`），鬚線為 1.5 IQR。

## 實驗結果

### 1. 目標分布（`target_range.csv`）

- PM2.5 範圍 0–85 μg/m³，平均 12.38、標準差 8.71、中位數 11。
- 分布右偏：第 95 百分位數 28、第 99 百分位數 41，最大值 85。少數高濃度時段會讓 MSE 類的指標偏大。

### 2. 缺值（`summary_stats.csv`、`missing_by_reason.csv`）

- 各測項缺值率都很低，介於 0.41%（風向、風速）到 3.45%（CH4、NMHC、THC）。
- train 期間共 2095 筆缺值，最主要的原因是儀器檢核無效（1267 筆），其次是程式檢核無效（450 筆）。
- 每個測項都有約 13 筆 `other_invalid:NA`，看起來是同一段時間整站停測。
- 以 lag 1h 的 18 個測項來看，6930 列中有 6402 列（約 92%）完全沒有缺值。

### 3. 與目標的關聯（`correlation_ranking.csv`、Fig 1）

| 排名 | 特徵（lag 1h） | Pearson r | Spearman ρ |
|---|---|---|---|
| 1 | PM2.5 | 0.880 | 0.859 |
| 2 | PM10 | 0.783 | 0.795 |
| 3 | CO | 0.601 | 0.622 |
| 4 | NO2 | 0.412 | 0.360 |
| 5 | AMB_TEMP | −0.328 | −0.349 |
| … | NMHC、NOx、THC、O3、CH4、SO2 | 0.22–0.41 | 0.26–0.32 |
| 末段 | RAINFALL、WIND_SPEED、WS_HR、RH、NO、風向 sin/cos | \|r\| < 0.1 | \|ρ\| < 0.16 |

- PM2.5(t−1) 是最強的特徵。Fig 1 的點緊貼 y = x，表示「直接沿用上一小時的值」就是很強的 baseline，模型至少要贏過它。
- PM10 和 CO 次之，和 PM2.5 來源相近（粒狀物、燃燒）。
- 氣象類特徵的線性相關都很弱，但不代表沒用，見第 5 點。

### 4. 相關係數隨 lag 的變化（`lag_correlation_decay.csv`、Fig 2）

- PM2.5 從 lag 1h 的 0.88 降到 lag 9h 的 0.55，PM10 走勢幾乎一樣（0.78 → 0.52）。CO、NO2、NMHC、THC、NOx 也是單調遞減。
- 到 lag 9h 時 PM2.5 還有 0.55，訊號還沒衰減完，但其中很多和近的 lag 重複（見第 6 點）。
- 例外：
  - **O3** 反而隨 lag 上升（0.26 → 0.31），可能和日夜光化學循環有關。
  - **AMB_TEMP** 各 lag 都維持在約 −0.33，幾乎不變，比較像是在反映季節，而不是小時尺度的變化。
  - **SO2** 先降後升（0.22 → 0.17 → 0.19）。

### 5. 時間與風（Fig 3、Fig 4）

- **小時**：各小時中位數都在 9–12 之間，一天內差異小，小時特徵的幫助有限。
- **月份**：季節差異明顯。3–4 月最高（中位數 16–18），6–7 月最低（約 6）。但 train 只涵蓋 1–10 月，test 在 10 月下旬到 12 月，**不宜直接把月份數字當特徵**，模型會遇到沒看過的值。
- **風向**：吹東南（SE）、東北（NE）風時略高，東（E）、南（S）風時略低。線性相關幾乎是 0，但分組後看得出差異。
- **風速**：0–1 m/s 時中位數約 12，1–6 m/s 約 10，≥ 6 m/s 約 7。差距不大，且 ≥ 6 m/s 那組樣本少，結論需保守看待。

### 6. 冗餘（`vif.csv`、Fig 5）

- **完全共線**：`NOx = NO + NO2`、`THC = CH4 + NMHC`。相關 VIF 約 1,800–19,000（NOx 19,367、NO2 11,950、THC 13,707、CH4 7,671、NO 2,718、NMHC 1,765）。
- **高度重複的配對**（Fig 5）：
  - CH4 與 THC：r = 0.96
  - NO2 與 NOx：r = 0.94
  - WIND_SPEED 與 WS_HR：r = 0.94
  - WIND_DIREC 與 WD_HR 的 sin/cos：r ≈ 0.9
  - PM10 與 PM2.5：r = 0.82
- 其餘測項 VIF 都在 10 以下（AMB_TEMP 2.1、CO 6.4、PM2.5 4.4、PM10 3.4、風速兩項約 8.7）。
- **PM2.5 lag 1–9h**：VIF 4.6–7.1，有中度共線，但沒有到必須刪除的程度。

## 對特徵選擇的啟示

### E. 特徵決策（已定稿，實作於 `Model/features.py` 的 `GROUPS`）

| 特徵 | 決策 | EDA 證據 | Model 中的特徵群 |
|---|---|---|---|
| PM2.5 lag 1h | 保留 | 與目標 r = 0.88，最強的單一特徵（Fig 1） | persistence |
| PM2.5 lag 2–9h | 轉換：改成趨勢（lag1 − lag2）、3/6/9h 平均、9h 標準差 | lag 9h 仍有 r = 0.55（Fig 2），但 lag 之間 VIF 4.6–7.1，資訊重複 | pm25_aggregates |
| PM10、CO | 保留 lag 1–3h | ρ = 0.80、0.62，僅次於 PM2.5 | co_pollutants |
| NO2、O3、SO2、CH4、NMHC | 保留 lag 1h | ρ = 0.26–0.36，有中度關聯 | co_pollutants |
| NO | 保留 lag 1h | ρ = −0.04，單獨幾乎無關；但丟掉 NOx 後，要和 NO2 一起保留才不會失去氮氧化物的資訊 | co_pollutants |
| NOx、THC | 丟掉 | 完全共線：NOx = NO + NO2、THC = CH4 + NMHC，VIF 19,367、13,707 | — |
| WS_HR、WD_HR | 丟掉 | 與 WIND_SPEED、WIND_DIREC 重複（r = 0.94、≈ 0.9） | — |
| WIND_SPEED | 保留，另加分段旗標（< 1、≥ 6 m/s） | 線性相關弱（ρ = −0.10），但 0–1 m/s 時中位數明顯較高（Fig 4） | wind、wind_bins |
| WIND_DIREC | 轉換：sin/cos，以及與風速相乘的風向量 u/v | 0–360 度不能直接用；分方位後有差異（Fig 4） | wind |
| PM2.5 × 風速 | 新增交互作用 | 風大時擴散較強，PM2.5 的持續性應該較弱 | interaction |
| AMB_TEMP、RH、RAINFALL | 保留 lag 1h，另加有雨旗標 | 溫度 ρ = −0.35，各 lag 幾乎不變，反映季節 | meteorology |
| 小時、週末 | 轉換：目標小時的 sin/cos，另加週末旗標 | 各小時中位數 9–12，差異小，但成本低（Fig 3）；週末在 EDA 未分析，建模時才加入（可能有交通差異） | calendar |
| 月份 | 不使用 | train 只有 1–10 月，test 是 10–12 月，會遇到沒看過的值（Fig 3） | — |
| `_missing` 旗標 | 只保留 PM2.5 lag 1h 的旗標 | 見下方說明 | missing_flag |
| `_missing_reason` | 丟掉 | 文字欄，不能直接當特徵；且缺值率低（見下方說明） | — |

**`_missing` 旗標的取捨理由：**

- 各測項缺值率只有 0.41%–3.45%（`summary_stats.csv`），lag 1h 的 18 個測項有 92% 的列完全沒有缺值。
- 162 個特徵各配一個旗標，特徵數會加倍，但絕大多數旗標幾乎都是 0。
- 只為 PM2.5 lag 1h 保留旗標：它是最重要的特徵，`standardized_all.csv` 用 train 平均補值，補值的列可能讓模型誤判，所以讓模型知道哪些列是補的。
- Model 的 ablation 顯示，拿掉這個旗標後 CV MAE 反而降低 0.003（`Model/tables/ablation.csv`），代表旗標本身幾乎沒有幫助。

### F. 假設（已定稿，在 `Model/02_analysis.ipynb` 檢驗）

| 假設 | EDA 依據 | 檢驗結果（`Model/tables/hypotheses.csv`） |
|---|---|---|
| H1 PM2.5(t−1) 是最主要的預測因子 | r = 0.88，所有特徵中最高（Fig 1） | 支持：β = +3.25，所有係數中最大 |
| H2 控制 PM2.5 後，PM10、CO 仍有額外資訊 | ρ = 0.80、0.62，且和 PM2.5 的 r = 0.82，不完全重複（Fig 5） | 支持：PM10、CO lag 1h 為正（+2.79、+1.76） |
| H3 風速效果為負且非線性，靜風時 PM2.5 偏高 | 0–1 m/s 中位數約 12，1–6 m/s 約 10（Fig 4） | 支持（僅低風速端）：< 1 m/s 旗標 β = +0.37；≥ 6 m/s 不顯著 |
| H4 溫度的負相關主要反映季節 | ρ = −0.35，各 lag 幾乎不變；月份差異明顯（Fig 2、Fig 3） | 支持：單變量斜率 −2.86，控制其他變數後只剩 −0.38 |
| H5 PM2.5 聚合特徵優於原始 lag 2–9h | lag 9h 仍有 r = 0.55，但 lag 之間 VIF 4.6–7.1（Fig 2） | 無法判定：CV MAE 差 +0.006，paired t-test p = 0.42 |

H5 是建模階段新增的假設，EDA 依據為 Fig 2 的 lag 衰減與 PM2.5 lag 之間的 VIF。

## 限制與待辦

- 只分析線性（Pearson）與單調（Spearman）關聯，尚未計算 mutual information（§4 C 的可選項目）。
- 只分析了 2025 年單一測站，季節結論無法驗證是否跨年成立。
- train 沒有 11、12 月的資料，對 test 期間的季節行為無從觀察。
- `lag_correlation_decay.csv` 中原始角度的兩列應忽略或改成 sin/cos。
