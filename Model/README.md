# Model：線性迴歸訓練與結果分析

對應作業 §5 Method、§6 Experimental Setup、§7 Results、§8 Analysis（作業 PDF p.10–17）。

任務和 EDA 相同：用過去 9 小時（lag 1–9h）的 18 個測項，預測新竹測站下一小時的 PM2.5（μg/m³）。
最終模型必須是線性迴歸，其他模型只當 baseline。

## 資料夾結構

```
Model/
├── README.md             本說明
├── features.py           共用模組：讀檔、特徵工程、特徵組、評估工具
├── 01_training.ipynb     §5–§7：特徵、模型選擇、CV、baseline、ablation、test 評估一次
├── 02_analysis.ipynb     §7–§8：顯著性、切片、係數、假設檢驗、錯誤分析
├── figures/              執行後產生，.png 與 .pdf
├── tables/               執行後產生，.csv（UTF-8 with BOM，Excel 可直接開）
└── outputs/              執行後產生：folds、CV 結果、test 預測、最終模型
```

## 執行方式

需要 Python 3.10 以上，以及 matplotlib 3.9 以上（和 EDA 相同）。

```bash
pip install -r requirements.txt
```

用 Jupyter 依序執行 `01_training.ipynb`、`02_analysis.ipynb`。從 repo 根目錄或 `Model/` 開啟都可以。
02 會讀 01 存在 `outputs/` 的結果，所以 01 要先跑完。

## 資料：hybrid 做法

| 用途 | 來源 | 處理 |
|---|---|---|
| 目標 `y` | `pure_all.csv` 的 `target_PM2.5` | 原始單位 μg/m³，不轉換 |
| 原始欄位（`z_` 開頭） | `standardized_all.csv` | 已用 train 平均數補值、train 統計量標準化，直接使用 |
| 衍生特徵（`eng_` 開頭） | 從 `pure_all.csv` 原始值計算 | Pipeline 內 median 補值 + StandardScaler，每個 fold 各自 fit |
| 原始值（`r_` 開頭） | `pure_all.csv` | 給 persistence、HistGradientBoosting 與穩健性檢查使用 |

這個做法要注意 3 件事：

1. `standardized_all.csv` 裡的 `WIND_DIREC`、`WD_HR` 是把 0–360 度的角度直接標準化，沒有意義。風向的 sin/cos 從 pure 計算，這兩欄只出現在 Set A。
2. 滾動平均、交互作用、分段這些衍生特徵需要原始單位，所以都從 pure 計算。
3. standardized 檔的 scaler 是用整個 train 期間 fit 的，CV 的驗證 fold 也參與了統計量的估計（不影響 test）。
   01 的第 9 步用「全部從 pure、fold 內標準化」重跑一次，量化這個影響，結果在 `tables/robustness_in_fold_scaling.csv`。

`load_hybrid()` 會 assert 兩檔逐列對齊；01 的第 2 步再檢查 standardized 是 pure 的仿射轉換，且只用 train 統計量。

## 特徵組（`features.GROUPS`）

| 特徵群 | 內容 | 依據（EDA） |
|---|---|---|
| persistence | PM2.5 lag 1h | r = 0.88 |
| pm25_aggregates | 趨勢 lag1 − lag2、3/6/9h 滾動平均、9h 標準差（取代 lag 2–9h，避免和 lag1 完全共線） | lag 9h 仍有 r = 0.55 |
| co_pollutants | PM10、CO lag 1–3h；NO、NO2、O3、SO2、CH4、NMHC lag 1h | PM10 r = 0.78、CO 0.60 |
| meteorology | 溫度、濕度、雨量 lag 1h，有雨旗標 | 溫度 r ≈ −0.33 |
| wind | 風速、風向 sin/cos、風向量 u/v | 控制特徵：Fig 4 風向分組有差異，但線性相關 ≈ 0；角度要轉換 |
| wind_bins | 風速 < 1、≥ 6 m/s 旗標 | Fig 4 非線性 |
| interaction | PM2.5(t−1) × 風速(t−1) | 風大時擴散強 |
| calendar | 目標小時 sin/cos、週末；不用月份 | 控制特徵：Fig 3 各小時中位數 9–12，差異小；週末 EDA 未分析；train 沒有 11–12 月，所以不用月份 |
| missing_flag | PM2.5 lag 1h 是否為補值 | 控制特徵：standardized 用平均數補值，旗標讓模型區分補值列 |

wind、calendar、missing_flag 三群沒有對應的假設，是**控制特徵**：EDA 看得出可能有影響，但證據不足以寫成可檢驗的假設。
放進模型是為了控制這些因素，不做方向上的預測。逐群移除的結果（ΔMAE：wind −0.009、calendar +0.001、missing_flag −0.003）也顯示它們的貢獻接近 0。
風的非線性效果另外由 wind_bins、interaction 兩群檢驗（H3）。

刪除共線欄位（刪除前的 VIF 來自 `EDA/tables/vif.csv`，18 個測項 @ lag 1h；刪除後的 VIF 來自 `tables/vif_set_c.csv`）：

| 刪除 | 原因 | 刪除前 VIF | 刪除後（Set C） |
|---|---|---|---|
| NOx | NOx = NO + NO2，完全共線 | NOx 19,367、NO2 11,950、NO 2,718 | NO 1.98、NO2 4.76 |
| THC | THC = CH4 + NMHC，完全共線 | THC 13,707、CH4 7,671、NMHC 1,765 | CH4 2.89、NMHC 4.71 |
| WS_HR | 和 WIND_SPEED r = 0.94（EDA Fig 5） | 8.69（不高，但內容幾乎相同） | WIND_SPEED 4.72 |
| WD_HR | sin/cos 和 WIND_DIREC r ≈ 0.9（EDA Fig 5） | sin 5.73、cos 5.39 | 風向 sin 4.96、cos 4.46 |

完全共線的組合刪掉加總欄，保留可以分開解讀的成分。WS_HR、WD_HR 的 VIF 沒有超過 10，刪除的依據是兩兩相關高、內容重複。

Ablation 用的另外兩組：Set A = 全部 162 個原始欄位；Set B = Set A 中 |Spearman ρ| 前 k 名，在每個訓練 fold 內重選，k ∈ {5, 10, 20}。

## 驗證流程

- test：依時間的後 20%（2025-10-19 21:00 至 12-31 23:00，1733 列），只在 01 的最後一步評估一次。
- CV：train 內 `TimeSeriesSplit(n_splits=5, gap=9)`，所有方法共用同一組 folds（`outputs/cv_folds.csv`）。
- 指標：MAE、RMSE、R²（μg/m³）。決策門檻 MAE ≤ 3 μg/m³（AQI「普通」級寬約 20 μg/m³）。
- Baseline：training mean、persistence、單一特徵 OLS、HistGradientBoosting（原始欄位）。
- 線性模型：OLS、Ridge、Lasso、ElasticNet，以 CV MAE 選擇。

## 輸出與論文段落對照

| 輸出 | 內容 | 論文段落 |
|---|---|---|
| `figures/fig0_pipeline` | Pipeline 圖 | §5.1 |
| `tables/vif_set_c.csv` | Set C 的 VIF | §5.2 |
| `tables/model_selection.csv`、`figures/fig1_tuning_curves` | 搜尋範圍、選定值、調參曲線 | §5.2 |
| `tables/cv_folds_summary.csv` | 每個 fold 的期間與列數 | §6.1 |
| `tables/main_results.csv`、`figures/fig2_cv_comparison` | 所有方法的 CV（mean ± std）與 test 結果 | §7.1 |
| `tables/significance.csv` | paired t-test、block bootstrap 95% CI | §7.1 |
| `tables/ablation.csv`、`tables/setB_selected_features.csv` | Set A / B / C 與逐群移除 | §7.2 |
| `tables/slice_mae.csv`、`figures/fig3_slice_mae` | 切片 MAE | §7.2 |
| `tables/coefficients.csv`、`figures/fig4_coefficients_forest` | β、95% CI、p 值（HAC）、VIF | §8.1 |
| `tables/hypotheses.csv` | H1–H5 的證據與初步判定 | §8.1 |
| `tables/largest_errors.csv`、`figures/fig5_residuals` | 最大誤差 5 筆、殘差圖 | §8.2 |
| `tables/robustness_in_fold_scaling.csv` | hybrid 標準化的影響 | §9 |

## 實驗結果

所有誤差單位為 μg/m³。CV 為 5 個 fold 的 mean ± std。差距一律是「最終模型 − 對照」，負值代表最終模型較好。

### 1. 模型選擇（`model_selection.csv`、Fig 1）

| 變體 | 選定的超參數 | CV MAE |
|---|---|---|
| OLS | — | 2.777 ± 0.458 |
| **Ridge** | α = 31.6 | **2.751 ± 0.418** |
| Lasso | α = 0.022 | 2.756 ± 0.416 |
| ElasticNet | α = 0.022，l1_ratio = 0.5 | 2.752 ± 0.420 |

- 4 種變體的差距都在 0.03 以內，遠小於 fold 間的標準差，選哪一種影響不大。
- 選 Ridge 的理由是共線：Set C 中 PM2.5 lag1 與 3/6/9h 平均的 VIF 為 28–66（`vif_set_c.csv`），Ridge 會收縮這些不穩定的係數。

### 2. 主要結果（`main_results.csv`、`significance.csv`、Fig 2）

| 方法 | CV MAE | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|
| Training mean | 7.36 ± 1.44 | 5.96 | 7.18 | −0.17 |
| Persistence (t−1) | 2.91 ± 0.40 | 2.55 | 3.57 | 0.71 |
| 單一特徵 OLS（PM2.5 t−1） | 2.99 ± 0.32 | 2.59 | 3.52 | 0.72 |
| HistGradientBoosting（原始欄位） | 3.02 ± 0.63 | 2.53 | 3.29 | 0.75 |
| **Ridge + Set C（最終）** | **2.75 ± 0.47** | 2.48 | 3.26 | 0.76 |

- 除了 training mean，所有方法的 test MAE 都低於 3 μg/m³ 的門檻，**連 persistence 也達標**。所以這個門檻無法區分方法好壞。
- CV 中最難的是 fold 1（2–4 月，春季高濃度），最終模型在這個 fold 的 MAE 是 3.43，超過門檻。
- 顯著性：

| 對照 | CV paired t-test | Test block bootstrap 95% CI | 結論 |
|---|---|---|---|
| Persistence | −0.16，p = 0.066 | [−0.19, +0.05] | 不顯著 |
| HistGradientBoosting | −0.27，p = 0.042 | [−0.13, +0.04] | CV 顯著，test 不顯著 |
| 單一特徵 OLS | −0.24，p = 0.031 | [−0.20, −0.01] | 兩者都顯著 |

  和最強 baseline（persistence）相比，只能說是「小幅、在 CV 上一致的改善」，不能說明顯勝出。
- **限制：selection bias。** 線性模型的變體與超參數是在同一組 5 個 fold 上選出的，報告的就是選擇時用的 CV 分數，所以調參過的模型（最終模型、Set A / B）CV 分數偏樂觀。
  training mean、persistence、單一特徵 OLS 沒有超參數，HistGradientBoosting 用預設參數，都不受影響。
  因此上表的 CV 配對檢定（特別是對 persistence 的 p = 0.066）也偏樂觀，最終結論以只評估一次的 test 為準。要消除這個偏差需要改用 nested CV。
- 最終模型在 test 上贏過非線性的 HistGradientBoosting（差距不顯著）。HistGradientBoosting 的 CV 標準差最大（0.63），在樣本少的早期 fold 表現不穩定。

### 3. 特徵 Ablation（`ablation.csv`）

| 特徵組 | 特徵數 | CV MAE | Test MAE |
|---|---|---|---|
| Set A：全部原始欄位 | 162 | 2.871 ± 0.555 | **2.435** |
| Set B：in-fold top-k（k = 20） | 20 | 2.754 ± 0.398 | 2.472 |
| Set C：ours | 34 | **2.751 ± 0.467** | 2.476 |

- **CV 上 Set C 優於 Set A（−0.12，p = 0.055），但 test 上 Set A 反而顯著較好**（C − A = +0.04，95% CI [+0.002, +0.083]）。
  CV 中有幫助的特徵設計，沒有推廣到冬季的 test。
- **Set B 和 Set C 幾乎一樣**（CV 差 0.003，p = 0.95）。Set B 選出的 20 個特徵全是 PM2.5、PM10、CO 的 lag，
  代表只靠相關係數排序就能做到差不多的效果，衍生特徵在這之外能多加的資訊很少。
- 從 Set C 一次拿掉一個特徵群，CV MAE 的變化：

| 拿掉的特徵群 | ΔMAE |
|---|---|
| co_pollutants | **+0.161** |
| pm25_aggregates | +0.065 |
| persistence | +0.035 |
| meteorology | +0.026 |
| wind_bins / calendar / interaction | +0.004 / +0.001 / +0.000 |
| missing_flag / wind | −0.003 / −0.009 |

  真正有貢獻的是同源污染物（PM10、CO）和 PM2.5 的歷史資訊；風、日曆、交互作用幾乎沒有貢獻。

### 4. 資料切片（`slice_mae.csv`、Fig 3）

| 切片 | n | Persistence | Set C |
|---|---|---|---|
| 高濃度（≥ 24 μg/m³，train p90） | 65 | **3.91** | 4.61 |
| 早上 06–09 點 | 288 | **2.47** | 2.61 |
| 風速 < 1 m/s | 237 | 2.68 | 2.63 |
| 上一小時有雨 | 96 | 1.97 | **1.87** |
| 夜間 22–05 點 | 583 | 2.55 | **2.43** |

- 高濃度時段，線性模型會往平均值收縮，比直接沿用上一小時的值還差。對於需要警示的高污染時段，這是最主要的弱點。
- 風速 ≥ 6 m/s 只有 11 筆，無法下結論。

### 5. 係數與假設（`coefficients.csv`、`hypotheses.csv`、Fig 4）

係數單位為「每 1 個標準差對應的 μg/m³」，p 值來自 OLS t-test 搭配 HAC（Newey–West，24 lags）標準誤。
Ridge 的係數和 OLS 的差距都在 0.16 以內。方向只有兩個接近 0 的係數不同（NMHC、NO2，|β| < 0.02），其餘全部一致。

| 特徵 | β | 95% CI | VIF |
|---|---|---|---|
| PM2.5 lag 1h | +3.25 | [2.41, 4.08] | 43.7 |
| PM10 lag 1h | +2.79 | [2.23, 3.35] | 9.8 |
| PM2.5 3h 平均 | +2.23 | [1.11, 3.35] | 66.4 |
| CO lag 1h | +1.76 | [1.38, 2.14] | 10.0 |
| PM10 lag 3h | −0.86 | [−1.22, −0.49] | 10.1 |
| PM10 lag 2h | −0.75 | [−1.32, −0.18] | 14.2 |
| CO lag 2h | −0.61 | [−1.07, −0.15] | 11.8 |
| NO lag 1h | −0.47 | [−0.79, −0.16] | 2.0 |
| 溫度 lag 1h | −0.38 | [−0.54, −0.22] | 2.6 |
| 風速 < 1 m/s | +0.37 | [0.23, 0.51] | 2.2 |

統計顯著（p < 0.05）又有實際意義（|β| ≥ 0.5）的只有前 7 個，全部是 PM2.5、PM10、CO。

| 假設 | 判定 | 證據 |
|---|---|---|
| H1 PM2.5(t−1) 最重要 | 支持 | β = +3.25，所有係數中最大 |
| H2 PM10、CO 有額外資訊 | 支持 | lag1 為正（+2.79、+1.76），lag 2–3 為負：模型看的是「上升趨勢」，而非絕對濃度。這幾項 VIF 為 10–14，個別係數的解讀要小心 |
| H3 風速效果為負、非線性（靜風時 PM2.5 偏高） | 支持（僅低風速端） | < 1 m/s 顯著為正（+0.37，p < 0.001）：靜風時污染累積。≥ 6 m/s（+0.04，p = 0.42）與 PM2.5 × 風速（−0.13，p = 0.40）都不顯著，非線性只出現在低風速一端。拿掉 wind_bins 後 CV MAE 只增加 0.004，預測上的貢獻很小 |
| H4 溫度反映季節 | 支持 | 單變量斜率 −2.86，控制其他變數後只剩 −0.38 |
| H5 聚合特徵優於原始 lag | 無法判定 | 差距 +0.006，5 個 fold 配對 t-test p = 0.42 |

### 6. 錯誤分析（`largest_errors.csv`、Fig 5）

誤差 = 實際 − 預測，單位 μg/m³。

| # | 時間 | 實際 | 預測 | 誤差 | PM2.5 t−3 → t−1 | 可能原因 |
|---|---|---|---|---|---|---|
| 1 | 12-21 04:00 | 27 | 10.0 | +17.0 | 10 → 12 → 11 | 污染事件開始，歷史資料沒有前兆 |
| 2 | 10-23 07:00 | 19 | 2.1 | +16.9 | 0 → 0 → 0 | 儀器異常：連續 3 小時為 0 |
| 3 | 12-21 05:00 | 42 | 27.1 | +14.9 | 12 → 11 → 27 | 同 #1，濃度持續上升，預測落後 |
| 4 | 10-23 08:00 | 0 | 13.6 | −13.6 | 0 → 0 → 19 | 同 #2，讀值在 0 和 19 之間跳動 |
| 5 | 11-03 13:00 | 4 | 17.5 | −13.5 | 16 → 19 → 20 | 濃度突降，可能是風向或氣團轉換 |

- 5 筆都是「濃度突然變化」，模型只看過去的資料，所以預測落後。#2、#4 應歸類為量測雜訊，不是模型的問題。
- **系統性高估**：test 的平均殘差為 −0.78，10、11、12 月分別是 −0.80、−0.72、−0.83。
  test 期間的平均濃度（9.65）比 train（12.38）低，模型的截距學到的是 train 的水準。
- 殘差對預測值的圖中有斜條紋，因為 PM2.5 的原始讀值是整數。

### 7. 穩健性檢查（`robustness_in_fold_scaling.csv`）

把 hybrid 的標準化改成全部在 fold 內進行，CV MAE 平均增加 0.027。差距主要來自 fold 1（+0.127，只有 1146 列訓練資料），其他 fold 都小於 0.007。
相對於 fold 間的標準差（0.47），這個影響可以忽略。

### 結論與下一步

1. 線性模型可以把下一小時 PM2.5 的誤差壓到 2.5 μg/m³ 左右，但相對於 persistence 的改善很小，而且不顯著。
2. 有用的資訊幾乎都來自 PM2.5、PM10、CO 的近期歷史；風和時間特徵在線性模型中沒有貢獻。
3. 主要弱點是高濃度時段和濃度突變，以及 train/test 季節不同造成的系統性高估。
4. 值得嘗試的下一步：
   - 加入能提前反映變化的資訊，例如上風處測站、氣象預報。
   - 用跨年度資料涵蓋 11、12 月的季節。
   - 把決策門檻改成高濃度時段的誤差，讓門檻能區分方法好壞。
