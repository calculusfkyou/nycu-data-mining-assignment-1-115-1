"""
新竹 2025 空氣品質資料 preprocessing
------------------------------------
流程：
1. 讀取寬表格 (日期 x 測項，00~23 為小時欄位)，轉成「每小時一列、每個測項一欄」的長時間序列
2. 無效值標記 (#, *, x, A, 空白) 一律轉成 NaN
3. 補齊完整的每小時時間索引
4. 缺失值處理：
   - 連續缺失 <= MAX_GAP 小時：線性插值 (風向用圓周插值)
   - 連續缺失 >  MAX_GAP 小時：保留 NaN (後續切窗時請丟掉包含它們的樣本)
   - 資料開頭 / 結尾的缺失：保留 NaN (不外插)
5. 輸出處理後的 CSV

用法：
    python preprocess.py
    python preprocess.py --input 新竹_2025.csv --output hsinchu_2025_clean.csv --max-gap 3
"""

import argparse

import numpy as np
import pandas as pd

HOUR_COLS = [f"{h:02d}" for h in range(24)]
ANGLE_ITEMS = ["WIND_DIREC", "WD_HR"]  # 角度 (0~360)，不能直接線性插值


def load_wide_csv(path: str) -> pd.DataFrame:
    """讀取原始寬表格 CSV，統一欄位名稱並移除多餘的空欄位。"""
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
    df.columns = [str(c).strip() for c in df.columns]
    # 丟掉沒有名稱的欄位 (檔案尾端的空白欄)
    df = df.loc[:, [c for c in df.columns if c != "" and not c.startswith("Unnamed")]]
    df["測項"] = df["測項"].str.strip()
    return df


def wide_to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """寬表格 -> index 為時間、欄位為測項的每小時資料表，無效值轉為 NaN。"""
    long = df.melt(
        id_vars=["測站", "日期", "測項"],
        value_vars=HOUR_COLS,
        var_name="hour",
        value_name="value",
    )

    # 時間戳記：日期 + 小時
    date = pd.to_datetime(long["日期"].str.strip(), format="%Y/%m/%d %H:%M:%S")
    long["datetime"] = date + pd.to_timedelta(long["hour"].astype(int), unit="h")

    value = long["value"].astype(str).str.strip()

    # RAINFALL 的 "NR" 表示無降雨，視為 0 而不是缺值
    is_rain = long["測項"] == "RAINFALL"
    value = value.mask(is_rain & value.str.upper().eq("NR"), "0")

    # 非數值 (含 #, *, x, A, 空白) 全部轉成 NaN。
    # 若儀器旗標是接在數字後面 (例如 "12.3#")，同樣會被視為無效值而轉成 NaN。
    long["value"] = pd.to_numeric(value, errors="coerce")

    hourly = long.pivot_table(
        index="datetime", columns="測項", values="value", aggfunc="first"
    )
    hourly = hourly.sort_index()

    # 補齊完整的每小時時間索引 (避免缺漏的時間點直接不見)
    full_index = pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")
    hourly = hourly.reindex(full_index)
    hourly.index.name = "datetime"
    return hourly


def _nan_run_lengths(is_nan: pd.Series) -> pd.Series:
    """回傳每個位置所屬的「連續 NaN 段長度」(非 NaN 位置為 0)。"""
    group_id = (is_nan != is_nan.shift()).cumsum()
    run_len = is_nan.groupby(group_id).transform("sum")
    return run_len.where(is_nan, 0)


def fill_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    """線性插值，但只補連續缺失長度 <= max_gap 的區段，且不外插頭尾。"""
    run_len = _nan_run_lengths(series.isna())
    filled = series.interpolate(method="linear", limit_area="inside")
    too_long = (run_len > max_gap) & series.isna()
    filled[too_long] = np.nan
    return filled


def fill_short_gaps_angle(series: pd.Series, max_gap: int) -> pd.Series:
    """角度資料用 sin/cos 分量插值後再轉回 0~360，避免 359° 與 1° 被插成 180°。"""
    rad = np.deg2rad(series)
    sin_f = fill_short_gaps(np.sin(rad), max_gap)
    cos_f = fill_short_gaps(np.cos(rad), max_gap)
    angle = np.rad2deg(np.arctan2(sin_f, cos_f)) % 360
    # 原本就有值的位置保持原值，只採用插值補出來的部分
    return series.where(series.notna(), angle)


def impute(hourly: pd.DataFrame, max_gap: int):
    """對每個測項補短缺失，並回傳 (補值後資料, 補值紀錄 mask)。"""
    filled = hourly.copy()
    for col in hourly.columns:
        if col in ANGLE_ITEMS:
            filled[col] = fill_short_gaps_angle(hourly[col], max_gap)
        else:
            filled[col] = fill_short_gaps(hourly[col], max_gap)
    imputed_mask = hourly.isna() & filled.notna()
    return filled, imputed_mask


def standardize(filled: pd.DataFrame, train_ratio: float):
    """Z-score 標準化 (x - mean) / std。

    為避免資料洩漏，mean / std 只用時間上最前面 train_ratio 比例的資料 (訓練期) 計算，
    再套用到全部時間點。NaN 會被略過 (計算時) 並原樣保留 (轉換後)。
    回傳 (標準化後資料, 各測項的 mean/std 表)。
    """
    n_train = int(len(filled) * train_ratio)
    train = filled.iloc[:n_train]

    mean = train.mean()  # pandas 預設略過 NaN
    std = train.std().replace(0, 1.0)  # 避免常數欄位除以 0

    scaled = (filled - mean) / std
    stats = pd.DataFrame({"mean": mean, "std": std})
    stats.index.name = "item"
    return scaled, stats


def drop_missing_rows(filled: pd.DataFrame) -> pd.DataFrame:
    """移除補值後仍有缺失 (長缺口、頭尾缺失) 的時間點。

    只要該小時任何一個測項仍為 NaN，整列就丟掉。注意：丟掉之後時間索引不再連續，
    後續切 9 小時視窗時，請用 datetime 索引確認視窗內相鄰兩點確實相差 1 小時。
    """
    return filled.dropna(how="any")


def print_summary(raw: pd.DataFrame, filled: pd.DataFrame) -> None:
    summary = pd.DataFrame(
        {
            "missing_before": raw.isna().sum(),
            "missing_after": filled.isna().sum(),
            "missing_before_%": (raw.isna().mean() * 100).round(2),
            "missing_after_%": (filled.isna().mean() * 100).round(2),
        }
    )
    print(f"時間範圍：{raw.index.min()} ~ {raw.index.max()}  (共 {len(raw)} 小時)")
    print(summary.to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="新竹_2025.csv", help="原始 CSV 路徑")
    parser.add_argument("--output", default="hsinchu_2025_clean.csv", help="輸出 CSV 路徑")
    parser.add_argument("--max-gap", type=int, default=3, help="最多補幾小時的連續缺失")
    parser.add_argument(
        "--scaled-output", default="hsinchu_2025_scaled.csv", help="標準化後輸出 CSV 路徑"
    )
    parser.add_argument(
        "--stats-output", default="hsinchu_2025_scaler_stats.csv", help="mean/std 輸出 CSV 路徑"
    )
    parser.add_argument(
        "--train-ratio", type=float, default=0.7, help="用前多少比例的時間資料計算 mean/std"
    )
    args = parser.parse_args()

    wide = load_wide_csv(args.input)
    hourly = wide_to_hourly(wide)
    filled, imputed_mask = impute(hourly, args.max_gap)

    print_summary(hourly, filled)

    # 移除補值後仍有缺失的時間點 (連續長缺失)
    n_before = len(filled)
    filled = drop_missing_rows(filled)
    print(f"\n移除仍含缺失的時間點：{n_before - len(filled)} 列 (剩 {len(filled)} / {n_before} 列)")

    # 另外輸出 PM2.5 是否為補出來的值，方便後續「目標值不補值」時過濾樣本
    out = filled.copy()
    if "PM2.5" in out.columns:
        out["PM2.5_is_imputed"] = imputed_mask["PM2.5"].loc[filled.index].astype(int)

    out.to_csv(args.output, encoding="utf-8-sig")
    print(f"\n已輸出：{args.output}")

    # 標準化 (只處理測項欄位，不含 PM2.5_is_imputed 旗標)
    scaled, stats = standardize(filled, args.train_ratio)
    scaled_out = scaled.copy()
    if "PM2.5" in out.columns:
        scaled_out["PM2.5_is_imputed"] = out["PM2.5_is_imputed"]
    scaled_out.to_csv(args.scaled_output, encoding="utf-8-sig")
    stats.to_csv(args.stats_output, encoding="utf-8-sig")
    print(f"已輸出標準化資料：{args.scaled_output}")
    print(f"已輸出 mean/std：{args.stats_output} (訓練期 = 前 {args.train_ratio:.0%} 時間)")


if __name__ == "__main__":
    main()