新竹_2025.csv 是 從政府網站下載下來的 raw data

有三種資料前處理方法
1. 只有 PM2.5 指標（Only_PM2.5）
    - preprocessing.py
    - pure.csv（將缺失值以前後一小時的平均取代，尚未進行標準化）
    - standardized.csv （經過標準化）
2. 所有指標 （All_Matrics）
    - preprocessing_all.py
    - standardized_all.csv（經過標準化，也有標註 train 跟 test 了 80%-20%）
    - 前面的 column 紀錄該指標是否為缺失值（0代表沒缺失）及缺失原因，後面才是真值
3. 所有指標 + 移除缺失值（All_and_Removed）
    - preprocess.py
    - hsinchu_2025_clean.csv（少量缺失做插值，太多連續缺失值的資料直接移除）
    - hsinchu_2025_scaled.csv（做過標準化）
    - 這邊還沒切 Train