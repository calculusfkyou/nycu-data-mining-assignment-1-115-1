import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


SOURCE_FILE = "新竹_2025.csv"
PURE_FILE = "pure_all.csv"
STANDARDIZED_FILE = "standardized_all.csv"
TARGET_METRIC = "PM2.5"
HISTORY_HOURS = 9
TRAIN_RATIO = 0.8
HOUR_COLUMNS = [f"{hour:02d}" for hour in range(24)]

# 這些符號都代表該次觀測無效或缺失；保留原因欄，避免把原因資訊丟掉。
MISSING_REASONS = {
	"": "blank",
	"#": "instrument_check_invalid",
	"*": "program_check_invalid",
	"X": "manual_check_invalid",
	"A": "instrument_fault_alarm",
}


def format_value(value: float) -> str:
	if value.is_integer():
		return str(int(value))
	return format(value, ".15g")


def read_measurements(
	source_path: Path,
) -> tuple[list[datetime], dict[str, dict[datetime, tuple[float | None, str]]]]:
	measurements: dict[str, dict[datetime, tuple[float | None, str]]] = {}
	all_timestamps: set[datetime] = set()

	with source_path.open("r", encoding="utf-8-sig", newline="") as source:
		reader = csv.DictReader(source)
		required_columns = {"測站", "日期", "測項", *HOUR_COLUMNS}
		if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
			raise ValueError("來源 CSV 欄位不完整，需包含測站、日期、測項及 00 至 23 小時欄位。")

		for line_number, row in enumerate(reader, start=2):
			if row.get("測站") != "新竹":
				continue

			metric = (row.get("測項") or "").strip()
			if not metric:
				continue
			try:
				day = datetime.strptime(row.get("日期", ""), "%Y/%m/%d %H:%M:%S")
			except ValueError as error:
				raise ValueError(f"第 {line_number} 列日期格式錯誤：{row.get('日期')}") from error

			metric_values = measurements.setdefault(metric, {})
			for hour in HOUR_COLUMNS:
				timestamp = day.replace(hour=int(hour), minute=0, second=0, microsecond=0)
				if timestamp in metric_values:
					raise ValueError(f"第 {line_number} 列出現重複資料：{metric} {timestamp}。")
				all_timestamps.add(timestamp)
				raw_value = (row.get(hour) or "").strip()
				reason = MISSING_REASONS.get(raw_value.upper())
				if reason is not None:
					metric_values[timestamp] = (None, reason)
					continue

				try:
					value = float(raw_value)
				except ValueError:
					# 未知的非數值標記也不當作實測值；原始文字留在原因欄供追查。
					metric_values[timestamp] = (None, f"other_invalid:{raw_value}")
					continue
				if not math.isfinite(value):
					metric_values[timestamp] = (None, f"non_finite:{raw_value}")
					continue
				metric_values[timestamp] = (value, "")

	if TARGET_METRIC not in measurements or not all_timestamps:
		raise ValueError(f"來源 CSV 中找不到新竹測站的 {TARGET_METRIC} 資料。")

	timestamps = sorted(all_timestamps)
	for previous, current in zip(timestamps, timestamps[1:]):
		if current - previous != timedelta(hours=1):
			raise ValueError(f"資料時間不連續：{previous} 之後是 {current}。")
	return timestamps, measurements


def build_supervised_rows(
	timestamps: list[datetime],
	measurements: dict[str, dict[datetime, tuple[float | None, str]]],
) -> tuple[list[str], list[dict[str, str | float | int]]]:
	metrics = sorted(measurements)
	if TARGET_METRIC not in metrics:
		raise ValueError(f"找不到目標測項 {TARGET_METRIC}。")

	feature_columns: list[str] = []
	for lag in range(HISTORY_HOURS, 0, -1):
		for metric in metrics:
			feature_columns.append(f"{metric}_lag_{lag}h")
	columns = ["target_datetime", "target_PM2.5", "dataset_split"]
	for feature in feature_columns:
		columns.extend([f"{feature}_missing", f"{feature}_missing_reason"])
	columns.extend(feature_columns)

	rows: list[dict[str, str | float | int]] = []
	for target_index in range(HISTORY_HOURS, len(timestamps)):
		target_time = timestamps[target_index]
		target_value, _ = measurements[TARGET_METRIC].get(target_time, (None, "row_missing"))
		# 無效目標沒有可信答案，因此略過該列，不用插補值訓練模型。
		if target_value is None:
			continue

		row: dict[str, str | float | int] = {
			"target_datetime": target_time.strftime("%Y-%m-%d %H:%M:%S"),
			"target_PM2.5": target_value,
			"dataset_split": "",
		}
		for lag in range(HISTORY_HOURS, 0, -1):
			feature_time = timestamps[target_index - lag]
			for metric in metrics:
				feature = f"{metric}_lag_{lag}h"
				value, reason = measurements[metric].get(feature_time, (None, "row_missing"))
				row[feature] = "" if value is None else value
				row[f"{feature}_missing"] = int(value is None)
				row[f"{feature}_missing_reason"] = reason
		rows.append(row)

	if len(rows) < 2:
		raise ValueError("有效 PM2.5 目標資料不足，無法建立訓練與測試資料。")

	# 依時間排序後切分，避免隨機切分讓未來資料混入訓練集。
	train_count = max(1, min(len(rows) - 1, int(len(rows) * TRAIN_RATIO)))
	for index, row in enumerate(rows):
		row["dataset_split"] = "train" if index < train_count else "test"
	return columns, rows


def standardize_rows(
	columns: list[str], rows: list[dict[str, str | float | int]], output_path: Path
) -> None:
	train_rows = [row for row in rows if row["dataset_split"] == "train"]
	feature_columns = [
		column
		for column in columns
		if column not in {"target_datetime", "target_PM2.5", "dataset_split"}
		and not column.endswith(("_missing", "_missing_reason"))
	]
	target_and_features = ["target_PM2.5", *feature_columns]
	means: dict[str, float] = {}
	standard_deviations: dict[str, float] = {}

	for column in target_and_features:
		observed_train_values = [float(row[column]) for row in train_rows if row[column] != ""]
		if not observed_train_values:
			raise ValueError(f"訓練資料的 {column} 全部缺失，無法估計補值平均數。")
		# 缺失特徵只以訓練期間的平均數補值；不使用未來列或雙側插值。
		means[column] = sum(observed_train_values) / len(observed_train_values)
		imputed_train_values = [
			means[column] if row[column] == "" else float(row[column]) for row in train_rows
		]
		variance = sum((value - means[column]) ** 2 for value in imputed_train_values) / len(train_rows)
		standard_deviations[column] = math.sqrt(variance)

	standardized_rows: list[dict[str, str | float | int]] = []
	for row in rows:
		standardized_row = dict(row)
		for column in target_and_features:
			value = means[column] if row[column] == "" else float(row[column])
			deviation = standard_deviations[column]
			standardized_row[column] = format_value((value - means[column]) / deviation) if deviation else "0"
		standardized_rows.append(standardized_row)

	with output_path.open("w", encoding="utf-8-sig", newline="") as output:
		writer = csv.DictWriter(output, fieldnames=columns)
		writer.writeheader()
		writer.writerows(standardized_rows)


def main() -> None:
	directory = Path(__file__).resolve().parent
	timestamps, measurements = read_measurements(directory / SOURCE_FILE)
	columns, rows = build_supervised_rows(timestamps, measurements)

	with (directory / PURE_FILE).open("w", encoding="utf-8-sig", newline="") as output:
		writer = csv.DictWriter(output, fieldnames=columns)
		writer.writeheader()
		writer.writerows(rows)

	standardized_path = directory / STANDARDIZED_FILE
	standardize_rows(columns, rows, standardized_path)
	train_count = sum(row["dataset_split"] == "train" for row in rows)
	print(f"已輸出 {len(rows)} 筆監督式資料至 {directory / PURE_FILE}")
	print(f"時間切分：訓練 {train_count} 筆、測試 {len(rows) - train_count} 筆")
	print(f"已輸出標準化資料至 {standardized_path}")


if __name__ == "__main__":
	main()
