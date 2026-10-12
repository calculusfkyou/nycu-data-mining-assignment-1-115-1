import csv
import math
from datetime import datetime, timedelta
from pathlib import Path


HOUR_COLUMNS = [f"{hour:02d}" for hour in range(24)]
MISSING_MARKERS = {"", "#", "*", "NA", "A", "X"}
SOURCE_FILE = "新竹_2025.csv"
OUTPUT_FILE = "pure.csv"
STANDARDIZED_FILE = "standarized.csv"


def format_value(value: float) -> str:
	if value.is_integer():
		return str(int(value))
	return format(value, ".15g")


def load_pm25(source_path: Path) -> tuple[list[datetime], list[float | None]]:
	timestamps: list[datetime] = []
	values: list[float | None] = []

	with source_path.open("r", encoding="utf-8-sig", newline="") as source:
		reader = csv.DictReader(source)
		required_columns = {"日期", "測項", *HOUR_COLUMNS}
		if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
			raise ValueError("CSV 欄位不完整，需包含日期、測項及 00 至 23 小時欄位。")

		for line_number, row in enumerate(reader, start=2):
			if row.get("測項") != "PM2.5":
				continue

			date_text = row.get("日期", "")
			try:
				day = datetime.strptime(date_text, "%Y/%m/%d %H:%M:%S")
			except ValueError as error:
				raise ValueError(f"第 {line_number} 列日期格式錯誤：{date_text}") from error

			for hour in HOUR_COLUMNS:
				timestamp = day.replace(hour=int(hour))
				raw_value = (row.get(hour) or "").strip()
				timestamps.append(timestamp)

				if raw_value.upper() in MISSING_MARKERS:
					values.append(None)
					continue

				try:
					value = float(raw_value)
				except ValueError as error:
					raise ValueError(
						f"第 {line_number} 列 {hour} 時的 PM2.5 數值無效：{raw_value}"
					) from error
				if not math.isfinite(value):
					raise ValueError(f"第 {line_number} 列 {hour} 時的 PM2.5 數值非有限值。")
				values.append(value)

	if not values:
		raise ValueError("來源 CSV 中找不到 PM2.5 資料。")

	for previous, current in zip(timestamps, timestamps[1:]):
		if current - previous != timedelta(hours=1):
			raise ValueError(f"資料時間不連續：{previous} 之後是 {current}。")

	return timestamps, values


def fill_missing_values(values: list[float | None]) -> list[float]:
	filled = [value for value in values]
	index = 0

	while index < len(filled):
		if filled[index] is not None:
			index += 1
			continue

		gap_start = index
		while index < len(filled) and filled[index] is None:
			index += 1
		gap_end = index

		if gap_start == 0 or gap_end == len(filled):
			raise ValueError("資料開頭或結尾有缺值，無法依前後小時補值。")

		previous = filled[gap_start - 1]
		following = filled[gap_end]
		if previous is None or following is None:
			raise ValueError("缺值區間缺少有效的前後小時資料。")

		gap_length = gap_end - gap_start
		for offset in range(1, gap_length + 1):
			interpolated = previous + (following - previous) * offset / (gap_length + 1)
			filled[gap_start + offset - 1] = float(math.ceil(interpolated))

	return [value for value in filled if value is not None]


def standardize_csv(source_path: Path, output_path: Path) -> None:
	with source_path.open("r", encoding="utf-8-sig", newline="") as source:
		reader = csv.DictReader(source)
		if reader.fieldnames is None or "PM2.5" not in reader.fieldnames:
			raise ValueError("輸入 CSV 必須包含 PM2.5 欄位。")
		fieldnames = reader.fieldnames
		rows = list(reader)

	if not rows:
		raise ValueError("輸入 CSV 沒有資料，無法進行標準化。")

	try:
		values = [float(row["PM2.5"]) for row in rows]
	except (KeyError, TypeError, ValueError) as error:
		raise ValueError("PM2.5 欄位包含無效數值，無法進行標準化。") from error
	if not all(math.isfinite(value) for value in values):
		raise ValueError("PM2.5 欄位包含非有限值，無法進行標準化。")

	mean = sum(values) / len(values)
	standard_deviation = math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))
	for row, value in zip(rows, values):
		row["PM2.5"] = format_value((value - mean) / standard_deviation) if standard_deviation else "0"

	with output_path.open("w", encoding="utf-8-sig", newline="") as output:
		writer = csv.DictWriter(output, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

	print(f"已輸出標準化資料至 {output_path}")


def main() -> None:
	directory = Path(__file__).resolve().parent
	timestamps, values = load_pm25(directory / SOURCE_FILE)
	filled_values = fill_missing_values(values)
	output_path = directory / OUTPUT_FILE

	with output_path.open("w", encoding="utf-8-sig", newline="") as output:
		writer = csv.writer(output)
		writer.writerow(["日期", "小時", "PM2.5"])
		for timestamp, value in zip(timestamps, filled_values):
			writer.writerow([timestamp.strftime("%Y-%m-%d"), timestamp.strftime("%H"), format_value(value)])

	print(f"已輸出 {len(filled_values)} 筆 PM2.5 小時資料至 {output_path}")
	standardize_csv(output_path, directory / STANDARDIZED_FILE)


if __name__ == "__main__":
	main()
