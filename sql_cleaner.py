import json
import re
from pathlib import Path
from typing import Any


DEFAULT_SQL_FOLDER = Path(__file__).resolve().parent / "candidates"


def strip_markdown_code_fences(text: str) -> str:
	"""Remove optional Markdown SQL code fences while preserving query text."""
	stripped = text.strip()
	fenced = re.match(r"^```(?:\w+)?\s*\n?(.*?)\n?```$", stripped, flags=re.DOTALL)
	if fenced:
		return fenced.group(1)
	return text


def decode_escaped_whitespace(text: str) -> str:
	"""Convert literal escaped whitespace sequences to their character forms."""
	return text.replace("\\r", "\r").replace("\\n", "\n").replace("\\t", "\t")


def remove_control_characters(text: str) -> str:
	"""Drop non-printable control chars except line breaks and tabs."""
	return "".join(ch for ch in text if ch in ("\n", "\r", "\t") or ord(ch) >= 32)


def normalize_whitespace(text: str) -> str:
	"""Normalize spacing without rewriting SQL structure."""
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	text = text.replace("\t", " ")

	lines = [re.sub(r" +", " ", line).strip() for line in text.split("\n")]
	lines = [line for line in lines if line]
	return " ".join(lines).strip()


def clean_sql_query(sql: Any) -> str:
	"""Clean noisy SQL text from Text-to-SQL systems without semantic rewriting."""
	if sql is None:
		return ""

	sql_text = str(sql)
	sql_text = strip_markdown_code_fences(sql_text)
	sql_text = decode_escaped_whitespace(sql_text)
	sql_text = remove_control_characters(sql_text)
	sql_text = normalize_whitespace(sql_text)
	return sql_text


def get_source_sql(record: dict[str, Any]) -> Any:
	"""Pick the best SQL source field available in a record."""
	for key in ("extracted_sql", "generated_sql", "sql", "query"):
		if key in record and record[key] is not None:
			return record[key]
	return None


def add_clean_sql_to_records(records: list[dict[str, Any]]) -> int:
	"""Add clean_sql to each record and return number of processed rows."""
	count = 0
	for record in records:
		source_sql = get_source_sql(record)
		record["clean_sql"] = clean_sql_query(source_sql)
		count += 1
	return count


def clean_sql_file(file_path: Path) -> int:
	"""Load one JSON file, enrich records with clean_sql, and overwrite safely."""
	with file_path.open("r", encoding="utf-8") as f:
		payload = json.load(f)

	records_container: list[dict[str, Any]] | None = None
	wrapped_payload = False
	wrapped_key = ""

	if isinstance(payload, list):
		records_container = payload
	elif isinstance(payload, dict):
		for key in ("records", "results", "data", "items"):
			value = payload.get(key)
			if isinstance(value, list):
				records_container = value
				wrapped_payload = True
				wrapped_key = key
				break

	if records_container is None:
		raise ValueError(f"Unsupported JSON shape in {file_path.name}")

	records: list[dict[str, Any]] = []
	for idx, item in enumerate(records_container):
		if not isinstance(item, dict):
			raise ValueError(f"Record at index {idx} in {file_path.name} is not an object")
		records.append(item)

	processed = add_clean_sql_to_records(records)

	output_payload: Any = records
	if wrapped_payload:
		payload[wrapped_key] = records
		output_payload = payload

	tmp_path = file_path.with_suffix(file_path.suffix + ".tmp")
	with tmp_path.open("w", encoding="utf-8") as f:
		json.dump(output_payload, f, ensure_ascii=False, indent=2)
	tmp_path.replace(file_path)

	return processed


def clean_all_sql_files(sql_folder: Path = DEFAULT_SQL_FOLDER) -> None:
	"""Clean all JSON result files in candidates/ and append clean_sql per record."""
	if not sql_folder.exists() or not sql_folder.is_dir():
		raise FileNotFoundError(f"SQL folder not found: {sql_folder}")

	json_files = sorted(sql_folder.glob("*.json"))
	if not json_files:
		print(f"No JSON files found in {sql_folder}")
		return

	total_files = 0
	total_records = 0
	for file_path in json_files:
		processed = clean_sql_file(file_path)
		total_files += 1
		total_records += processed
		print(f"Cleaned {processed} records in {file_path.name}")

	print(f"Done. Updated {total_files} files and {total_records} records.")


if __name__ == "__main__":
	clean_all_sql_files()
