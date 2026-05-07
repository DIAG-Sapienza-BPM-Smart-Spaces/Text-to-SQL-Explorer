import json
import os
import random
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


def _safe_set_page_config() -> None:
	"""Set page config when running standalone, skip when embedded."""
	if os.environ.get("STREAMLIT_EMBEDDED_MODE") == "1":
		return
	try:
		st.set_page_config(page_title="Binary Judge Visualization", layout="wide")
	except Exception:
		# Streamlit only allows setting page config once per app run.
		pass


_safe_set_page_config()

# Development toggle: when True, test binary data is ignored.
DEVELOPMENT_MODE = False


MODEL_LABELS = {
	"cogito_70b": "Cogito 70B",
	"llama_70b": "Llama 70B",
	"deepseek-chat": "DeepSeek Chat",
	"qwen2.5-coder_32b": "Qwen2.5 Coder 32B",
	"qwen3-coder_30b": "Qwen3 Coder 30B",
	"ground_truth": "Ground Truth",
}

QUERY_FILE_TO_DATASET = {
	"datasets_files/bird/dev.json": "BIRD Developer",
	"datasets_files/bird/train.json": "BIRD Training",
	"datasets_files/spider/dev.json": "SPIDER",
	"datasets_files/spider/test.json": "SPIDER Test",
	"datasets_files/spider/train_spider.json": "SPIDER",
	"datasets_files/spider/train_others.json": "SPIDER",
	"dev.json": "BIRD Developer",
	"bird_training_queries.json": "BIRD Training",
	"spider_queries.json": "SPIDER",
	"train.json": "BIRD Training",
	"test.json": "SPIDER Test",
}


def normalize_query_file_key(path_value: str) -> str:
	"""Normalize a query file path so it can be used as a stable dictionary key."""
	if not path_value:
		return ""
	return str(path_value).replace("\\", "/").strip().lower()


def to_label(model_name: str) -> str:
	if not model_name:
		return "Unknown"
	if model_name in MODEL_LABELS:
		return MODEL_LABELS[model_name]
	token = model_name.replace("_", " ").strip()
	return " ".join(word.capitalize() for word in token.split())


def load_json(path: Path) -> Any:
	with open(path, "r", encoding="utf-8") as f:
		return json.load(f)


def collect_candidate_models_from_folder(project_root: str) -> List[str]:
	"""Collect candidate model ids from candidates/evaluation_sql_metrics_* files."""
	root = Path(project_root)
	candidates_dir = root / "candidates"
	if not candidates_dir.exists():
		return []

	prefix = "evaluation_sql_metrics_"
	suffix = "_vs_ground_truth.json"
	models = set()
	for path in candidates_dir.glob(f"{prefix}*{suffix}"):
		name = path.name
		if not (name.startswith(prefix) and name.endswith(suffix)):
			continue
		model_id = name[len(prefix):-len(suffix)]
		if model_id:
			models.add(model_id)

	return sorted(models)


@st.cache_data
def load_query_indexes(project_root: str) -> Dict[str, Dict[Tuple[str, int], Dict[str, Any]]]:
	root = Path(project_root)
	indexes: Dict[str, Dict[Tuple[str, int], Dict[str, Any]]] = {}

	candidate_paths = [
		root / "datasets_files" / "BIRD" / "dev.json",
		root / "datasets_files" / "BIRD" / "train.json",
		root / "datasets_files" / "SPIDER" / "dev.json",
		root / "datasets_files" / "SPIDER" / "test.json",
		root / "datasets_files" / "SPIDER" / "train_spider.json",
		root / "datasets_files" / "SPIDER" / "train_others.json",
	]

	for path in candidate_paths:
		if not path.exists():
			continue

		try:
			payload = load_json(path)
		except Exception:
			continue

		if not isinstance(payload, list):
			continue

		index: Dict[Tuple[str, int], Dict[str, Any]] = {}
		for row in payload:
			if not isinstance(row, dict):
				continue

			db_id = row.get("db_id") or row.get("database")
			qid = row.get("question_id", row.get("id"))
			if db_id is None or qid is None:
				continue

			try:
				key = (str(db_id), int(qid))
			except (TypeError, ValueError):
				continue

			index[key] = {
				"question": row.get("question") or row.get("query") or "",
				"evidence": row.get("evidence") or "",
				"ground_truth_sql": row.get("SQL") or row.get("sql") or "",
				"difficulty": row.get("difficulty") or row.get("complexity"),
				"complexity": row.get("complexity"),
				"length": row.get("length"),
				"tables": row.get("tables"),
				"attributes": row.get("attributes"),
			}

		rel_key = normalize_query_file_key(str(path.relative_to(root)))
		indexes[rel_key] = index
		# Backward-compatibility fallback for payloads that only include file names.
		indexes.setdefault(path.name, index)

	return indexes


@st.cache_data
def load_binary_data(project_root: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
	root = Path(project_root)
	binary_dir = root / "binary_choices"
	test_binary_path = root / "test_data" / "test_binary_choices.json"

	query_indexes = load_query_indexes(project_root)
	configured_candidates = collect_candidate_models_from_folder(project_root)
	configured_candidate_set = set(configured_candidates)

	all_rows: List[Dict[str, Any]] = []
	candidate_models = set(configured_candidates)
	judge_models = set()

	if binary_dir.exists():
		for path in sorted(binary_dir.glob("*_binary_choices.json")):
			try:
				payload = load_json(path)
			except Exception:
				continue

			candidate_model = payload.get("candidate_model", "")
			judge_model = payload.get("judge_model", "")
			query_file_raw = payload.get("query_file", "")
			query_file_name = Path(query_file_raw).name
			query_file_key = normalize_query_file_key(query_file_raw)
			dataset_name = QUERY_FILE_TO_DATASET.get(
				query_file_key,
				QUERY_FILE_TO_DATASET.get(query_file_name, "Unknown"),
			)
			query_index = query_indexes.get(query_file_key, query_indexes.get(query_file_name, {}))

			if configured_candidate_set and candidate_model and candidate_model not in configured_candidate_set:
				continue
			if (not configured_candidate_set) and candidate_model:
				candidate_models.add(candidate_model)
			if judge_model:
				judge_models.add(judge_model)

			rows = payload.get("results", [])
			if not isinstance(rows, list):
				continue

			for row in rows:
				if not isinstance(row, dict):
					continue

				db_id = row.get("db_id") or row.get("Database")
				qid = row.get("question_id", row.get("Query ID"))
				if db_id is None or qid is None:
					continue

				try:
					qid_int = int(qid)
				except (TypeError, ValueError):
					continue

				query_meta = query_index.get((str(db_id), qid_int), {})
				candidate_model_value = row.get("candidate_model", candidate_model)
				if configured_candidate_set and candidate_model_value not in configured_candidate_set:
					continue

				enriched_row = {
					"candidate_model": candidate_model_value,
					"judge_model": row.get("Judge Model", judge_model),
					"dataset": dataset_name,
					"db_id": str(db_id),
					"question_id": qid_int,
					"nl_query": query_meta.get("question", ""),
					"evidence": query_meta.get("evidence", ""),
					"ground_truth_sql": query_meta.get("ground_truth_sql", ""),
					"difficulty": query_meta.get("difficulty"),
					"complexity": query_meta.get("complexity"),
					"length": query_meta.get("length"),
					"tables": query_meta.get("tables"),
					"attributes": query_meta.get("attributes"),
					"candidate_sql": row.get("candidate_sql", ""),
					"choice": str(row.get("choice", "")).strip().upper(),
					"reasoning": row.get("Reasoning", ""),
					"execution_vs_ground_truth": row.get("execution_vs_ground_truth") or row.get("candidate_metrics"),
					"candidate_metrics": row.get("candidate_metrics"),
				}
				all_rows.append(enriched_row)

	if (not DEVELOPMENT_MODE) and test_binary_path.exists():
		try:
			test_payload = load_json(test_binary_path)
		except Exception:
			test_payload = []

		if isinstance(test_payload, list):
			for row in test_payload:
				if not isinstance(row, dict):
					continue

				candidate_model = row.get("candidate_model", "")
				judge_model = row.get("judge_model", "")
				dataset_name = row.get("dataset", "Unknown")
				db_id = row.get("db_id")
				qid = row.get("question_id")
				if db_id is None or qid is None:
					continue

				try:
					qid_int = int(qid)
				except (TypeError, ValueError):
					continue

				if candidate_model:
					if configured_candidate_set and candidate_model not in configured_candidate_set:
						continue
					if not configured_candidate_set:
						candidate_models.add(candidate_model)
				if judge_model:
					judge_models.add(judge_model)

				query_meta = {}
				for query_index in query_indexes.values():
					query_meta = query_index.get((str(db_id), qid_int), {})
					if query_meta:
						break

				enriched_row = {
					"candidate_model": candidate_model,
					"judge_model": judge_model,
					"dataset": dataset_name,
					"db_id": str(db_id),
					"question_id": qid_int,
					"nl_query": query_meta.get("question", ""),
					"evidence": query_meta.get("evidence", ""),
					"ground_truth_sql": query_meta.get("ground_truth_sql", ""),
					"difficulty": query_meta.get("difficulty"),
					"complexity": query_meta.get("complexity"),
					"length": query_meta.get("length"),
					"tables": query_meta.get("tables"),
					"attributes": query_meta.get("attributes"),
					"candidate_sql": row.get("candidate_sql", ""),
					"choice": str(row.get("choice", "")).strip().upper(),
					"reasoning": row.get("Reasoning", row.get("reasoning", "")),
					"execution_vs_ground_truth": row.get("execution_vs_ground_truth") or row.get("candidate_metrics"),
					"candidate_metrics": row.get("candidate_metrics"),
				}
				all_rows.append(enriched_row)

	return (
		all_rows,
		sorted(candidate_models),
		sorted(judge_models),
	)


def choice_color(choice: str) -> str:
	if choice == "ACCEPT":
		return "#0f766e"
	if choice == "REJECT":
		return "#b91c1c"
	return "#475569"


def fmt_float(value: Any) -> str:
	if value is None:
		return "N/A"
	try:
		return f"{float(value):.3f}"
	except (TypeError, ValueError):
		return "N/A"


def pick_random_result(rows: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
	if not rows:
		return None

	by_dataset: Dict[str, List[Dict[str, Any]]] = {}
	for row in rows:
		ds = row.get("dataset", "Unknown")
		by_dataset.setdefault(ds, []).append(row)

	random_dataset = random.choice(list(by_dataset.keys()))
	return random.choice(by_dataset[random_dataset])


def render_hover_value(label: str, value: Any, max_chars: int = 80) -> None:
	text = str(value) if value is not None else "N/A"
	compact = text if len(text) <= max_chars else f"{text[:max_chars - 1]}..."

	st.markdown(
		f"""
		<div style="margin-bottom:0.5rem;">
			<div style="font-size:0.8rem;color:#64748b;font-weight:600;">{escape(label)}</div>
			<div title="{escape(text)}" style="
				border: 1px solid rgba(100,116,139,0.25);
				border-radius: 0.55rem;
				padding: 0.42rem 0.55rem;
				background: rgba(148,163,184,0.08);
				white-space: nowrap;
				overflow: hidden;
				text-overflow: ellipsis;
				font-size: 0.92rem;
			">{escape(compact)}</div>
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_query_details_panel(sample: Dict[str, Any], compact: bool = False) -> None:
	if compact:
		st.markdown("##### Query Details")
	else:
		st.subheader("Selected Query")

	info_cols = st.columns(3)
	with info_cols[0]:
		render_hover_value("Dataset", sample.get("dataset", "Unknown"), max_chars=22)
	with info_cols[1]:
		render_hover_value("Database", sample.get("db_id", "Unknown"), max_chars=22)
	with info_cols[2]:
		render_hover_value("Query ID", sample.get("question_id", "-"), max_chars=22)

	nl_query = sample.get("nl_query") or ""
	if not nl_query:
		metrics_fallback = sample.get("candidate_metrics") if isinstance(sample.get("candidate_metrics"), dict) else {}
		nl_query = metrics_fallback.get("question", "") if isinstance(metrics_fallback, dict) else ""

	evidence = sample.get("evidence") or ""
	if not evidence:
		metrics_fallback = sample.get("candidate_metrics") if isinstance(sample.get("candidate_metrics"), dict) else {}
		evidence = metrics_fallback.get("evidence", "") if isinstance(metrics_fallback, dict) else ""

	with st.container(border=True):
		st.markdown("**Natural Language Query**")
		if nl_query:
			st.write(nl_query)
		else:
			st.info("Natural language query not available for this sample.")

		st.markdown("**Evidence**")
		if evidence:
			st.write(evidence)
		else:
			st.caption("No evidence available for this sample.")


def render_sql_panel(sample: Dict[str, Any], compact: bool = False) -> None:
	if compact:
		st.markdown("##### SQL Comparison")
	else:
		st.subheader("Selected Query")

	with st.container(border=True):
		st.markdown("**Candidate SQL**")
		st.code(sample.get("candidate_sql", ""), language="sql")

		gt_sql = sample.get("ground_truth_sql", "")
		if gt_sql:
			st.markdown("**Ground Truth SQL**")
			st.code(gt_sql, language="sql")


def render_decision_panel(sample: Dict[str, Any], compact: bool = False) -> None:
	if compact:
		st.markdown("##### Decision")
	else:
		st.subheader("Judge Decision")
	choice = sample.get("choice", "UNKNOWN")
	color = choice_color(choice)
	font_size = "24px" if compact else "34px"
	padding = "10px" if compact else "18px"
	border_radius = "8px" if compact else "12px"

	st.markdown(
		f"""
		<div style="
			border: 2px solid {color};
			border-radius: {border_radius};
			padding: {padding};
			text-align: center;
			font-size: {font_size};
			font-weight: 700;
			color: {color};
			background: rgba(148, 163, 184, 0.08);
		">
			{choice}
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_reasoning_panel(sample: Dict[str, Any], compact: bool = False) -> None:
	if compact:
		st.markdown("##### Reasoning")
	else:
		st.subheader("Judge Reasoning")
	with st.container(border=True):
		reasoning = sample.get("reasoning", "")
		if reasoning:
			st.write(reasoning)
		else:
			st.info("Reasoning not available for this sample.")


def render_performance_panel(sample: Dict[str, Any], compact: bool = False) -> None:
	if compact:
		st.markdown("##### Execution Metrics")
	else:
		st.subheader("Actual Performance")
	perf = sample.get("execution_vs_ground_truth")
	if not isinstance(perf, dict):
		# Real binary artifacts often expose metric fields under candidate_metrics.
		perf = sample.get("candidate_metrics")

	with st.container(border=True):
		if not isinstance(perf, dict):
			st.info("No execution-vs-ground-truth metrics available for this sample.")
			return

		# Prefer current metric names from pairwise comparisons; keep fallbacks for older files.
		execution_accuracy = perf.get("execution_accuracy")
		exact_match = perf.get("exact_match")
		sql_f1_score = perf.get("sql_f1_score", perf.get("f1_score"))
		response_f1_score = perf.get("response_schema_f1_score", perf.get("schema_precision"))
		cell_f1_score = perf.get("cell_f1_score", perf.get("cell_value_accuracy"))

		metric_rows = [
			("Execution Accuracy", fmt_float(execution_accuracy)),
			("Exact Match", fmt_float(exact_match)),
			("SQL F1 Score", fmt_float(sql_f1_score)),
			("Response Schema F1 Score", fmt_float(response_f1_score)),
			("Cell F1 Score", fmt_float(cell_f1_score)),
		]

		label_size = "0.82rem" if compact else "0.95rem"
		value_size = "0.98rem" if compact else "1.15rem"
		row_padding = "0.3rem 0" if compact else "0.5rem 0"

		for metric_name, metric_value in metric_rows:
			st.markdown(
				f"""
				<div style=\"display:flex;justify-content:space-between;align-items:center;padding:{row_padding};border-bottom:1px solid rgba(148,163,184,0.2);\">
					<div style=\"font-size:{label_size};color:#94a3b8;\">{escape(metric_name)}</div>
					<div style=\"font-size:{value_size};font-weight:700;\">{escape(metric_value)}</div>
				</div>
				""",
				unsafe_allow_html=True,
			)

		comparison_flag = perf.get("comparison_performed")
		if comparison_flag is True:
			st.success("Comparison performed successfully.")
		elif comparison_flag is False:
			st.warning("Comparison failed or was skipped.")

		if perf.get("fast_path"):
			st.caption(f"Fast path: {perf['fast_path']}")
		if perf.get("error"):
			st.error(f"Comparison error: {perf['error']}")


def main(show_title: bool = True, compact: bool = False) -> None:
	project_root = Path(__file__).resolve().parent
	all_rows, candidate_models, judge_models = load_binary_data(str(project_root))

	if compact:
		st.markdown(
			"""
			<style>
			div[data-testid="stSelectbox"] > label,
			div[data-testid="stButton"] button,
			div[data-testid="stCaptionContainer"] p {
				font-size: 0.85rem;
			}
			div[data-testid="stButton"] button {
				padding-top: 0.25rem;
				padding-bottom: 0.25rem;
			}
			</style>
			""",
			unsafe_allow_html=True,
		)

	if show_title:
		st.title("Binary Judge Demo")
		st.caption("Interactive view of ACCEPT/REJECT judgments over candidate SQL queries.")

	if not all_rows:
		st.error(
			"No valid binary choice data found in the binary_choices folder. "
			"Add *_binary_choices.json files and rerun the app."
		)
		return

	if compact:
		control_col, details_col, sql_col = st.columns([0.72, 0.95, 2.25], gap="large")

		with control_col:
			with st.container(border=True):
				st.markdown("##### Selection")
				selected_candidate = st.selectbox(
					"Candidate model",
					options=candidate_models,
					format_func=to_label,
				)

				single_judge = st.selectbox(
					"Judge",
					options=judge_models,
					format_func=to_label,
				)
				selected_judge_key = single_judge

				run_clicked = st.button("Run", type="primary", use_container_width=True)

		filtered = [
			row
			for row in all_rows
			if row.get("candidate_model") == selected_candidate and row.get("judge_model") == selected_judge_key
		]

		if run_clicked:
			sampled = pick_random_result(filtered)
			st.session_state["binary_sample"] = sampled
			st.session_state["binary_filters"] = {
				"candidate_model": selected_candidate,
				"judge_model": selected_judge_key,
			}

		sample = st.session_state.get("binary_sample")
		sample_filters = st.session_state.get("binary_filters", {})
		show_sample = sample and sample_filters == {"candidate_model": selected_candidate, "judge_model": selected_judge_key}

		with control_col:
			if show_sample:
				render_decision_panel(sample, compact=True)

		with details_col:
			if not filtered:
				st.warning(
					"No rows available for the selected candidate/judge combination. "
					"Try another judge or candidate model."
				)
			if show_sample:
				render_query_details_panel(sample, compact=True)
				render_performance_panel(sample, compact=True)
			else:
				st.info("Select the models and click Run to sample one query.")

		with sql_col:
			if show_sample:
				render_sql_panel(sample, compact=True)
				render_reasoning_panel(sample, compact=True)
			else:
				st.info("SQL comparison and reasoning will appear here after running a sample.")
	else:
		left_col, right_col = st.columns([0.85, 2.35], gap="large")

		with left_col:
			with st.container(border=True):
				st.subheader("Selection")
				selected_candidate = st.selectbox(
					"Candidate model",
					options=candidate_models,
					format_func=to_label,
				)

				single_judge = st.selectbox(
					"Judge",
					options=judge_models,
					format_func=to_label,
				)
				selected_judge_key = single_judge

				run_clicked = st.button("Run", type="primary", use_container_width=True)

		filtered = [
			row
			for row in all_rows
			if row.get("candidate_model") == selected_candidate and row.get("judge_model") == selected_judge_key
		]

		if run_clicked:
			sampled = pick_random_result(filtered)
			st.session_state["binary_sample"] = sampled
			st.session_state["binary_filters"] = {
				"candidate_model": selected_candidate,
				"judge_model": selected_judge_key,
			}

		sample = st.session_state.get("binary_sample")
		sample_filters = st.session_state.get("binary_filters", {})
		show_sample = sample and sample_filters == {"candidate_model": selected_candidate, "judge_model": selected_judge_key}

		with left_col:
			st.caption(f"Matching rows: {len(filtered)}")
			if show_sample:
				render_decision_panel(sample, compact=False)
				render_performance_panel(sample, compact=False)

		with right_col:
			if not filtered:
				st.warning(
					"No rows available for the selected candidate/judge combination. "
					"Try another judge or candidate model."
				)

			if show_sample:
				render_query_details_panel(sample, compact=False)
				render_sql_panel(sample, compact=False)
				render_reasoning_panel(sample, compact=False)
			else:
				st.info("Select the models and click Run to sample one query and visualize the binary decision.")


if __name__ == "__main__":
	main()
