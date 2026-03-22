import json
import random
from html import escape
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import streamlit as st


st.set_page_config(page_title="Binary Judge Visualization", layout="wide")


MODEL_LABELS = {
	"cogito_70b": "Cogito 70B",
	"llama_70b": "Llama 70B",
	"deepseek-chat": "DeepSeek Chat",
	"qwen2.5-coder_32b": "Qwen2.5 Coder 32B",
	"qwen3-coder_30b": "Qwen3 Coder 30B",
	"ground_truth": "Ground Truth",
}

QUERY_FILE_TO_DATASET = {
	"dev.json": "BIRD Developer",
	"bird_training_queries.json": "BIRD Training",
	"spider_queries.json": "SPIDER",
	"train.json": "BIRD Training",
	"test.json": "SPIDER Test",
}


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

		indexes[path.name] = index

	return indexes


@st.cache_data
def load_binary_data(project_root: str) -> Tuple[List[Dict[str, Any]], List[str], List[str]]:
	root = Path(project_root)
	binary_dir = root / "binary_choices"
	fake_binary_path = root / "fake_data" / "fake_binary_choices.json"

	query_indexes = load_query_indexes(project_root)

	all_rows: List[Dict[str, Any]] = []
	candidate_models = set()
	judge_models = set()

	if binary_dir.exists():
		for path in sorted(binary_dir.glob("*_binary_choices.json")):
			try:
				payload = load_json(path)
			except Exception:
				continue

			candidate_model = payload.get("candidate_model", "")
			judge_model = payload.get("judge_model", "")
			query_file_name = Path(payload.get("query_file", "")).name
			dataset_name = QUERY_FILE_TO_DATASET.get(query_file_name, "Unknown")
			query_index = query_indexes.get(query_file_name, {})

			if candidate_model:
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

				enriched_row = {
					"candidate_model": row.get("candidate_model", candidate_model),
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
					"execution_vs_ground_truth": row.get("execution_vs_ground_truth"),
				}
				all_rows.append(enriched_row)

	if fake_binary_path.exists():
		try:
			fake_payload = load_json(fake_binary_path)
		except Exception:
			fake_payload = []

		if isinstance(fake_payload, list):
			for row in fake_payload:
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
					"execution_vs_ground_truth": row.get("execution_vs_ground_truth"),
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


def compose_judge_key(mode: str, single_judge: str, ensemble_judges: List[str]) -> str:
	if mode == "Single":
		return single_judge
	return " + ".join(sorted(ensemble_judges))


def ensemble_choice_from_votes(votes: List[str]) -> str:
	accept_count = sum(1 for vote in votes if vote == "ACCEPT")
	reject_count = sum(1 for vote in votes if vote == "REJECT")

	if accept_count > reject_count:
		return "ACCEPT"
	if reject_count > accept_count:
		return "REJECT"
	return "UNDECIDED"


def build_ensemble_rows(
	all_rows: List[Dict[str, Any]],
	candidate_model: str,
	selected_ensemble: List[str],
) -> List[Dict[str, Any]]:
	if len(selected_ensemble) < 2:
		return []

	selected_set = set(selected_ensemble)
	grouped: Dict[Tuple[str, int], List[Dict[str, Any]]] = {}

	for row in all_rows:
		if row.get("candidate_model") != candidate_model:
			continue
		judge_name = row.get("judge_model")
		if judge_name not in selected_set:
			continue

		key = (row.get("db_id", ""), int(row.get("question_id", -1)))
		grouped.setdefault(key, []).append(row)

	aggregated_rows: List[Dict[str, Any]] = []
	for _, rows in grouped.items():
		present_judges = {r.get("judge_model") for r in rows}
		if present_judges != selected_set:
			continue

		votes = [str(r.get("choice", "")).upper() for r in rows]
		final_choice = ensemble_choice_from_votes(votes)

		# All rows in the group refer to the same candidate SQL/query, so base metadata is shared.
		base = dict(rows[0])
		vote_lines = [
			f"- {to_label(str(r.get('judge_model', 'Unknown')))}: {str(r.get('choice', 'UNKNOWN')).upper()}"
			for r in sorted(rows, key=lambda x: str(x.get("judge_model", "")))
		]
		base["judge_model"] = " + ".join(sorted(selected_set))
		base["choice"] = final_choice
		base["reasoning"] = "\n".join(
			[
				"Ensemble majority voting over single judges.",
				"",
				"Votes:",
				*vote_lines,
				"",
				f"Final decision: {final_choice}",
			]
		)
		aggregated_rows.append(base)

	return aggregated_rows


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


def render_query_panel(sample: Dict[str, Any]) -> None:
	st.subheader("Selected Query")

	info_cols = st.columns(5)
	with info_cols[0]:
		render_hover_value("Dataset", sample.get("dataset", "Unknown"), max_chars=22)
	with info_cols[1]:
		render_hover_value("Database", sample.get("db_id", "Unknown"), max_chars=22)
	with info_cols[2]:
		render_hover_value("Query ID", sample.get("question_id", "-"), max_chars=22)
	with info_cols[3]:
		render_hover_value("Difficulty", sample.get("difficulty", "N/A"), max_chars=22)
	with info_cols[4]:
		render_hover_value("Complexity", sample.get("complexity", "N/A"), max_chars=22)

	with st.container(border=True):
		st.markdown("**NL Query**")
		render_hover_value("Text", sample.get("nl_query", "Not available"), max_chars=180)

		evidence = sample.get("evidence", "")
		if evidence:
			st.markdown("**Evidence**")
			render_hover_value("Text", evidence, max_chars=180)

		st.markdown("**Candidate SQL**")
		st.code(sample.get("candidate_sql", ""), language="sql")

		gt_sql = sample.get("ground_truth_sql", "")
		if gt_sql:
			st.markdown("**Ground Truth SQL**")
			st.code(gt_sql, language="sql")


def render_decision_panel(sample: Dict[str, Any]) -> None:
	st.subheader("Judge Decision")
	choice = sample.get("choice", "UNKNOWN")
	color = choice_color(choice)

	st.markdown(
		f"""
		<div style="
			border: 2px solid {color};
			border-radius: 12px;
			padding: 18px;
			text-align: center;
			font-size: 34px;
			font-weight: 700;
			color: {color};
			background: rgba(148, 163, 184, 0.08);
		">
			{choice}
		</div>
		""",
		unsafe_allow_html=True,
	)


def render_reasoning_panel(sample: Dict[str, Any]) -> None:
	st.subheader("Judge Reasoning")
	with st.container(border=True):
		reasoning = sample.get("reasoning", "")
		if reasoning:
			st.write(reasoning)
		else:
			st.info("Reasoning not available for this sample.")


def render_performance_panel(sample: Dict[str, Any]) -> None:
	st.subheader("Actual Performance")
	perf = sample.get("execution_vs_ground_truth")

	with st.container(border=True):
		if not isinstance(perf, dict):
			st.info("No execution-vs-ground-truth metrics available for this sample.")
			return

		# Prefer current metric names from pairwise comparisons; keep fallbacks for older files.
		schema_precision = perf.get("schema_precision", perf.get("precision"))
		schema_recall = perf.get("schema_recall", perf.get("recall"))
		cell_value_accuracy = perf.get("cell_value_accuracy", perf.get("acc_cell"))
		row_set_jaccard = perf.get("row_set_jaccard", perf.get("acc_row"))
		execution_accuracy = perf.get("execution_accuracy")
		f1_score = perf.get("f1_score")

		metric_cols_top = st.columns(3)
		metric_cols_top[0].metric("Schema Precision", fmt_float(schema_precision))
		metric_cols_top[1].metric("Schema Recall", fmt_float(schema_recall))
		metric_cols_top[2].metric("Cell Value Accuracy", fmt_float(cell_value_accuracy))

		metric_cols_bottom = st.columns(3)
		metric_cols_bottom[0].metric("Row Set Jaccard", fmt_float(row_set_jaccard))
		metric_cols_bottom[1].metric("Execution Accuracy", fmt_float(execution_accuracy))
		metric_cols_bottom[2].metric("F1 Score", fmt_float(f1_score))

		comparison_flag = perf.get("comparison_performed")
		if comparison_flag is True:
			st.success("Comparison performed successfully.")
		elif comparison_flag is False:
			st.warning("Comparison failed or was skipped.")

		if perf.get("fast_path"):
			st.caption(f"Fast path: {perf['fast_path']}")
		if perf.get("error"):
			st.error(f"Comparison error: {perf['error']}")


def main() -> None:
	project_root = Path(__file__).resolve().parent
	all_rows, candidate_models, judge_models = load_binary_data(str(project_root))

	st.title("Binary Judge Demo")
	st.caption("Interactive view of ACCEPT/REJECT judgments over candidate SQL queries.")

	if not all_rows:
		st.error(
			"No valid binary choice data found in the binary_choices folder. "
			"Add *_binary_choices.json files and rerun the app."
		)
		return

	known_models = sorted({*candidate_models, *judge_models})
	ensemble_options = [" + ".join(combo) for r in range(2, len(known_models) + 1) for combo in combinations(known_models, r)]

	left_col, right_col = st.columns([0.85, 2.35], gap="large")

	with left_col:
		with st.container(border=True):
			st.subheader("Selection")
			selected_ensemble: List[str] = []
			selected_candidate = st.selectbox(
				"Candidate model",
				options=candidate_models,
				format_func=to_label,
			)

			judge_mode = st.radio("Judge type", options=["Single", "Ensemble"], horizontal=True)
			if judge_mode == "Single":
				single_judge = st.selectbox(
					"Judge",
					options=judge_models,
					format_func=to_label,
				)
				selected_judge_key = compose_judge_key(judge_mode, single_judge, [])
			else:
				selected_ensemble = st.multiselect(
					"Judge ensemble models",
					options=known_models,
					format_func=to_label,
					default=known_models[:2] if len(known_models) >= 2 else known_models,
				)
				selected_judge_key = compose_judge_key(judge_mode, "", selected_ensemble)
				if selected_judge_key and selected_judge_key not in ensemble_options:
					st.warning("Choose at least two models to form an ensemble.")

			run_clicked = st.button("Run", type="primary", use_container_width=True)

	if judge_mode == "Single":
		filtered = [
			row
			for row in all_rows
			if row.get("candidate_model") == selected_candidate and row.get("judge_model") == selected_judge_key
		]
	else:
		filtered = build_ensemble_rows(all_rows, selected_candidate, selected_ensemble)

	if run_clicked:
		sampled = pick_random_result(filtered)
		st.session_state["binary_sample"] = sampled
		st.session_state["binary_filters"] = {
			"candidate_model": selected_candidate,
			"judge_model": selected_judge_key,
		}

	with left_col:
		st.caption(f"Matching rows: {len(filtered)}")

	sample = st.session_state.get("binary_sample")
	sample_filters = st.session_state.get("binary_filters", {})

	with right_col:
		if not filtered:
			st.warning(
				"No rows available for the selected candidate/judge combination. "
				"Try another judge or switch to Single mode."
			)

		if sample and sample_filters == {"candidate_model": selected_candidate, "judge_model": selected_judge_key}:
			top_left, top_right = st.columns([1, 2])
			with top_left:
				render_decision_panel(sample)
			with top_right:
				render_performance_panel(sample)

			render_query_panel(sample)
			render_reasoning_panel(sample)
		else:
			st.info("Select the models and click Run to sample one query and visualize the binary decision.")


if __name__ == "__main__":
	main()
