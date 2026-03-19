import hashlib
import json
import os
from datetime import datetime, timezone
from itertools import combinations

models = ["Deepseek-Chat", "Qwen2.5 Coder 32B", "Qwen3 Coder 30B", "Cogito 70b"]

# Map model names to existing fake result files.
RESULT_FILES = {
	"Deepseek-Chat": "results/deepseek-chat_results.json",
	"Qwen2.5 Coder 32B": "results/qwen2.5_coder_32b_results.json",
	"Qwen3 Coder 30B": "results/qwen3_coder_30b_results.json",
	"Cogito 70b": "results/cogito_70b_results.json",
}


def stable_noise(*parts, min_value=-1.0, max_value=1.0):
	"""Generate deterministic pseudo-random noise in [min_value, max_value]."""
	raw = "|".join(str(p) for p in parts)
	digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
	unit = int(digest[:12], 16) / float(16**12 - 1)
	return min_value + (max_value - min_value) * unit


def build_candidate_combinations(all_models):
	"""Return all model combinations with size >= 2, preserving model order."""
	combos = []
	for size in range(2, len(all_models) + 1):
		combos.extend(combinations(all_models, size))
	return combos


def load_scores_by_query():
	"""Load fake results and index query metadata/scores per dataset and id."""
	query_index = {}

	for model in models:
		file_path = RESULT_FILES[model]
		with open(file_path, "r", encoding="utf-8") as f:
			payload = json.load(f)

		for dataset_name, queries in payload.get("datasets", {}).items():
			dataset_bucket = query_index.setdefault(dataset_name, {})

			for row in queries:
				query_id = row["id"]
				entry = dataset_bucket.setdefault(
					query_id,
					{
						"id": query_id,
						"query": row.get("query", ""),
						"database": row.get("database", ""),
						"complexity": row.get("complexity", 0),
						"length": row.get("length", 0),
						"tables": row.get("tables", 0),
						"attributes": row.get("attributes", 0),
						"scores": {},
					},
				)

				metrics = row.get("metrics", {})
				exec_acc = float(metrics.get("Execution Accuracy", 0.0))
				exact_match = float(metrics.get("Exact Match", 0.0))

				# Use the average of available fake metrics as candidate quality score.
				entry["scores"][model] = round((exec_acc + exact_match) / 2.0, 4)

	return query_index


def normalize_scores_within_combo(candidates, scores):
	"""Normalize candidate scores inside one combination to [0, 1]."""
	raw = [scores[candidate] for candidate in candidates]
	low = min(raw)
	high = max(raw)
	if high == low:
		return {candidate: 0.5 for candidate in candidates}
	return {candidate: (scores[candidate] - low) / (high - low) for candidate in candidates}


def apply_controlled_selection_error(mode, dataset_name, query_id, candidates, ranked, complexity):
	"""Select winner with deterministic, complexity-aware error injection."""
	if len(ranked) == 1:
		return ranked[0][1]

	top_score = ranked[0][0]
	second_score = ranked[1][0]
	margin = max(0.0, top_score - second_score)

	base_error = 0.24 if mode == "single_selector" else 0.13
	complexity_boost = min(0.18, 0.05 * complexity)
	margin_protection = min(0.20, 0.25 * margin)
	error_rate = max(0.04, base_error + complexity_boost - margin_protection)

	error_trigger = stable_noise(
		"selection_error_trigger",
		mode,
		dataset_name,
		query_id,
		"+".join(candidates),
		min_value=0.0,
		max_value=1.0,
	)
	if error_trigger >= error_rate:
		return ranked[0][1]

	# When an error happens, prefer close alternatives instead of random weak picks.
	alternatives = ranked[1:]
	weights = []
	for index, (score, _) in enumerate(alternatives, start=1):
		gap = max(0.0, top_score - score)
		weight = 1.0 / (1.0 + 4.0 * gap + 0.25 * (index - 1))
		weights.append(max(0.02, weight))

	total_weight = sum(weights)
	pick_unit = stable_noise(
		"selection_error_pick",
		mode,
		dataset_name,
		query_id,
		"+".join(candidates),
		min_value=0.0,
		max_value=1.0,
	)
	target = pick_unit * total_weight

	running = 0.0
	for (score, candidate), weight in zip(alternatives, weights):
		running += weight
		if running >= target:
			return candidate

	return alternatives[-1][1]


def score_candidate_for_judge(judge, dataset_name, query_id, candidates, candidate, scores, mode):
	"""Compute one judge's score for a candidate with judge-specific and query-specific variation."""
	normalized = normalize_scores_within_combo(candidates, scores)
	base = normalized[candidate]

	# Judge-specific stable preference for each candidate model.
	affinity = stable_noise("affinity", judge, candidate, min_value=-0.45, max_value=0.45)

	# Query-sensitive preference to create meaningful per-query variability.
	query_signal = stable_noise(
		"query_signal",
		mode,
		judge,
		dataset_name,
		query_id,
		"+".join(candidates),
		candidate,
		min_value=-0.9,
		max_value=0.9,
	)

	if mode == "llms_ensemble":
		# Ensemble judges should be strongly quality-driven.
		ensemble_jitter = stable_noise(
			"ensemble_jitter",
			judge,
			dataset_name,
			query_id,
			"+".join(candidates),
			candidate,
			min_value=-0.4,
			max_value=0.4,
		)
		return 0.52 * base + 0.25 * affinity + 0.26 * query_signal + ensemble_jitter

	# Keep only mild self-bias so choices are not always identical.
	self_bias = 0.12 if candidate == judge else 0.0
	single_jitter = stable_noise(
		"single_jitter",
		judge,
		dataset_name,
		query_id,
		"+".join(candidates),
		candidate,
		min_value=-0.5,
		max_value=0.5,
	)

	# Single-judge mode stays diverse, but slightly more quality-driven than before.
	return 0.42 * base + 0.28 * affinity + 0.36 * query_signal + self_bias + single_jitter


def choose_with_single_judge(judge, dataset_name, query_id, candidates, scores, complexity):
	"""Pick one candidate model name for a single LLM judge."""
	ranked = []
	for candidate in candidates:
		total = score_candidate_for_judge(
			judge, dataset_name, query_id, candidates, candidate, scores, mode="single_selector"
		)
		ranked.append((total, candidate))

	# Highest score wins; tie-break with original model ordering.
	ranked.sort(key=lambda x: (x[0], -models.index(x[1])), reverse=True)
	return apply_controlled_selection_error(
		"single_selector", dataset_name, query_id, candidates, ranked, complexity
	)


def choose_with_ensemble(dataset_name, query_id, candidates, scores, complexity):
	"""Pick one candidate by majority vote over all judge models."""
	votes = {candidate: 0 for candidate in candidates}
	avg_total_scores = {candidate: 0.0 for candidate in candidates}
	normalized = normalize_scores_within_combo(candidates, scores)

	for judge in models:
		# Reuse judge scoring with different mode key to diversify ensemble internals.
		local_ranked = []
		for candidate in candidates:
			total = score_candidate_for_judge(
				judge, dataset_name, query_id, candidates, candidate, scores, mode="llms_ensemble"
			)
			local_ranked.append((total, candidate))
			avg_total_scores[candidate] += total

		local_ranked.sort(key=lambda x: (x[0], -models.index(x[1])), reverse=True)
		winner = local_ranked[0][1]
		votes[winner] += 1

	# Majority and quality should dominate, with only a small diversity effect.
	diversity_bonus = {
		candidate: stable_noise(
			"ensemble_diversity",
			dataset_name,
			query_id,
			"+".join(candidates),
			candidate,
			min_value=-0.04,
			max_value=0.04,
		)
		for candidate in candidates
	}

	full_combo_penalty = {
		candidate: (
			stable_noise(
				"full_combo_penalty",
				dataset_name,
				query_id,
				candidate,
				min_value=0.0,
				max_value=0.03,
			)
			if len(candidates) == len(models)
			else 0.0
		)
		for candidate in candidates
	}

	final_scores = {
		candidate: (
			votes[candidate]
			+ 0.90 * normalized[candidate]
			+ diversity_bonus[candidate]
			- full_combo_penalty[candidate]
			+ 1.25 * (avg_total_scores[candidate] / len(models))
		)
		for candidate in candidates
	}

	ranked = sorted(
		[(final_scores[c], c) for c in candidates],
		key=lambda x: (x[0], -models.index(x[1])),
		reverse=True,
	)

	return apply_controlled_selection_error(
		"llms_ensemble", dataset_name, query_id, candidates, ranked, complexity
	)


def generate_judge_output(judge_name, query_index, combos):
	"""Build output JSON for one single judge model."""
	output = {
		"judge": judge_name,
		"mode": "single_selector",
		"candidate_models": models,
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"datasets": {},
	}

	for dataset_name, by_id in query_index.items():
		rows = []
		for query_id in sorted(by_id.keys()):
			item = by_id[query_id]
			scores = item["scores"]
			combo_choices = {}

			for combo in combos:
				# Skip combinations with missing candidate scores.
				if not all(candidate in scores for candidate in combo):
					continue

				combo_label = " + ".join(combo)
				choice = choose_with_single_judge(
					judge_name,
					dataset_name,
					query_id,
					combo,
					scores,
					item["complexity"],
				)
				combo_choices[combo_label] = choice

			rows.append(
				{
					"id": item["id"],
					"query": item["query"],
					"database": item["database"],
					"complexity": item["complexity"],
					"length": item["length"],
					"tables": item["tables"],
					"attributes": item["attributes"],
					"choices": combo_choices,
				}
			)

		output["datasets"][dataset_name] = rows

	return output


def generate_ensemble_output(query_index, combos):
	"""Build output JSON for one ensemble selector over all judges."""
	output = {
		"judge": "LLMs Ensemble",
		"mode": "llms_ensemble",
		"judges": models,
		"candidate_models": models,
		"generated_at": datetime.now(timezone.utc).isoformat(),
		"datasets": {},
	}

	for dataset_name, by_id in query_index.items():
		rows = []
		for query_id in sorted(by_id.keys()):
			item = by_id[query_id]
			scores = item["scores"]
			combo_choices = {}

			for combo in combos:
				if not all(candidate in scores for candidate in combo):
					continue

				combo_label = " + ".join(combo)
				choice = choose_with_ensemble(
					dataset_name,
					query_id,
					combo,
					scores,
					item["complexity"],
				)
				combo_choices[combo_label] = choice

			rows.append(
				{
					"id": item["id"],
					"query": item["query"],
					"database": item["database"],
					"complexity": item["complexity"],
					"length": item["length"],
					"tables": item["tables"],
					"attributes": item["attributes"],
					"choices": combo_choices,
				}
			)

		output["datasets"][dataset_name] = rows

	return output


def main():
	os.makedirs("choices", exist_ok=True)

	query_index = load_scores_by_query()
	combos = build_candidate_combinations(models)

	# One file for each single LLM judge.
	for judge in models:
		output = generate_judge_output(judge, query_index, combos)
		output_path = f"fake_choices/{judge.replace(' ', '_').lower()}_single_selector_choices.json"
		with open(output_path, "w", encoding="utf-8") as f:
			json.dump(output, f, indent=2)
		print(f"Saved {output_path}")

	# One file for the ensemble of judges.
	ensemble_output = generate_ensemble_output(query_index, combos)
	ensemble_path = "fake_choices/llms_ensemble_selector_choices.json"
	with open(ensemble_path, "w", encoding="utf-8") as f:
		json.dump(ensemble_output, f, indent=2)
	print(f"Saved {ensemble_path}")


if __name__ == "__main__":
	main()