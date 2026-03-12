import json
import random
from itertools import combinations
import os

# Set seed for reproducibility
random.seed(42)

# Database names and their corresponding files
datasets = {
    "BIRD Training": "datasets/bird_training_queries.json",
    "BIRD Developer": "datasets/bird_developer_queries.json",
    "SPIDER": "datasets/spider_queries.json"
}

# Model performance averages
model_performance = {
    "Claude": (80, 85),      # avg 82.5%
    "GPT": (75, 80),         # avg 77.5%
    "Cogito 70b": (70, 75),  # avg 72.5%
    "Llama 70b": (65, 70)    # avg 67.5%
}

models = list(model_performance.keys())
metrics = ["Execution Accuracy", "Exact Match", "TDEX", "Ensemble", "LLMs as a Judge"]


def generate_metric_value(model, metric_name, query_complexity):
    """Generate a metric value based on model performance and query complexity"""
    min_perf, max_perf = model_performance[model]
    
    # Adjust performance based on complexity (0-3)
    complexity_penalty = query_complexity * 3
    adjusted_min = max(0, min_perf - complexity_penalty)
    adjusted_max = max(0, max_perf - complexity_penalty)
    
    # Add some variation
    value = random.uniform(adjusted_min, adjusted_max)
    
    # Ensure value is between 0 and 100
    return round(min(100, max(0, value)), 2)


def generate_tdex_metric(query_complexity):
    """Generate TDEX metric values for all models"""
    return {model: generate_metric_value(model, "TDEX", query_complexity) 
            for model in models}


def generate_llms_judge_metric(query_complexity):
    """Generate LLMs as a Judge metric values for all models"""
    return {model: generate_metric_value(model, "LLMs as a Judge", query_complexity) 
            for model in models}


def generate_ensemble_metric(query_complexity):
    """Generate Ensemble metric values for all model combinations"""
    ensemble_values = {}
    
    # Single models
    for model in models:
        ensemble_values[model] = generate_metric_value(model, "Ensemble", query_complexity)
    
    # Pairs of models
    for pair in combinations(models, 2):
        key = " + ".join(pair)
        # Ensemble typically performs better than individual models
        avg_perf = sum([model_performance[m][0] + model_performance[m][1] for m in pair]) / (2 * len(pair))
        complexity_penalty = query_complexity * 2.5
        value = random.uniform(max(0, avg_perf - complexity_penalty), 
                               min(100, avg_perf + 5 - complexity_penalty))
        ensemble_values[key] = round(value, 2)
    
    # Triplets of models
    for triplet in combinations(models, 3):
        key = " + ".join(triplet)
        avg_perf = sum([model_performance[m][0] + model_performance[m][1] for m in triplet]) / (2 * len(triplet))
        complexity_penalty = query_complexity * 2
        value = random.uniform(max(0, avg_perf - complexity_penalty), 
                               min(100, avg_perf + 7 - complexity_penalty))
        ensemble_values[key] = round(value, 2)
    
    # All models
    key = " + ".join(models)
    avg_perf = sum([model_performance[m][0] + model_performance[m][1] for m in models]) / (2 * len(models))
    complexity_penalty = query_complexity * 1.5
    value = random.uniform(max(0, avg_perf - complexity_penalty), 
                           min(100, avg_perf + 10 - complexity_penalty))
    ensemble_values[key] = round(value, 2)
    
    return ensemble_values


def generate_results_for_query(query, model):
    """Generate all metric results for a single query"""
    complexity = query.get("complexity", 0)
    
    results = {
        "id": query["id"],
        "query": query["query"],
        "database": query["database"],
        "complexity": complexity,
        "length": query.get("length", 0),
        "tables": query.get("tables", 1),
        "attributes": query.get("attributes", 3),
        "metrics": {
            "Execution Accuracy": generate_metric_value(model, "Execution Accuracy", complexity),
            "Exact Match": generate_metric_value(model, "Exact Match", complexity),
            "TDEX": generate_tdex_metric(complexity),
            "Ensemble": generate_ensemble_metric(complexity),
            "LLMs as a Judge": generate_llms_judge_metric(complexity)
        }
    }
    
    return results


def main():
    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)
    
    # For each model, generate a results file
    for model in models:
        print(f"Generating results for {model}...")
        
        model_results = {
            "model": model,
            "datasets": {}
        }
        
        # Load queries from each dataset
        for dataset_name, dataset_file in datasets.items():
            print(f"  Processing {dataset_name}...")
            
            with open(dataset_file, 'r') as f:
                queries = json.load(f)
            
            # Generate results for each query
            dataset_results = []
            for query in queries:
                query_results = generate_results_for_query(query, model)
                dataset_results.append(query_results)
            
            model_results["datasets"][dataset_name] = dataset_results
        
        # Save results to file
        output_file = f"results/{model.replace(' ', '_').lower()}_results.json"
        with open(output_file, 'w') as f:
            json.dump(model_results, f, indent=2)
        
        print(f"  Saved to {output_file}")
    
    print("\nAll results generated successfully!")


if __name__ == "__main__":
    main()

