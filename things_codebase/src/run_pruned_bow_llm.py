import argparse
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import torch
from tqdm import tqdm

from llm_client import LLMManager
from metrics import evaluate_and_save_metrics


def _resolve_model(model):
    if model == "qwen2.5":
        return "together", "Qwen/Qwen2.5-7B-Instruct-Turbo"
    if model == "llama":
        return "together", "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
    if model == "chatgpt":
        return "openai", "gpt-4o-mini-2024-07-18"
    if model == "gemini":
        return "google", "gemini-2.5-flash-lite"
    raise ValueError(f"Model {model} not found")


def _flatten_for_csv(records):
    csv_flattened_records = []
    for rec in records:
        flat_rec = {
            "subject": rec.get("subject"),
            "raw_bow": ", ".join(rec.get("raw_bow", [])),
            "pruned_bow": ", ".join(rec.get("pruned_bow", [])),
            "caption": rec.get("caption", ""),
            "relational_facts": rec.get("relational_facts", ""),
            "retrieved_exemplars": " | ".join(rec.get("retrieved_exemplars", [])),
            "generated_caption": rec.get("generated_caption", ""),
        }
        csv_flattened_records.append(flat_rec)
    return csv_flattened_records


def _extract_caption(text_item):
    if isinstance(text_item, (list, tuple)):
        return text_item[0] if text_item else ""
    if hasattr(text_item, "shape"):
        return text_item[0] if text_item.shape else str(text_item)
    return str(text_item)


def _normalize_dataset(dataset, num_samples=None):
    if isinstance(dataset, list):
        return dataset[:num_samples] if num_samples else dataset
    if not isinstance(dataset, dict):
        raise ValueError("Expected dict-format test_from_train.pt or list of samples.")

    base_len = len(dataset.get("text", []))
    if num_samples is not None:
        base_len = min(base_len, num_samples)

    samples = []
    for i in range(base_len):
        exemplars = dataset.get("retrieved_exemplars", [None])[i] if "retrieved_exemplars" in dataset else []
        exemplar_caps = []
        if isinstance(exemplars, list):
            for ex in exemplars[:2]:
                if isinstance(ex, dict):
                    exemplar_caps.append(ex.get("caption", ""))
                else:
                    exemplar_caps.append(str(ex))
        relational_facts = dataset.get("relational_facts", [None])[i] if "relational_facts" in dataset else []
        if isinstance(relational_facts, list):
            relational_facts = "; ".join([str(fact) for fact in relational_facts])
        sample = {
            "text": dataset.get("text", [None])[i],
            "caption": _extract_caption(dataset.get("text", [""])[i])
            if "text" in dataset
            else "",
            "top_words": dataset.get("top_words", [None])[i]
            if "top_words" in dataset
            else [],
            "raw_bow": dataset.get("raw_bow", [None])[i]
            if "raw_bow" in dataset
            else [],
            "pruned_bow": dataset.get("pruned_bow", [None])[i]
            if "pruned_bow" in dataset
            else [],
            "relational_facts": relational_facts or "",
            "retrieved_exemplars": exemplar_caps,
            "subject": dataset.get("subject", [None])[i] if "subject" in dataset else None,
            "img": dataset.get("img", [None])[i] if "img" in dataset else None,
        }
        samples.append(sample)
    return samples


def _generate_parallel(samples, provider, model_name, ablation_version, num_workers):
    thread_local = threading.local()

    def get_manager():
        if not hasattr(thread_local, "manager"):
            thread_local.manager = LLMManager(provider=provider, model_name=model_name)
        return thread_local.manager

    def worker(idx, sample):
        manager = get_manager()
        generated = manager.generate(sample, ablation_version=ablation_version)
        output_record = sample.copy()
        output_record["generated_caption"] = generated
        return idx, output_record

    results = [None] * len(samples)
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(worker, idx, sample) for idx, sample in enumerate(samples)
        ]
        for future in tqdm(as_completed(futures), total=len(futures)):
            idx, output_record = future.result()
            results[idx] = output_record
    return results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LLM decoding on pruned BoW datasets with optional relational facts."
    )
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen2.5", "llama", "chatgpt", "gemini", "all"],
        required=True,
    )
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument(
        "--ablations",
        type=str,
        choices=["full", "no_exemplars", "sense_baseline", "both", "all"],
        default="both",
        help="Which ablation to run: full, no_exemplars, sense_baseline, both, or all.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    dataset = torch.load(args.input_path, weights_only=False)
    test_subset = _normalize_dataset(dataset, num_samples=args.num_samples)

    if args.ablations == "full":
        ablations = [None]
    elif args.ablations == "no_exemplars":
        ablations = ["no_exemplars"]
    elif args.ablations == "sense_baseline":
        ablations = ["sense_baseline"]
    elif args.ablations == "all":
        ablations = [None, "no_exemplars", "sense_baseline"]
    else:
        ablations = [None, "no_exemplars"]
    models = (
        ["qwen2.5", "llama", "chatgpt", "gemini"]
        if args.model == "all"
        else [args.model]
    )
    base_name = os.path.splitext(os.path.basename(args.input_path))[0]

    for model in models:
        provider, model_name = _resolve_model(model)
        for ablation in ablations:
            label_tag = ablation if ablation else "full"
            csv_path = os.path.join(
                args.output_dir, f"{base_name}_{model}_{label_tag}.csv"
            )

            results = _generate_parallel(
                test_subset, provider, model_name, ablation, args.num_workers
            )
            csv_records = _flatten_for_csv(results)
            pd.DataFrame(csv_records).to_csv(csv_path, index=False)
            print(f"Saved outputs to {csv_path}")
            evaluate_and_save_metrics(csv_path, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
