import argparse
import os
import re
import torch
from tqdm import tqdm

from graph_filter import GraphRAGRefiner
from exemplar_retriever import TrainingExemplarRetriever


def _parse_subject_id(path):
    match = re.search(r"subject_(\d+)_top_words\.pt$", os.path.basename(path))
    if not match:
        return None
    return int(match.group(1))


def _collect_subject_files(input_dir):
    files = []
    for name in sorted(os.listdir(input_dir)):
        if name.startswith("subject_") and name.endswith("_top_words.pt"):
            files.append(os.path.join(input_dir, name))
    return files


def _extract_raw_bow(sample):
    top_words = sample.get("top_words", [])
    raw_bow = [entry.get("word") for entry in top_words if entry.get("word")]
    return raw_bow


def _extract_raw_bow_from_list(top_words):
    if not top_words:
        return []
    return [entry.get("word") for entry in top_words if entry.get("word")]


def prune_subject_file(path, graph_rag, top_n_facts):
    samples = torch.load(path, weights_only=False)
    if not isinstance(samples, list):
        raise ValueError(f"Unsupported format in {path}; expected a list of samples.")

    pruned_records = []
    for sample in tqdm(samples, desc=os.path.basename(path)):
        raw_bow = _extract_raw_bow(sample)
        pruned_bow, relational_facts = graph_rag.retrieve_and_filter_subgraph(
            seed_words=raw_bow, top_n_facts=top_n_facts
        )
        updated_sample = dict(sample)
        updated_sample["pruned_bow"] = pruned_bow
        updated_sample["relational_facts"] = relational_facts
        pruned_records.append(updated_sample)

    return pruned_records


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prune BoW using GraphRAGRefiner and attach exemplar captions."
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Directory containing subject_XX_top_words.pt files.",
    )
    parser.add_argument(
        "--test_path",
        type=str,
        default=None,
        help="Merged test .pt file with top_words and eeg_clip_latent.",
    )
    parser.add_argument(
        "--train_latents_path",
        type=str,
        default=None,
        help="Merged training latent .pt file (from build_train_latents.py).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save pruned subject files (defaults to input_dir).",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Output path for merged test file (defaults to overwriting --test_path).",
    )
    parser.add_argument(
        "--conceptnet_db_path",
        type=str,
        default="data/conceptnet_local.db",
        help="Path to conceptnet_local.db for GraphRAGRefiner.",
    )
    parser.add_argument(
        "--top_n_facts",
        type=int,
        default=5,
        help="Number of relational facts to retrieve (used by GraphRAGRefiner).",
    )
    parser.add_argument(
        "--top_n_exemplars",
        type=int,
        default=2,
        help="Number of exemplar captions to retrieve per sample.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()

    graph_rag = GraphRAGRefiner(db_path=args.conceptnet_db_path)
    if args.test_path:
        if not args.train_latents_path:
            raise ValueError("Provide --train_latents_path when using --test_path.")
        test_data = torch.load(args.test_path, weights_only=False)
        if not isinstance(test_data, dict):
            raise ValueError(f"Unsupported format in {args.test_path}; expected a dict.")
        if "top_words" not in test_data or "eeg_clip_latent" not in test_data:
            raise ValueError("test file missing required fields: top_words and eeg_clip_latent.")

        exemplar_retriever = TrainingExemplarRetriever(
            training_data_path=args.train_latents_path, device=args.device
        )

        total = len(test_data["top_words"])
        pruned_bows = [None] * total
        raw_bows = [None] * total
        relational_facts_list = [None] * total
        retrieved_exemplars = [None] * total

        for idx in tqdm(range(total), desc="Pruning + exemplar retrieval"):
            top_words = test_data["top_words"][idx]
            raw_bow = _extract_raw_bow_from_list(top_words)
            pruned_bow, relational_facts = graph_rag.retrieve_and_filter_subgraph(
                seed_words=raw_bow, top_n_facts=args.top_n_facts
            )
            eeg_latent = test_data["eeg_clip_latent"][idx]
            if not torch.is_tensor(eeg_latent):
                eeg_latent = torch.from_numpy(eeg_latent)
            exemplars = exemplar_retriever.retrieve_top_exemplars(
                refined_latent=eeg_latent, top_n=args.top_n_exemplars
            )

            raw_bows[idx] = raw_bow
            pruned_bows[idx] = pruned_bow
            relational_facts_list[idx] = relational_facts
            retrieved_exemplars[idx] = exemplars

        test_data["raw_bow"] = raw_bows
        test_data["pruned_bow"] = pruned_bows
        test_data["relational_facts"] = relational_facts_list
        test_data["retrieved_exemplars"] = retrieved_exemplars

        output_path = args.output_path or args.test_path
        torch.save(test_data, output_path)
        print(f"Saved {total} samples to {output_path}")
        return

    if not args.input_dir:
        raise ValueError("Provide --input_dir for per-subject pruning or --test_path for merged mode.")

    output_dir = args.output_dir or args.input_dir
    os.makedirs(output_dir, exist_ok=True)

    subject_files = _collect_subject_files(args.input_dir)
    if not subject_files:
        raise FileNotFoundError(f"No subject_XX_top_words.pt files in {args.input_dir}.")

    for path in subject_files:
        subject_id = _parse_subject_id(path)
        pruned_records = prune_subject_file(path, graph_rag, args.top_n_facts)
        out_name = (
            f"subject_{subject_id:02d}_pruned_bow_5_words.pt"
            if subject_id is not None
            else f"{os.path.splitext(os.path.basename(path))[0]}_pruned_bow.pt"
        )
        out_path = os.path.join(output_dir, out_name)
        torch.save(pruned_records, out_path)
        print(f"Saved {len(pruned_records)} samples to {out_path}")


if __name__ == "__main__":
    main()
