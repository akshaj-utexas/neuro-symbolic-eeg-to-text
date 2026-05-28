import argparse
import os
import subprocess
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="THINGS SYNAPSE Pipeline")

    # Mode Selection
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "inference"],
        required=True,
        help="train: Train text mapper | inference: Top-words → GraphRAG → LLM",
    )

    # Shared paths
    parser.add_argument("--vocab_corpus", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")

    # ===== Training (Text Mapper) =====
    parser.add_argument("--encoded_eeg_dataset", type=str, default=None)
    parser.add_argument("--output", type=str, default="output/text-mapper")
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--hidden_dim", type=int, default=1024)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--use_scaling", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)

    # ===== Inference (Top-words + GraphRAG + LLM) =====
    parser.add_argument("--test_path", type=str, default=None)
    parser.add_argument("--encoder_path", type=str, default=None)
    parser.add_argument("--text_mapper_ckpt", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="output/text-mapper-inference")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--use_precomputed_latents", action="store_true")
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_argument("--time_low", type=float, default=20)
    parser.add_argument("--time_high", type=float, default=460)

    # Graph refinement
    parser.add_argument("--train_latents_path", type=str, default=None)
    parser.add_argument("--conceptnet_db_path", type=str, default="data/conceptnet_local.db")
    parser.add_argument("--top_n_facts", type=int, default=5)
    parser.add_argument("--top_n_exemplars", type=int, default=2)
    parser.add_argument("--pruned_output_path", type=str, default=None)

    # LLM decoding + metrics
    parser.add_argument(
        "--model",
        type=str,
        choices=["qwen2.5", "llama", "chatgpt", "gemini", "all"],
        default="chatgpt",
    )
    parser.add_argument("--llm_output_dir", type=str, default="output/llm-prompts")
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument(
        "--ablations",
        type=str,
        choices=["full", "no_exemplars", "sense_baseline", "both", "all"],
        default="both",
    )

    return parser.parse_args()


def _run(cmd, label):
    print(f"\n--- {label} ---")
    subprocess.run(cmd, check=True)


def _require(arg, name):
    if not arg:
        raise ValueError(f"Missing required argument: {name}")


def _resolve_output_paths(test_path, output_dir, output_path, pruned_output_path):
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(test_path))[0]
    inferred_output = output_path or os.path.join(output_dir, f"{base}_top_words.pt")
    pruned_output = pruned_output_path or os.path.join(output_dir, f"{base}_pruned.pt")
    return inferred_output, pruned_output


def main():
    args = parse_args()
    repo_root = os.path.dirname(os.path.abspath(__file__))

    if args.mode == "train":
        _require(args.encoded_eeg_dataset, "--encoded_eeg_dataset")
        _require(args.vocab_corpus, "--vocab_corpus")

        cmd = [
            sys.executable,
            os.path.join(repo_root, "src", "train_text_mapper.py"),
            "--encoded_eeg_dataset",
            args.encoded_eeg_dataset,
            "--vocab_corpus",
            args.vocab_corpus,
            "--output",
            args.output,
            "--num_epochs",
            str(args.num_epochs),
            "--batch_size",
            str(args.batch_size),
            "--learning_rate",
            str(args.learning_rate),
            "--weight_decay",
            str(args.weight_decay),
            "--hidden_dim",
            str(args.hidden_dim),
            "--focal_alpha",
            str(args.focal_alpha),
            "--focal_gamma",
            str(args.focal_gamma),
            "--val_split",
            str(args.val_split),
            "--seed",
            str(args.seed),
            "--device",
            args.device,
        ]
        if args.use_scaling:
            cmd.append("--use_scaling")
        if args.max_samples is not None:
            cmd.extend(["--max_samples", str(args.max_samples)])

        _run(cmd, "Training text mapper (SimilarityRefiner)")
        return

    if args.mode == "inference":
        _require(args.test_path, "--test_path")
        _require(args.encoder_path, "--encoder_path")
        _require(args.text_mapper_ckpt, "--text_mapper_ckpt")
        _require(args.train_latents_path, "--train_latents_path")

        inferred_output, pruned_output = _resolve_output_paths(
            args.test_path, args.output_dir, args.output_path, args.pruned_output_path
        )

        cmd = [
            sys.executable,
            os.path.join(repo_root, "src", "infer_text_mapper.py"),
            "--test_path",
            args.test_path,
            "--encoder_path",
            args.encoder_path,
            "--text_mapper_ckpt",
            args.text_mapper_ckpt,
            "--output_path",
            inferred_output,
            "--top_k",
            str(args.top_k),
            "--batch_size",
            str(args.batch_size),
            "--time_low",
            str(args.time_low),
            "--time_high",
            str(args.time_high),
            "--device",
            args.device,
        ]
        if args.vocab_corpus:
            cmd.extend(["--vocab_corpus", args.vocab_corpus])
        if args.use_precomputed_latents:
            cmd.append("--use_precomputed_latents")

        _run(cmd, "Inference: EEG → top-words")

        cmd = [
            sys.executable,
            os.path.join(repo_root, "src", "prune_subject_bow.py"),
            "--test_path",
            inferred_output,
            "--train_latents_path",
            args.train_latents_path,
            "--conceptnet_db_path",
            args.conceptnet_db_path,
            "--top_n_facts",
            str(args.top_n_facts),
            "--top_n_exemplars",
            str(args.top_n_exemplars),
            "--output_path",
            pruned_output,
            "--device",
            args.device,
        ]
        _run(cmd, "GraphRAG pruning + exemplars")

        cmd = [
            sys.executable,
            os.path.join(repo_root, "src", "run_pruned_bow_llm.py"),
            "--input_path",
            pruned_output,
            "--output_dir",
            args.llm_output_dir,
            "--model",
            args.model,
            "--num_workers",
            str(args.num_workers),
            "--ablations",
            args.ablations,
        ]
        if args.num_samples is not None:
            cmd.extend(["--num_samples", str(args.num_samples)])

        _run(cmd, "LLM decoding + metrics")
        return

    raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()
